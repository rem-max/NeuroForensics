import copy
import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from security.utils.partial_models_adaptive import (
    CNNAdaptivePartialModel,
    LeNetAdaptivePartialModel,
    VGGAdaptivePartialModel,
    ResNetAdaptivePartialModel
)
from flcore.trainmodel.resnet import BasicBlock
from system.security.defense.NeuroForensics import NeuroForensics
from security.defense.gradient_clipping import GradientClipping
from security.defense.robust_aggregation import RobustAggregation
from security.defense.flame import FLAMEDefense
from security.attack.model_replacement import ModelReplacementAttack
import torch
from security.defense.AlignIns import AlignIns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.malicious_blacklist = set()

        # Attack initialization
        if hasattr(self, 'attack_type') and self.attack_type == 'model_replacement':
            print("Initializing Model Replacement attack.")
            self.model_replacement_attacker = ModelReplacementAttack(self.args)
        else:
            self.model_replacement_attacker = None

        if hasattr(self, 'defense_type') and self.defense_type == 'neuroforensics' and getattr(args, 'adaptive_threshold', False):
            from security.defense.adaptive_threshold import AdaptiveThresholdManager
            print("Initializing Adaptive Threshold for NeuroForensics defense...")
            self.adaptive_threshold = AdaptiveThresholdManager(
                strategy=getattr(args, 'adaptive_strategy', 'historical_3sigma'),
                initial_threshold=getattr(args, 'defense_threshold', 2.0),
                history_window=getattr(args, 'adaptive_history_window', 10),
                variance_threshold=getattr(args, 'adaptive_sigma_threshold', 5.0),
                top_k_for_threshold=getattr(args, 'adaptive_top_k', 3)
            )
            print(f"Adaptive threshold enabled with historical 3-sigma strategy")
        else:
            self.adaptive_threshold = None
        
        if hasattr(self, 'defense_type') and self.defense_type == 'gradient_clipping':
            print("Initializing Gradient Clipping defense...")
            self.gradient_clipper = GradientClipping(
                max_norm=getattr(args, 'grad_clip_max_norm', 1.0),
                norm_type=getattr(args, 'grad_clip_norm_type', 2.0),
                detect_threshold=getattr(args, 'grad_clip_detect_threshold', 2.0),
                device=self.device
            )
            print(f"Gradient Clipping enabled (max_norm={self.gradient_clipper.max_norm}, "
                  f"norm_type={self.gradient_clipper.norm_type})")
        else:
            self.gradient_clipper = None
        
        if hasattr(self, 'defense_type') and self.defense_type == 'robust_aggregation':
            print("Initializing Robust Aggregation defense...")
            self.robust_aggregator = RobustAggregation(
                aggregation_method=getattr(args, 'robust_agg_method', 'median'),
                trim_ratio=getattr(args, 'robust_trim_ratio', 0.1),
                detect_threshold=getattr(args, 'robust_detect_threshold', 2.0),
                krum_f=getattr(args, 'robust_krum_f', 4),
                multi_krum_k=getattr(args, 'robust_krum_k', 5),
                device=self.device
            )
            print(f"Robust Aggregation enabled (method={self.robust_aggregator.aggregation_method}, "
                  f"trim_ratio={self.robust_aggregator.trim_ratio}, "
                  f"krum_f={self.robust_aggregator.krum_f}, multi_krum_k={self.robust_aggregator.multi_krum_k})")
        else:
            self.robust_aggregator = None
        
        if hasattr(self, 'defense_type') and self.defense_type == 'flame':
            print("Initializing FLAME defense...")
            self.flame_defender = FLAMEDefense(
                min_cluster_ratio=getattr(args, 'flame_min_cluster_ratio', 0.5),
                noise_lambda=getattr(args, 'flame_noise_lambda', 0.000012),
                clip_threshold=getattr(args, 'defense_threshold', 3.0)
            )
            print(f"FLAME enabled (min_cluster_ratio={self.flame_defender.min_cluster_ratio}, "
                  f"noise_lambda={self.flame_defender.noise_lambda})")
        else:
            self.flame_defender = None

        if hasattr(self, 'defense_type') and self.defense_type == 'alignins':
            print("Initializing AlignIns defense...")
            self.alignins_defense = AlignIns(
                lambda_s=getattr(args, 'alignins_lambda_s', 1.0),
                lambda_c=getattr(args, 'alignins_lambda_c', 1.0),
                sparsity=getattr(args, 'alignins_sparsity', 0.3),
                seed=getattr(args, 'seed', 0)
            )
            print(f"AlignIns initialized with lambda_s={self.alignins_defense.lambda_s}, "
                  f"lambda_c={self.alignins_defense.lambda_c}, sparsity={self.alignins_defense.sparsity}")
        else:
            self.alignins_defense = None

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            
            self.current_round = i

            self.selected_clients = self.select_clients()

            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()

            if self.defense_type == 'gradient_clipping' and len(self.uploaded_models) > 0 :
                print(f"\n--- [Round {i}] Executing Gradient Clipping Defense ---")
                
                suspicious_clients = self.gradient_clipper.detect_suspicious_clients(
                    self.selected_clients, self.uploaded_ids, self.uploaded_models, self.global_model
                )
                
                if len(suspicious_clients) > 0:
                    print(f"Detected {len(suspicious_clients)} suspicious clients with abnormal gradients:")
                    for cid in suspicious_clients:
                        print(f"  - Client {cid}: Added to blacklist")
                        self.malicious_blacklist.add(cid)
                else:
                    print("No suspicious clients detected in this round.")
                
                print(f"--- Blacklist now contains {len(self.malicious_blacklist)} client(s). ---")
                
                print(f"\n--- Applying gradient clipping to all uploaded models ---")
                self.uploaded_models = self.gradient_clipper.clip_model_updates(
                    self.uploaded_models, self.global_model
                )
                print(f"--- Gradient clipping applied ---")
            
            elif self.defense_type == 'robust_aggregation' and len(self.uploaded_models) > 0 :
                print(f"\n--- [Round {i}] Executing Robust Aggregation Defense ---")
                
                suspicious_clients = self.robust_aggregator.detect_suspicious_clients(
                    self.selected_clients, self.uploaded_ids, self.uploaded_models, self.global_model
                )
                
                if len(suspicious_clients) > 0:
                    print(f"Detected {len(suspicious_clients)} suspicious clients with abnormal gradient distributions:")
                    for cid in suspicious_clients:
                        print(f"  - Client {cid}: Added to blacklist")
                        self.malicious_blacklist.add(cid)
                else:
                    print("No suspicious clients detected in this round.")
                
                print(f"--- Blacklist now contains {len(self.malicious_blacklist)} client(s). ---")
            
            elif self.defense_type == 'neuroforensics' and len(self.uploaded_models) > 0 and i < 15:
          
                if self.adaptive_threshold is not None:
                    self.defense_threshold = self.adaptive_threshold.start_round(i)
                    print(f"--- Adaptive threshold updated for round {i}: {self.defense_threshold:.4f} ---")
                
                print(
                    f"\n--- [Round {i}] Executing NeuroForensics Defense to update blacklist (Threshold={self.defense_threshold}) ---")
                AdaptiveModelClass = None
                if self.model_str == "CNN":
                    AdaptiveModelClass = CNNAdaptivePartialModel
                elif self.model_str == "LeNet":
                    AdaptiveModelClass = LeNetAdaptivePartialModel
                elif self.model_str == "VGG16":
                    AdaptiveModelClass = VGGAdaptivePartialModel
                elif self.model_str in ["ResNet10", "ResNet18"]:
                    AdaptiveModelClass = ResNetAdaptivePartialModel

                multi_level_positions = None
                if hasattr(self, 'multi_level_positions') and self.multi_level_positions:
                    multi_level_positions = [int(x.strip()) for x in self.multi_level_positions.split(',')]
                
                neuro_forensics_detector = NeuroForensics(
                    seed=int(time.time()),
                    enable_multi_level=getattr(self, 'enable_multi_level', False),
                    multi_level_positions=multi_level_positions,
                    fusion_strategy=getattr(self, 'fusion_strategy', 'weighted_average'),
                    use_similarity_detection=getattr(self.args, 'neuroforensics_use_similarity', False),
                    similarity_metric=getattr(self.args, 'neuroforensics_similarity_metric', 'cosine')
                )
                if 'MNIST' in self.dataset or 'FashionMNIST' in self.dataset:
                    shape = (1, 1, 28, 28)
                elif 'Cifar10' in self.dataset or 'Cifar100' in self.dataset:
                    shape = (1, 3, 32, 32)
                else:
            
                    print(
                        f"Warning: Unknown dataset '{self.dataset}' for defense model shape. Defaulting to CIFAR-10 shape.")
                    shape = (1, 3, 32, 32)

                matp_for_visualization = {} 
                malicious_matp = None
                malicious_client_id = None
                
                client_matps_for_similarity = {}  # {client_id: Mat_p}
                client_m_primes = {}  # {client_id: M'}，
                
                use_similarity_mode = hasattr(neuro_forensics_detector, 'use_similarity_detection') and neuro_forensics_detector.use_similarity_detection

                for client_id, client_model_state in zip(self.uploaded_ids, self.uploaded_models):
                    try:
                        if self.model_str in ["ResNet10", "ResNet18"]:
                         
                            block_type = BasicBlock
                            if self.model_str == "ResNet10":
                                layer_config = [1, 1, 1, 1]  
                            else:  # ResNet18
                                layer_config = [2, 2, 2, 2]  
                            classifier_part = AdaptiveModelClass(
                                block=block_type,
                                layers=layer_config,
                                num_classes=self.num_classes,
                                inspect_layer_position=self.lsep_layer,
                                original_input_img_shape=shape
                            ).to(self.device)
                        else:
                            classifier_part = AdaptiveModelClass(  # or VGGAdaptivePartialModel
                                num_classes=self.num_classes,
                                inspect_layer_position=self.lsep_layer,
                                original_input_img_shape=shape
                            ).to(self.device)
                        classifier_part.load_state_dict(client_model_state, strict=False)

                        schedule = {'device': self.device, 'num_classes': self.num_classes, 
                                   'noniid_mode': True}  
                        results = neuro_forensics_detector.test(model=classifier_part, dataset=self.test_data,
                                                         schedule=schedule)
                        
                        if use_similarity_mode:
                          
                            print(f"\n[Similarity Mode] Client {client_id}: Extracting Mat_p only...")
                            mat_p = results.get('Mat_p', None)
                            if mat_p is not None:
                                if isinstance(mat_p, torch.Tensor):
                                    client_matps_for_similarity[client_id] = mat_p.cpu().numpy()
                                else:
                                    client_matps_for_similarity[client_id] = mat_p
                                print(f"  Mat_p extracted, shape: {client_matps_for_similarity[client_id].shape}")
                            else:
                                print(f"  ❌ Failed to extract Mat_p for client {client_id}")
                            
                            continue
                        
                        else:
                            m_prime_original = results['M_prime']  
                            m_prime = results.get('M_prime_enhanced', m_prime_original)  
                            综合分数 = results.get('综合分数', m_prime)  
                            v_features = results.get('v_features', {})  
                            
                            print(f"\n[NeuroForensics Non-IID Enhanced] Client {client_id}:")
                            print(f"  原始M' = {m_prime_original:.4f}")
                            print(f"  增强M' = {m_prime:.4f} (综合分数: {综合分数:.4f}/10)")
                            print(f"  v向量特征:")
                            print(f"    - 峰度 (kurtosis): {v_features.get('kurtosis', 0):.4f}")
                            print(f"    - 归一化熵: {v_features.get('normalized_entropy', 0):.4f}")
                            print(f"    - 最大值占比: {v_features.get('max_ratio', 0):.4f}")
                            print(f"    - 稳健Z-score: {v_features.get('robust_zscore', 0):.4f}")

                            mat_p = results.get('Mat_p', None)
                            mat_p_original = results.get('Mat_p_original', None)  
                            
                            if mat_p is not None:
                                if isinstance(mat_p, torch.Tensor):
                                    client_matps_for_similarity[client_id] = mat_p.cpu().numpy()
                                else:
                                    client_matps_for_similarity[client_id] = mat_p
                            client_m_primes[client_id] = m_prime
                            
                            if mat_p is not None:
                              
                                if isinstance(mat_p, torch.Tensor):
                                    Mat_p_check = mat_p.cpu().numpy()
                                else:
                                    Mat_p_check = mat_p
                                
                                client_type = "🔴 MALICIOUS" if (hasattr(self, 'malicious_client_ids') and client_id in self.malicious_client_ids) else "🔵 BENIGN"
                                
                                print(f"\n{'='*80}")
                                print(f"[Round {i}] Client {client_id} ({client_type}) - Mat_p Matrix Analysis")
                                print(f"{'='*80}")
                                
                                if mat_p_original is not None:
                                    if isinstance(mat_p_original, torch.Tensor):
                                        Mat_p_orig = mat_p_original.cpu().numpy()
                                    else:
                                        Mat_p_orig = mat_p_original
                                
                                if i == 5:
                                    matp_for_visualization[client_id] = mat_p  
                                    
                                    if hasattr(self, 'malicious_client_ids') and client_id in self.malicious_client_ids:
                                        if malicious_matp is None:  
                                            malicious_matp = mat_p  
                                            malicious_client_id = client_id
                                            print(f"\n>>> 🔴 Captured Mat_p (对角线置零后) from MALICIOUS Client {client_id} at Round 5 for visualization (M'={m_prime:.4f}) <<<\n")

                            if self.adaptive_threshold is not None:
                                current_threshold = self.adaptive_threshold.add_detection(m_prime, client_id)
                            else:
                                current_threshold = self.defense_threshold

                            if m_prime < current_threshold:
                                print(f"Client {client_id}: PASSED detection this round. (M'={m_prime:.4f}, threshold={current_threshold:.4f})")
                            else:
                                print(
                                    f"Client {client_id}: FLAGGED AS MALICIOUS. (M'={m_prime:.4f}, threshold={current_threshold:.4f}) ADDING to blacklist.")
                        
                                self.malicious_blacklist.add(client_id)
                    except Exception as e:
                        print(f"Error inspecting client {client_id}: {e}.")

                if use_similarity_mode and len(client_matps_for_similarity) >= 2:
                    print(f"\n{'='*70}")
                    print(f"[Ablation Study] Mat_p Similarity-Based Detection")
                    print(f"  Mode: Replace M' threshold with similarity clustering")
                    print(f"  Metric: {neuro_forensics_detector.similarity_metric}")
                    print(f"  Clients with Mat_p: {len(client_matps_for_similarity)}")
                    print(f"{'='*70}\n")
                    
                    similarity_results = neuro_forensics_detector.detect_by_similarity(
                        client_matps=client_matps_for_similarity,
                        threshold_percentile=25  
                    )
                    
                    for suspicious_id in similarity_results['suspicious_clients']:
                        self.malicious_blacklist.add(suspicious_id)
                        print(f"  [Similarity] Client {suspicious_id} added to blacklist (low similarity)")
                    
                    if hasattr(self, 'malicious_client_ids'):
                        true_malicious = set(self.malicious_client_ids) & set(self.uploaded_ids)
                        similarity_tp = len(self.malicious_blacklist & true_malicious)
                        similarity_fp = len(self.malicious_blacklist - true_malicious)
                        
                        print(f"\n[Ground Truth Analysis]:")
                        print(f"  Actual malicious in this round: {sorted(list(true_malicious))}")
                        print(f"  Similarity detected: {sorted(list(self.malicious_blacklist))}")
                        print(f"  TP (True Positive): {similarity_tp}")
                        print(f"  FP (False Positive): {similarity_fp}")
                        
                elif use_similarity_mode:
                    print(f"\n⚠️ Warning: Similarity mode enabled but insufficient Mat_p data (need ≥2 clients)")
                    print(f"  Available Mat_p: {len(client_matps_for_similarity)} clients")
                
                elif not use_similarity_mode:
                  
                    pass

                print(f"--- Blacklist now contains {len(self.malicious_blacklist)} client(s). ---")
                
                if False:  
                    try:
                       
                        
                        print(f"\n{'='*70}")
                        print(f" 正在生成第5轮恶意客户端Mat_p热力图（对角线置零后）...")
                        print(f" 数据集: {self.dataset} | 模型: {self.model_str}")
                        print(f" 攻击类型: {getattr(self, 'attack_type', 'None')} | 防御: NeuroForensics")
                        print(f" 🔴 恶意客户端ID: {malicious_client_id}")
                        print(f"{'='*70}\n")
                        
                        if isinstance(malicious_matp, torch.Tensor):
                            Mat_p = malicious_matp.cpu().numpy()
                        else:
                            Mat_p = malicious_matp
                        
                        diag_check = np.diag(Mat_p)
                        print(f"验证对角线: {diag_check} (应该全为0)")
                        
                        num_classes = Mat_p.shape[0]
                      
                        diag_values = np.diag(Mat_p)
                        
                        if num_classes > 1:
                            col_sums = Mat_p.sum(axis=0) / (num_classes - 1)
                            v = col_sums * (num_classes / (num_classes - 1))
                        else:
                            col_sums = Mat_p.sum(axis=0)
                            v = col_sums
                            
                        q1, q3 = np.percentile(v, [25, 75])
                        iqr = q3 - q1
                        if iqr > 1e-12:
                            M = (v.max() - q3) / iqr
                        else:
                            M = 100.0 if v.max() - q3 > 1e-6 else 0.0
                        M_prime = abs(M - 1.0)
                        
                        print(f"恶意客户端 Mat_p 矩阵分析:")
                        print(f"  对角线平均自信度: {diag_values.mean():.4f}")
                        print(f"  对角线值: {diag_values}")
                        print(f"  最可疑类别: Class {v.argmax()} (平均被误分类概率: {col_sums[v.argmax()]:.4f})")
                        
                        print(f"\n  各类别作为目标的平均概率（列均值）:")
                        for j in range(num_classes):
                            col_mean = Mat_p[:, j].mean()
                            print(f"    Class {j}: {col_mean:.4f} {'⚠️ 目标!' if j == getattr(self, 'y_target', -1) else ''}")
                        
                        print(f"\n  异常度量 M: {M:.4f}, M': {M_prime:.4f}")
                        if M_prime > 2.0:
                            print(f"  ⚠️ 检测到后门！(M' > 2.0)")
                        else:
                            print(f"  ⚠️ 注意：M' ≤ 2.0，但这是已知的恶意客户端！")
                        
                        print(f"\n  完整 Mat_p 矩阵 (行=Source, 列=Predicted):")
                        np.set_printoptions(precision=3, suppress=True, linewidth=200)
                        print(Mat_p)
                        
                        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                        plt.rcParams['axes.unicode_minus'] = False
                        
                        fig, ax = plt.subplots(figsize=(12, 10))
                        
                        non_diag_mask = ~np.eye(num_classes, dtype=bool)
                        non_diag_values = Mat_p[non_diag_mask]
                        actual_min = non_diag_values.min()
                        actual_max = non_diag_values.max()
                        
                        vmin_adaptive = max(0.0, actual_min - 0.01)
                        vmax_adaptive = min(1.0, actual_max + 0.01)
                        
                        print(f"颜色范围调整: vmin={vmin_adaptive:.4f}, vmax={vmax_adaptive:.4f} (数据范围: {actual_min:.4f}-{actual_max:.4f})")
                        
                        hmap = sns.heatmap(
                            Mat_p,
                            annot=True,
                            fmt='.3f',
                            cmap='RdYlBu_r',  
                            vmin=vmin_adaptive,
                            vmax=vmax_adaptive,
                            cbar_kws={'label': 'Transition Probability (Diagonal=0)'},
                            linewidths=0.5,
                            linecolor='gray',
                            square=True,
                            ax=ax,
                            annot_kws={'fontsize': 14}  
                        )
                        
                        cbar = hmap.collections[0].colorbar
                        cbar.ax.tick_params(labelsize=16)
                        cbar.set_label('Transition Probability (Diagonal=0)', fontsize=18, fontweight='bold')
                        
                        ax.set_xlabel('Predicted Class', fontsize=22, fontweight='bold')
                        ax.set_ylabel('Source Class', fontsize=22, fontweight='bold')
                        
                        attack_str = getattr(self, 'attack_type', 'None')
                        title = f'NeuroForensics Mat_p Matrix (Diagonal=0) - Round 5 (MALICIOUS Client)\n'
                        title += f'{self.dataset} | {self.model_str} | Attack: {attack_str}\n'
                        title += f'🔴 Malicious Client: {malicious_client_id} | M\': {M_prime:.4f}'
                        
                        ax.set_title(title, fontsize=24, fontweight='bold', pad=20, color='darkred')
                        
                        ax.set_xticks(np.arange(num_classes) + 0.5)
                        ax.set_yticks(np.arange(num_classes) + 0.5)
                        ax.set_xticklabels(range(num_classes), fontsize=16)
                        ax.set_yticklabels(range(num_classes), fontsize=16)
                        
                        for j in range(num_classes):
                            ax.add_patch(plt.Rectangle((j, j), 1, 1, fill=False, edgecolor='black', lw=1.5, linestyle=':'))
                        
                        if hasattr(self, 'y_target'):
                            target_class = self.y_target
                            print(f"\n🎯 后门目标类别: {target_class} (从 args.y_target 获取)")
                            print(f"   将在热力图中标注第 {target_class} 列 (0-indexed)\n")
                            
                            ax.add_patch(plt.Rectangle((target_class, 0), 1, num_classes, 
                                                     fill=False, edgecolor='red', lw=4, linestyle='--'))
                         
                            ax.annotate('🎯 Backdoor Target', 
                                        xy=(target_class + 0.5, -0.5), 
                                        xytext=(target_class + 0.5, -2),
                                        ha='center', fontsize=16, fontweight='bold', color='red',
                                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
                        else:
                            print(f"\n⚠️ 警告: self.y_target 未设置，无法标注后门目标类别\n")
                        
                        plt.tight_layout()
                        
                        os.makedirs('visualizations', exist_ok=True)
                        save_path = f'visualizations/matp_round5_{self.dataset}_{self.model_str}_{attack_str}_MALICIOUS_client{malicious_client_id}_diagonal_zero.png'
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        print(f"\n✅ 恶意客户端 Mat_p热力图（对角线=0）已保存到: {save_path}")
                        
                        data_path = f'visualizations/matp_round5_{self.dataset}_{self.model_str}_{attack_str}_MALICIOUS_client{malicious_client_id}_diagonal_zero.npy'
                        np.save(data_path, Mat_p)
                        print(f"✅ Mat_p矩阵数据（对角线=0）已保存到: {data_path}")
                        
                        benign_matp = None
                        benign_client_id = None
                        for cid, mat_p in matp_for_visualization.items():
                            if not (hasattr(self, 'malicious_client_ids') and cid in self.malicious_client_ids):
                                benign_matp = mat_p
                                benign_client_id = cid
                                break
                        
                        if benign_matp is not None:
                            if isinstance(benign_matp, torch.Tensor):
                                benign_Mat_p = benign_matp.cpu().numpy()
                            else:
                                benign_Mat_p = benign_matp
                            
                            print(f"\n🔵 额外生成良性客户端 {benign_client_id} 的热力图用于对比...")

                            num_classes_b = benign_Mat_p.shape[0]
                            if num_classes_b > 1:
                                col_sums_b = benign_Mat_p.sum(axis=0) / (num_classes_b - 1)
                                v_b = col_sums_b * (num_classes_b / (num_classes_b - 1))
                            else:
                                col_sums_b = benign_Mat_p.sum(axis=0)
                                v_b = col_sums_b
                                
                            q1_b, q3_b = np.percentile(v_b, [25, 75])
                            iqr_b = q3_b - q1_b
                            if iqr_b > 1e-12:
                                M_b = (v_b.max() - q3_b) / iqr_b
                            else:
                                M_b = 100.0 if v_b.max() - q3_b > 1e-6 else 0.0
                            M_prime_b = abs(M_b - 1.0)
                            print(f"   良性客户端 M: {M_b:.4f}, M': {M_prime_b:.4f}")
                            # === [END NEW] ===
                            
                            non_diag_mask_b = ~np.eye(num_classes_b, dtype=bool)
                            non_diag_values_b = benign_Mat_p[non_diag_mask_b]
                            vmin_b = max(0.0, non_diag_values_b.min() - 0.01)
                            vmax_b = min(1.0, non_diag_values_b.max() + 0.01)
                            
                            fig2, ax2 = plt.subplots(figsize=(12, 10))
                            hmap2 = sns.heatmap(
                                benign_Mat_p,
                                annot=True,
                                fmt='.3f',
                                cmap='RdYlBu_r',
                                vmin=vmin_b,
                                vmax=vmax_b,
                                cbar_kws={'label': 'Transition Probability (Diagonal=0)'},
                                linewidths=0.5,
                                linecolor='gray',
                                square=True,
                                ax=ax2,
                                annot_kws={'fontsize': 14}  
                            )

                            cbar2 = hmap2.collections[0].colorbar
                            cbar2.ax.tick_params(labelsize=16)
                            cbar2.set_label('Transition Probability (Diagonal=0)', fontsize=18, fontweight='bold')
                            
                            ax2.set_xlabel('Predicted Class', fontsize=22, fontweight='bold')
                            ax2.set_ylabel('Source Class', fontsize=22, fontweight='bold')
                            
                            title2 = f'NeuroForensics Mat_p Matrix (Diagonal=0) - Round 5 (BENIGN Client)\n'
                            title2 += f'{self.dataset} | {self.model_str} | Attack: {attack_str}\n'
                            title2 += f'🔵 Benign Client: {benign_client_id} | M\': {M_prime_b:.4f}'
                            ax2.set_title(title2, fontsize=24, fontweight='bold', pad=20, color='darkblue')
                            
                            ax2.set_xticks(np.arange(num_classes_b) + 0.5)
                            ax2.set_yticks(np.arange(num_classes_b) + 0.5)
                            ax2.set_xticklabels(range(num_classes_b), fontsize=16)
                            ax2.set_yticklabels(range(num_classes_b), fontsize=16)
                            
                            for j in range(num_classes_b):
                                ax2.add_patch(plt.Rectangle((j, j), 1, 1, fill=False, edgecolor='green', lw=3))
                            
                            if hasattr(self, 'y_target'):
                                target_class = self.y_target
                                ax2.add_patch(plt.Rectangle((target_class, 0), 1, num_classes_b, 
                                                            fill=False, edgecolor='orange', lw=2, linestyle=':'))
                                ax2.annotate('(Target ref)', 
                                             xy=(target_class + 0.5, -0.5), 
                                             xytext=(target_class + 0.5, -1.5),
                                             ha='center', fontsize=14, color='orange')
                            
                            plt.tight_layout()
                            
                            benign_save_path = f'visualizations/matp_round5_{self.dataset}_{self.model_str}_{attack_str}_BENIGN_client{benign_client_id}_diagonal_zero.png'
                            plt.savefig(benign_save_path, dpi=300, bbox_inches='tight')
                            print(f"✅ 良性客户端 Mat_p热力图（对角线=0）已保存到: {benign_save_path}")
                            
                            benign_data_path = f'visualizations/matp_round5_{self.dataset}_{self.model_str}_{attack_str}_BENIGN_client{benign_client_id}_diagonal_zero.npy'
                            np.save(benign_data_path, benign_Mat_p)
                            print(f"✅ 良性客户端数据（对角线=0）已保存到: {benign_data_path}\n")
                            
                            plt.close('all')
                        else:
                            plt.close()
                    
                    except Exception as e:
                        print(f"\n❌ 生成Mat_p热力图时出错: {e}")
                        import traceback
                        traceback.print_exc()
                elif i == 2 and malicious_matp is None:
                    print(f"\n⚠️ 警告：第2轮没有恶意客户端参与训练，无法生成恶意客户端的Mat_p热力图")
                    print(f"   本轮参与的客户端: {self.uploaded_ids}")
                    if hasattr(self, 'malicious_client_ids'):
                        print(f"   系统中的恶意客户端: {self.malicious_client_ids}")
                
                if self.adaptive_threshold is not None:
                    updated_threshold = self.adaptive_threshold.end_round(i)
                    print(f"--- Round {i} completed. Threshold for next round: {updated_threshold:.4f} ---")
                
            elif self.defense_type == 'alignins' and len(self.uploaded_models) > 0:
                print(f"\n--- [Round {i}] Executing AlignIns Defense (lambda_s={self.alignins_defense.lambda_s}, lambda_c={self.alignins_defense.lambda_c}) ---")
                
                from torch.nn.utils import parameters_to_vector, vector_to_parameters
                
                
                global_params = parameters_to_vector(
                    [self.global_model.state_dict()[name] for name in self.global_model.state_dict()]
                ).detach().to(self.device)
                
                if torch.isnan(global_params).any() or torch.isinf(global_params).any():
                    print(f"[ERROR] Global model parameters contain NaN/Inf!")
                    print(f"  NaN count: {torch.isnan(global_params).sum().item()}")
                    print(f"  Inf count: {torch.isinf(global_params).sum().item()}")
                
                client_updates_dict = {}
                local_updates_list = []  
                for client_id, client_model_state in zip(self.uploaded_ids, self.uploaded_models):
                    
                    client_params = parameters_to_vector(
                        [client_model_state[name] for name in client_model_state]
                    ).detach().to(self.device)
                    
                    if torch.isnan(client_params).any() or torch.isinf(client_params).any():
                        print(f"[ERROR] Client {client_id} parameters contain NaN/Inf!")
                        print(f"  NaN count: {torch.isnan(client_params).sum().item()}")
                        print(f"  Inf count: {torch.isinf(client_params).sum().item()}")
                    
                    update = client_params - global_params
                    
                    if torch.isnan(update).any() or torch.isinf(update).any():
                        print(f"[ERROR] Client {client_id} update contains NaN/Inf!")
                        print(f"  NaN count: {torch.isnan(update).sum().item()}")
                        print(f"  Inf count: {torch.isinf(update).sum().item()}")
                    
                    client_updates_dict[client_id] = update
                    local_updates_list.append(update)
                
                benign_idx, benign_ids, detection_info = self.alignins_defense.detect_malicious_clients(
                    client_updates_dict, global_params, self.uploaded_ids
                )
                
                print(f"\n--- AlignIns Detection Results ---")
                tda_scores = [(cid, detection_info['tda'][cid]) for cid in self.uploaded_ids]
                mpsa_scores = [(cid, detection_info['mpsa'][cid]) for cid in self.uploaded_ids]
                mz_tda_scores = [(cid, detection_info['mz_tda'][cid]) for cid in self.uploaded_ids]
                mz_mpsa_scores = [(cid, detection_info['mz_mpsa'][cid]) for cid in self.uploaded_ids]
                print(f"TDA scores: {tda_scores}")
                print(f"MPSA scores: {mpsa_scores}")
                print(f"MZ-TDA scores: {mz_tda_scores}")
                print(f"MZ-MPSA scores: {mz_mpsa_scores}")
                print(f"Benign clients: {benign_ids}")
                print(f"Flagged clients: {detection_info['flagged_ids']}")
                
                if hasattr(self, 'malicious_client_ids') and len(self.malicious_client_ids) > 0:
                  
                    actual_malicious_ids_global = set(self.malicious_client_ids)
                    
                    actual_malicious_in_round = [cid for cid in self.uploaded_ids if cid in actual_malicious_ids_global]
                    actual_benign_in_round = [cid for cid in self.uploaded_ids if cid not in actual_malicious_ids_global]
                    
                    detected_benign_ids = set(benign_ids)
                    detected_malicious_ids = set(detection_info['flagged_ids'])
                    
                    true_positives = len([cid for cid in actual_benign_in_round if cid in detected_benign_ids])
                    TPR = true_positives / len(actual_benign_in_round) if len(actual_benign_in_round) > 0 else 0
                    
                    false_positives = len([cid for cid in actual_malicious_in_round if cid in detected_benign_ids])
                    FPR = false_positives / len(actual_malicious_in_round) if len(actual_malicious_in_round) > 0 else 0
                    
                    true_negatives = len([cid for cid in actual_malicious_in_round if cid in detected_malicious_ids])
                    TNR = true_negatives / len(actual_malicious_in_round) if len(actual_malicious_in_round) > 0 else 0
                    
                    print(f"\n--- AlignIns Performance Metrics ---")
                    print(f"Global malicious clients: {sorted(list(actual_malicious_ids_global))}")
                    print(f"Actual malicious in round: {actual_malicious_in_round}")
                    print(f"Actual benign in round: {actual_benign_in_round}")
                    print(f"True Positive Rate (TPR):  {TPR:.4f} ({true_positives}/{len(actual_benign_in_round)}) - 正确识别良性")
                    print(f"False Positive Rate (FPR): {FPR:.4f} ({false_positives}/{len(actual_malicious_in_round)}) - 恶意误判为良性")
                    print(f"True Negative Rate (TNR):  {TNR:.4f} ({true_negatives}/{len(actual_malicious_in_round)}) - 正确识别恶意")
                    print(f"Total detected benign: {len(benign_ids)}/{len(self.uploaded_ids)}")
                
                if len(benign_idx) > 0:
                    print(f"\n--- Applying AlignIns Model Clipping ---")
                    clipped_updates = self.alignins_defense.apply_model_clipping(local_updates_list, benign_idx)
                    print(f"Model clipping completed. Clipped {len(clipped_updates)} updates.")
                    
                    benign_clipped_dict = {}
                    benign_weights_dict = {}
                    for idx in benign_idx:
                        client_id = self.uploaded_ids[idx]
                        benign_clipped_dict[client_id] = clipped_updates[idx]
                  
                        benign_weights_dict[client_id] = self.uploaded_weights[idx]
                    
                    total_weight = sum(benign_weights_dict.values())
                    if total_weight > 0:
                        benign_weights_dict = {k: v/total_weight for k, v in benign_weights_dict.items()}
                    
                    aggregated_update = torch.zeros_like(global_params)
                    for client_id, clipped_update in benign_clipped_dict.items():
                        weight = benign_weights_dict[client_id]
                        
                        if torch.isnan(clipped_update).any() or torch.isinf(clipped_update).any():
                            print(f"[WARNING] Client {client_id} clipped update contains NaN/Inf! Skipping.")
                            continue
                        
                        aggregated_update += weight * clipped_update
                    
                    if torch.isnan(aggregated_update).any() or torch.isinf(aggregated_update).any():
                        print(f"\n[ERROR] Aggregated update contains NaN/Inf before adding to global params!")
                        print(f"NaN count in aggregated_update: {torch.isnan(aggregated_update).sum().item()}")
                        print(f"Inf count in aggregated_update: {torch.isinf(aggregated_update).sum().item()}")
                    
                    print(f"\n--- Aggregated Update Statistics ---")
                    print(f"Update norm: {torch.norm(aggregated_update).item():.4f}")
                    print(f"Update max: {aggregated_update.max().item():.4f}")
                    print(f"Update min: {aggregated_update.min().item():.4f}")
                    print(f"Update mean: {aggregated_update.mean().item():.4f}")
                    
                    new_global_params = global_params + aggregated_update
                    
                    print(f"\n--- New Global Parameters Statistics ---")
                    print(f"Params norm: {torch.norm(new_global_params).item():.4f}")
                    print(f"Params max: {new_global_params.max().item():.4f}")
                    print(f"Params min: {new_global_params.min().item():.4f}")
                    print(f"Params mean: {new_global_params.mean().item():.4f}")
                    
                    if torch.isnan(new_global_params).any() or torch.isinf(new_global_params).any():
                        print(f"\n[ERROR] AlignIns aggregation resulted in NaN/Inf parameters! Skipping update.")
                        print(f"NaN count: {torch.isnan(new_global_params).sum().item()}")
                        print(f"Inf count: {torch.isinf(new_global_params).sum().item()}")
                    else:
                        vector_to_parameters(new_global_params, self.global_model.parameters())
                        print(f"Global model updated with {len(benign_idx)} benign clients (AlignIns aggregation).")
                    
                    self.uploaded_models = []
                    self.uploaded_ids = []
                    self.uploaded_weights = []
              
                    continue
                else:
                    print(f"\n--- WARNING: No benign clients detected! Skipping AlignIns aggregation. ---")

            elif self.defense_type == 'none':
                print(f"\n--- [Round {i}] Defense is OFF. Aggregating all models. ---")

            final_ids_to_aggregate = []
            final_models_to_aggregate = []

            if len(self.malicious_blacklist) > 0:
                print(f"--- Applying blacklist: {sorted(list(self.malicious_blacklist))} ---")

            for client_id, model_state in zip(self.uploaded_ids, self.uploaded_models):
                if client_id not in self.malicious_blacklist:
                   
                    final_ids_to_aggregate.append(client_id)
                    final_models_to_aggregate.append(model_state)
                else:
                    print(f"Client {client_id}: Excluded from aggregation because it is on the blacklist.")

            if len(final_models_to_aggregate) > 0:
                self.aggregate_parameters(final_ids_to_aggregate, final_models_to_aggregate)
            else:
                print("Warning: No models left to aggregate after blacklisting. Skipping aggregation for this round.")

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def aggregate_parameters(self, ids_to_aggregate, models_to_aggregate):
        """
        新的、统一的聚合方法。
        它接收一个模型ID列表和一个模型权重字典列表，并进行**加权平均**或**鲁棒聚合**。
        """
        assert len(models_to_aggregate) > 0, "aggregation list can not be empty"

        id_to_weight_map = dict(zip(self.uploaded_ids, self.uploaded_weights))

        weights_to_aggregate = [id_to_weight_map[client_id] for client_id in ids_to_aggregate]

        total_weight = sum(weights_to_aggregate)
        normalized_weights = [w / total_weight for w in weights_to_aggregate]

        if self.defense_type == 'flame' and self.flame_defender is not None:
           
            current_round = getattr(self, 'current_round', None)
            aggregated_state, detection_info = self.flame_defender.aggregate_with_flame(
                models_to_aggregate, normalized_weights, self.global_model.state_dict(),
                uploaded_ids=ids_to_aggregate, current_round=current_round
            )
            
            print(f"\n{'='*50}")
            print(f"FLAME Detection Results (Round {current_round}):")
            print(f"  Benign clients: {detection_info['benign_client_ids']}")
            print(f"  Suspicious clients: {detection_info['suspicious_client_ids']}")
            print(f"  Cluster labels: {detection_info['cluster_labels']}")
            print(f"{'='*50}\n")
            
            self.global_model.load_state_dict(aggregated_state)
        elif self.defense_type == 'robust_aggregation' and self.robust_aggregator is not None:
      
            aggregated_state = self.robust_aggregator.apply_robust_aggregation(
                models_to_aggregate, normalized_weights, self.global_model.state_dict()
            )
            self.global_model.load_state_dict(aggregated_state)
        else:
            temp_model = copy.deepcopy(self.global_model)

            for param in self.global_model.parameters():
                param.data.zero_()

            for w, model_state in zip(normalized_weights, models_to_aggregate):
                temp_model.load_state_dict(model_state)
                for server_param, client_param in zip(self.global_model.parameters(), temp_model.parameters()):
                    server_param.data += client_param.data.clone() * w

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.selected_clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def _inject_darkfed_updates(self, round_idx):
        """Inject fake client updates from DarkFed Sybil clients"""
        print(f"\n--- [Round {round_idx}] DarkFed: Injecting fake client updates ---")

        # Reset fake client updates for L_cd computation
        if hasattr(self.darkfed_attacker, '_current_fake_updates'):
            self.darkfed_attacker._current_fake_updates = []

        device = next(self.global_model.parameters()).device
        fake_client_models = []
        fake_client_ids = []

        # Create updates from multiple fake clients
        for fake_id in range(self.darkfed_attacker.num_fake_clients):
            print(f"Creating malicious update from fake client {fake_id}")
            fake_model = self.darkfed_attacker.create_fake_client_updates(self.global_model, fake_id)
            fake_client_models.append(fake_model)
            fake_client_ids.append(f"fake_{fake_id}")

        # Add fake client models to uploaded models
        if not hasattr(self, 'uploaded_models'):
            self.uploaded_models = []
            self.uploaded_ids = []

        for fake_model, fake_id in zip(fake_client_models, fake_client_ids):
            
            self.uploaded_models.append(fake_model.state_dict())
            self.uploaded_ids.append(fake_id)

        if hasattr(self, 'uploaded_weights') and len(self.uploaded_weights) > 0:
            avg_real_weight = sum(self.uploaded_weights) / len(self.uploaded_weights) if self.uploaded_weights else 0.1

            enhanced_weight = avg_real_weight * 1.5
            fake_weights = [enhanced_weight] * len(fake_client_models)
            self.uploaded_weights.extend(fake_weights)

            total_weight = sum(self.uploaded_weights)
            if total_weight > 0:
                self.uploaded_weights = [w / total_weight for w in self.uploaded_weights]

        print(f"Injected {len(fake_client_models)} fake client updates")
        print(f"Total models for aggregation: {len(self.uploaded_models)}")
        print(
            f"Weights after fake client injection: {[f'{w:.4f}' for w in self.uploaded_weights[-len(fake_client_models):]]}")