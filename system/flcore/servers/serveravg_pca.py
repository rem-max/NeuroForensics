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
            
            elif self.defense_type == 'neuroforensics' and len(self.uploaded_models) > 0:
                print(f"\n--- [Round {i}] Executing NeuroForensics (PCA-based) Defense ---")
                
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
                
                # Initialize NeuroForensics detector (supports multi-level detection and PCA fusion)
                neuro_forensics_detector = NeuroForensics(
                    seed=int(time.time()),
                    enable_multi_level=getattr(self, 'enable_multi_level', False),
                    multi_level_positions=multi_level_positions,
                    fusion_strategy=getattr(self, 'fusion_strategy', 'weighted_average'),
                    use_pca=True  # Enable PCA fusion in serveravg_pca
                )

                if 'MNIST' in self.dataset or 'FashionMNIST' in self.dataset:
                    shape = (1, 1, 28, 28)
                elif 'Cifar10' in self.dataset or 'Cifar100' in self.dataset:
                    shape = (1, 3, 32, 32)
                else:
                    print(f"Warning: Unknown dataset '{self.dataset}'. Defaulting to CIFAR-10 shape.")
                    shape = (1, 3, 32, 32)

                client_metrics_map = {} # {client_id: metrics_dict}
                
                matp_for_visualization = {} 
                malicious_matp = None
                malicious_client_id = None

                print(f"[NeuroForensics] Extracting metrics from {len(self.uploaded_models)} clients...")

                for client_id, client_model_state in zip(self.uploaded_ids, self.uploaded_models):
                    try:

                        if self.model_str in ["ResNet10", "ResNet18"]:
                            block_type = BasicBlock
                            layer_config = [1, 1, 1, 1] if self.model_str == "ResNet10" else [2, 2, 2, 2]
                            classifier_part = AdaptiveModelClass(
                                block=block_type,
                                layers=layer_config,
                                num_classes=self.num_classes,
                                inspect_layer_position=self.lsep_layer,
                                original_input_img_shape=shape
                            ).to(self.device)
                        else:
                            classifier_part = AdaptiveModelClass(
                                num_classes=self.num_classes,
                                inspect_layer_position=self.lsep_layer,
                                original_input_img_shape=shape
                            ).to(self.device)
                        
                        classifier_part.load_state_dict(client_model_state, strict=False)

                        schedule = {'device': self.device, 'num_classes': self.num_classes}
                        
                        results = neuro_forensics_detector.test(model=classifier_part, dataset=None, schedule=schedule)
                        
                        client_metrics_map[client_id] = results['metrics']
                        
                        m = results['metrics']

                        tag = "🔴" if (hasattr(self, 'malicious_client_ids') and client_id in self.malicious_client_ids) else "🔵"
                        print(f"  Client {client_id} {tag}: H={m['S_entropy']:.2f}, Max={m['S_max']:.2f}, Kurt={m['S_kurt']:.2f}, Box={m['S_boxplot']:.2f}")

                        if i == 5:
                             matp_for_visualization[client_id] = results.get('Mat_p', None)
                             if tag == "🔴" and malicious_matp is None:
                                 malicious_matp = results.get('Mat_p', None)
                                 malicious_client_id = client_id

                    except Exception as e:
                        print(f"Error inspecting client {client_id}: {e}")

                if len(client_metrics_map) >= 2:
                    print(f"\n[NeuroForensics] Running PCA Fusion on {len(client_metrics_map)} clients...")
                    pca_scores = neuro_forensics_detector.detect_by_pca(client_metrics_map)
                    
                    all_scores = list(pca_scores.values())
                    mean_score = np.mean(all_scores)
                    std_score = np.std(all_scores)
                    
                    threshold_k = 2.0 

                    dynamic_threshold = mean_score + threshold_k * (std_score + 1e-6)
                    
                    print(f"[NeuroForensics] PCA Score Stats: Mean={mean_score:.4f}, Std={std_score:.4f}")
                    print(f"[NeuroForensics] Dynamic Threshold = {dynamic_threshold:.4f} (Mean + {threshold_k}*Std)")

                    new_detections = 0
                    for cid, score in pca_scores.items():
                        is_malicious = score > dynamic_threshold
                        
                        status = "🔴 FLAGGED" if is_malicious else "🔵 PASS"
                        truth = "(True Malicious)" if (hasattr(self, 'malicious_client_ids') and cid in self.malicious_client_ids) else ""
                        
                        print(f"  Client {cid}: PCA Score = {score:.4f}  {status} {truth}")

                        if is_malicious:
                            self.malicious_blacklist.add(cid)
                            new_detections += 1
                    
                    if new_detections > 0:
                        print(f"\n[NeuroForensics] Detected {new_detections} malicious clients this round.")
                    else:
                        print(f"\n[NeuroForensics] No anomalies detected above threshold.")

                else:
                    print("[NeuroForensics] Not enough clients for PCA (need >= 2). Skipping detection.")

                print(f"--- Blacklist now contains {len(self.malicious_blacklist)} client(s). ---")
                
                if i == 5 and malicious_matp is not None:
                    try:
                        print(f"\n{'='*70}")
                        print(f" Generating Visualization for Round 5...")
                        print(f"{'='*70}\n")
                        
                        if isinstance(malicious_matp, torch.Tensor):
                            Mat_p = malicious_matp.cpu().numpy()
                        else:
                            Mat_p = malicious_matp
                        
                        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                        plt.rcParams['axes.unicode_minus'] = False
                        
                        fig, ax = plt.subplots(figsize=(12, 10))
                        
                        num_classes = Mat_p.shape[0]
                        non_diag_mask = ~np.eye(num_classes, dtype=bool)
                        non_diag_values = Mat_p[non_diag_mask]
                        vmin_adaptive = max(0.0, non_diag_values.min() - 0.01)
                        vmax_adaptive = min(1.0, non_diag_values.max() + 0.01)
                        
                        hmap = sns.heatmap(
                            Mat_p, annot=True, fmt='.3f', cmap='RdYlBu_r',
                            vmin=vmin_adaptive, vmax=vmax_adaptive,
                            cbar_kws={'label': 'Transition Probability (Diagonal=0)'},
                            square=True, ax=ax, annot_kws={'fontsize': 14}
                        )
                        
                        title = f'NeuroForensics Mat_p Matrix - Round 5 (MALICIOUS Client {malicious_client_id})\n'
                        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
                        
                        os.makedirs('visualizations', exist_ok=True)
                        save_path = f'visualizations/matp_round5_PCA_MALICIOUS_client{malicious_client_id}.png'
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        print(f"✅ Visualization saved to: {save_path}")
                        plt.close('all')

                    except Exception as e:
                        print(f"Visualization error: {e}")

                
            elif self.defense_type == 'alignins' and len(self.uploaded_models) > 0:
                print(f"\n--- [Round {i}] Executing AlignIns Defense (lambda_s={self.alignins_defense.lambda_s}, lambda_c={self.alignins_defense.lambda_c}) ---")
                
                from torch.nn.utils import parameters_to_vector, vector_to_parameters
                
                global_params = parameters_to_vector(
                    [self.global_model.state_dict()[name] for name in self.global_model.state_dict()]
                ).detach().to(self.device)
                
                client_updates_dict = {}
                local_updates_list = [] 
                for client_id, client_model_state in zip(self.uploaded_ids, self.uploaded_models):
                    client_params = parameters_to_vector(
                        [client_model_state[name] for name in client_model_state]
                    ).detach().to(self.device)
                    
                    update = client_params - global_params
                    client_updates_dict[client_id] = update
                    local_updates_list.append(update)
                
                benign_idx, benign_ids, detection_info = self.alignins_defense.detect_malicious_clients(
                    client_updates_dict, global_params, self.uploaded_ids
                )
                
                print(f"AlignIns detected benign clients: {benign_ids}")
                
                if len(benign_idx) > 0:
                    print(f"\n--- Applying AlignIns Model Clipping ---")
                    clipped_updates = self.alignins_defense.apply_model_clipping(local_updates_list, benign_idx)
                    
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
                        aggregated_update += weight * clipped_update
                    
                    new_global_params = global_params + aggregated_update
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