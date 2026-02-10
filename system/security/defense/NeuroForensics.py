
import os
import os.path as osp
import json
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Base

class _PartialLaterHead(nn.Module):
    def __init__(self, model: nn.Module, lsep_shape: Tuple[int, int, int]):
        super().__init__()
        self.lsep_shape = lsep_shape  # (C,H,W)
        flatten_dim = int(np.prod(lsep_shape))

        num_classes = 10  
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            num_classes = model.fc.out_features
        elif hasattr(model, 'classifier'):
            for layer in reversed(list(model.classifier.children())):
                if isinstance(layer, nn.Linear):
                    num_classes = layer.out_features
                    break

        model_name = model.__class__.__name__
        lsep_pos = getattr(model, 'inspect_layer_position', 2)

        if "LeNet" in model_name and hasattr(model, 'bottleneck'):
 
            if lsep_pos == 2:  
                self.head = nn.Sequential(
                    model.bottleneck,
                    model.bn,
                    model.dropout,
                    model.fc
                )
            else:

                self.head = nn.Linear(flatten_dim, num_classes)

        elif hasattr(model, "classifier"):  
            if lsep_pos == 2:
       
                remaining_layers = list(model.classifier.children())[5:]  # [5] Dropout, [6] Linear
                if remaining_layers:
                    self.head = nn.Sequential(*remaining_layers)
                else:
                    self.head = nn.Linear(flatten_dim, num_classes)
            else:
    
                original_first_linear = model.classifier[0]
                if isinstance(original_first_linear, nn.Linear) and original_first_linear.in_features == flatten_dim:
    
                    self.head = model.classifier
                else:
            
                    self.head = nn.Sequential(
                        nn.Linear(flatten_dim, 4096),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(4096, 4096),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(4096, num_classes)
                    )

        elif hasattr(model, "fc"):  
            original_fc = model.fc
            if isinstance(original_fc, nn.Linear):
                if original_fc.in_features == flatten_dim:
                   
                    self.head = original_fc
                else:
                   
                    self.head = nn.Linear(flatten_dim, original_fc.out_features)
            else:
                self.head = nn.Linear(flatten_dim, num_classes)
        else:
            self.head = nn.Linear(flatten_dim, num_classes)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(x_flat.size(0), -1)  # [B, D]
        return self.head(x)  # logits


def _require_partial_and_build_head(model: nn.Module) -> Tuple[nn.Module, int]:
    """
    Support arbitrary inspect_layer_position for partial model backend.
    Extract (C,H,W) at Lsep from model.input_shapes[pos], build _PartialLaterHead and return in_dim=C*H*W.
    """
    if not (hasattr(model, "input_shapes") and hasattr(model, "inspect_layer_position")):
        raise RuntimeError(
            "NeuroForensics(partial-only) only supports partial backend model. "
            "Please use networks.partial_models_adaptive.*AdaptivePartialModel."
        )
    pos = int(getattr(model, "inspect_layer_position"))

    # Remove hardcoded pos != 2 restriction, support arbitrary lsep position
    if pos < 0 or pos >= len(model.input_shapes):
        raise RuntimeError(f"inspect_layer_position {pos} 超出范围，可用范围：0-{len(model.input_shapes) - 1}")

    shp = model.input_shapes[pos]  # (1,C,H,W) or (C,H,W)
    if len(shp) == 4:
        c, h, w = int(shp[1]), int(shp[2]), int(shp[3])
    elif len(shp) == 3:
        c, h, w = int(shp[0]), int(shp[1]), int(shp[2])
    else:
        raise RuntimeError(f"Unable to parse Lsep shape: {shp}")

    head = _PartialLaterHead(model, (c, h, w))
    in_dim = int(c * h * w)
    return head, in_dim


# ---------- Optimize Dummy IR in Lsep(flat) space ----------
def _gen_dummy_ir_batch_(
        model_head: nn.Module,
        class_id: int,
        batch_size: int,
        in_dim: int,
        device: torch.device,
        optim_step: int = 200,
        lr: float = 1e-2,
        weight_decay: float = 5e-3,
        clamp_nonneg: bool = True,  # After ReLU at VGG16@Lsep(=2), non-negative
        generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn((batch_size, in_dim), device=device, generator=generator, requires_grad=True)
    y = torch.full((batch_size,), class_id, device=device, dtype=torch.long)
    opt = torch.optim.Adam([x], lr=lr, weight_decay=weight_decay)

    for _ in range(optim_step):
        out = model_head(x)  # logits
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if clamp_nonneg:
            with torch.no_grad():
                x.clamp_(0)

    return x.detach()


# ---------- Average Dummy IR → Backend → Probability Matrix Mat_p ----------
def _build_Mat_p_by_mean_forward(
        model_head: nn.Module,
        dummy_inner_embeddings_all: torch.Tensor,  # [K, B, D] or [K, num_restarts, D]
        num_classes: int,
        device: torch.device,
) -> np.ndarray:
    Mat_p = np.zeros((num_classes, num_classes), dtype=np.float64)

    try:
        param_dtype = next(model_head.parameters()).dtype
    except StopIteration:
        param_dtype = torch.float32

    T = 1.5

    with torch.no_grad():
        for src in range(num_classes):
            dummies = dummy_inner_embeddings_all[src]
            dummy_avg = dummies.mean(dim=0, keepdim=True).to(device=device, dtype=param_dtype)
            logits = model_head(dummy_avg) / T
            probs = torch.softmax(logits, dim=1)
            Mat_p[src, :] = probs.squeeze(0).cpu().numpy().astype(np.float64)

    return Mat_p


def _boxplot_anomaly_from_v(v: np.ndarray) -> float:
    if v.size < 4:
        return 0.0
    q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
    iqr = q3 - q1
    if iqr <= 1e-12:
        return 100.0
    return float((v.max() - q3) / iqr)


# ---------- Multi-Level Detection Module ----------
class MultiLevelDetector:
    """
    Multi-Level Detection: Backdoor detection at multiple model layers

    Core ideas:
    1. Backdoor attacks typically leave traces at multiple layers
    2. Detection results at different layers can validate each other
    3. Multi-level detection reduces false positives and improves confidence

    Innovations:
    - M' value fusion strategies across layers
    - Inter-layer consistency analysis
    - Adaptive layer weight adjustment
    """

    def __init__(self,
                 fusion_strategy: str = 'weighted_average',
                 consistency_threshold: float = 0.3):
        """
        Args:
            fusion_strategy: Fusion strategy ('max', 'average', 'weighted_average', 'voting')
            consistency_threshold: Inter-layer consistency threshold
        """
        self.fusion_strategy = fusion_strategy
        self.consistency_threshold = consistency_threshold
        self.layer_weights = None  # Will be computed adaptively at runtime

    def detect_multi_level(self,
                           neuroforensics_instance,
                           model: nn.Module,
                           dataset,
                           schedule: Dict[str, Any],
                           level_positions: List[int] = None) -> Dict[str, Any]:
        """
        Detect at multiple layers

        Args:
            neuroforensics_instance: NeuroForensics instance
            model: Model to detect
            dataset: Dataset
            schedule: Config dict
            level_positions: List of layer positions to detect (e.g., [1, 2, 3])

        Returns:
            Dict containing multi-level detection results
        """
        # Get original lsep position
        original_lsep = getattr(model, 'inspect_layer_position', 2)

        if level_positions is None:
            # Default: detect 3 layers, must include lsep layer
            if hasattr(model, 'input_shapes'):
                total_levels = len(model.input_shapes)
                lsep_pos = original_lsep
                if lsep_pos < 0 or lsep_pos >= total_levels:
                    lsep_pos = min(2, total_levels - 1)

                if total_levels >= 3:
        
                    shallow_pos = 0 
                    deep_pos = total_levels - 1  
                    level_positions = sorted(list(set([shallow_pos, lsep_pos, deep_pos])))

                    if len(level_positions) < 3:
                        mid_pos = total_levels // 2
                        level_positions = sorted(list(set([shallow_pos, mid_pos, lsep_pos, deep_pos])))[:3]
                else:
                    level_positions = list(range(total_levels))
            else:
                level_positions = [original_lsep]  
        else:
            if original_lsep not in level_positions:
                level_positions = sorted(level_positions + [original_lsep])
                print(f"[Info] Lsep layer (position {original_lsep}) automatically added to detection levels")

        print(f"\n{'=' * 60}")
        print(f"Multi-Level Detection: Testing at positions {level_positions}")
        print(f"  - Lsep layer (default NeuroForensics position): {original_lsep}")
        print(f"{'=' * 60}")

        level_results = []
        for level_idx, pos in enumerate(level_positions):
            print(f"\n--- Level {level_idx + 1}/{len(level_positions)}: Position {pos} ---")

            if hasattr(model, 'inspect_layer_position'):
                model.inspect_layer_position = pos

            try:
                result = neuroforensics_instance.score(model, dataset, schedule)
                result['level_position'] = pos
                level_results.append(result)

                print(f"  M'={result['M_prime']:.4f}, M={result['M']:.4f}, "
                      f"pred_target={result['pred_target']}")
            except Exception as e:
                print(f"  Warning: Detection failed at position {pos}: {e}")
                continue

        if hasattr(model, 'inspect_layer_position'):
            model.inspect_layer_position = original_lsep

        if not level_results:
            raise RuntimeError("Multi-level detection failed at all positions")

        fused_result = self._fuse_results(level_results)

        consistency_analysis = self._analyze_consistency(level_results)
        fused_result['consistency_analysis'] = consistency_analysis

        print(f"\n{'=' * 60}")
        print(f"Multi-Level Detection Results:")
        print(f"  Fused M': {fused_result['fused_M_prime']:.4f}")

        if 'fusion_strategy_used' in fused_result:
            print(f"  Fusion Strategy: {fused_result['fusion_strategy_used']}")

        if 'layer0_m' in fused_result:
            layer0_m = fused_result['layer0_m']
            if layer0_m > 2.0:
                print(f"  Layer0 M: {layer0_m:.4f} > 2.0 (异常严重 → 只用Layer1&2平均)")
            else:
                print(f"  Layer0 M: {layer0_m:.4f} ≤ 2.0 (正常范围 → softmax全层加权)")

        print(f"  Consistency Score: {consistency_analysis['consistency_score']:.4f}")
        print(f"  Consistent Prediction: {consistency_analysis['is_consistent']}")
        print(f"{'=' * 60}\n")

        return fused_result

    def _fuse_results(self, level_results: List[Dict[str, Any]]) -> Dict[str, Any]:
       
        m_prime_values = [r['M_prime'] for r in level_results]
        m_values = [r['M'] for r in level_results]
        pred_targets = [r['pred_target'] for r in level_results]

        if self.fusion_strategy == 'max':
            fused_m_prime = max(m_prime_values)
            fused_idx = m_prime_values.index(fused_m_prime)

        elif self.fusion_strategy == 'average':
            fused_m_prime = np.mean(m_prime_values)
            fused_idx = 0

        elif self.fusion_strategy == 'weighted_average':
            weights = self._compute_layer_weights(level_results)
            fused_m_prime = np.average(m_prime_values, weights=weights)
            fused_idx = 0

            self._last_fusion_weights = weights

        elif self.fusion_strategy == 'voting':
            threshold = np.mean(m_prime_values)
            votes = [1 if m > threshold else 0 for m in m_prime_values]
            fused_m_prime = np.mean(m_prime_values) if sum(votes) > len(votes) / 2 else min(m_prime_values)
            fused_idx = 0
        else:
            fused_m_prime = np.mean(m_prime_values)
            fused_idx = 0

        level_positions = [r['level_position'] for r in level_results]
        fused_result = {
            'fused_M_prime': float(fused_m_prime),
            'fused_M': float(np.mean(m_values)),
            'level_M_prime_values': [float(m) for m in m_prime_values],
            'level_M_values': [float(m) for m in m_values],
            'level_positions': level_positions,
            'level_pred_targets': pred_targets,
            'fusion_strategy': self.fusion_strategy,
            'most_suspicious_level': level_results[m_prime_values.index(max(m_prime_values))]['level_position'],
            'detailed_results': level_results
        }

        if 0 in level_positions:
            layer0_idx = level_positions.index(0)
            layer0_m = m_values[layer0_idx]
            layer0_m_prime = m_prime_values[layer0_idx]
            fused_result['layer0_m'] = float(layer0_m)
            fused_result['layer0_m_prime'] = float(layer0_m_prime)

            if self.fusion_strategy == 'weighted_average' and hasattr(self, '_last_fusion_weights'):
                if layer0_m > 2.0:
                    fused_result['fusion_strategy_used'] = 'average_layer1&2 (Layer0_M>2.0, excluded)'
                else:
                    fused_result['fusion_strategy_used'] = 'softmax_adaptive (Layer0_M≤2.0, all layers)'

        return fused_result

    def _compute_layer_weights(self, level_results: List[Dict[str, Any]]) -> np.ndarray:
        m_prime_values = np.array([r['M_prime'] for r in level_results])
        m_values = np.array([r['M'] for r in level_results])
        level_positions = [r['level_position'] for r in level_results]

        layer0_idx = None
        layer0_m = None
        for i, pos in enumerate(level_positions):
            if pos == 0:
                layer0_idx = i
                layer0_m = m_values[i]
                break

        if layer0_idx is not None and layer0_m is not None:
            if layer0_m > 2.0:
                weights = np.zeros(len(level_results))

                non_layer0_indices = [i for i, pos in enumerate(level_positions) if pos != 0]

                if len(non_layer0_indices) > 0:
                    for idx in non_layer0_indices:
                        weights[idx] = 1.0 / len(non_layer0_indices)

                    print(f"  [Weight Strategy] Layer0 M={layer0_m:.4f} > 2.0 → Ignore Layer0, use Layer1&2 average:")
                    for pos, w in zip(level_positions, weights):
                        if w > 0:
                            print(f"    - Layer {pos}: weight={w:.4f} [average of layer1&2]")
                        else:
                            print(f"    - Layer {pos}: weight={w:.4f} [excluded]")
                else:
                    # Fallback to using layer0 if only layer0 available
                    weights[layer0_idx] = 1.0
                    print(f"  [Warning] Only Layer0 available, using Layer0 weight")
            else:
                # Strategy 2: M(layer0) ≤ 2, use softmax adaptive weights for all layers
                weights = np.exp(m_prime_values) / np.sum(np.exp(m_prime_values))

                print(f"  [Weight Strategy] Layer0 M={layer0_m:.4f} ≤ 2.0 → Using all layers Softmax adaptive weights:")
                for pos, w in zip(level_positions, weights):
                    print(f"    - Layer {pos}: weight={w:.4f} [softmax adaptive]")
        else:
            # If no layer0, fallback to standard softmax
            print(f"  [Warning] Layer0 not found, using standard softmax weights")
            weights = np.exp(m_prime_values) / np.sum(np.exp(m_prime_values))
            for pos, w in zip(level_positions, weights):
                print(f"    - Layer {pos}: weight={w:.4f}")

        return weights

    def _analyze_consistency(self, level_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze consistency of multi-level detection results

        Consistency indicators:
        1. Standard deviation of M' values (smaller = more consistent)
        2. Consistency of predicted targets
        3. Correlation of v vectors
        """
        m_prime_values = [r['M_prime'] for r in level_results]
        pred_targets = [r['pred_target'] for r in level_results]

        # 1. Coefficient of variation for M' values
        m_prime_std = np.std(m_prime_values)
        m_prime_mean = np.mean(m_prime_values)
        cv = m_prime_std / (m_prime_mean + 1e-10)  # Coefficient of variation

        # 2. Target prediction consistency (mode ratio)
        from collections import Counter
        target_counts = Counter(pred_targets)
        most_common_target, most_common_count = target_counts.most_common(1)[0]
        target_consistency = most_common_count / len(pred_targets)

        # 3. Comprehensive consistency score (higher = more consistent)
        consistency_score = (1.0 - min(cv, 1.0)) * 0.5 + target_consistency * 0.5

        # 4. Determine if consistent
        is_consistent = (cv < self.consistency_threshold) and (target_consistency > 0.5)

        analysis = {
            'consistency_score': float(consistency_score),
            'is_consistent': bool(is_consistent),
            'm_prime_std': float(m_prime_std),
            'm_prime_mean': float(m_prime_mean),
            'coefficient_of_variation': float(cv),
            'target_consistency': float(target_consistency),
            'most_common_target': int(most_common_target),
            'target_agreement_ratio': float(most_common_count / len(pred_targets))
        }

        return analysis


# ---------- NeuroForensics（partial-only） ----------
class NeuroForensics(Base):
    def __init__(self,
                 steps: int = 200,
                 lr: float = 1e-2,
                 weight_decay: float = 5e-3,
                 M_const: Optional[float] = None,
                 seed: int = 0,
                 deterministic: bool = False,
                 enable_multi_level: bool = False,
                 multi_level_positions: Optional[List[int]] = None,
                 fusion_strategy: str = 'weighted_average',
                 use_similarity_detection: bool = False,
                 similarity_metric: str = 'cosine',
                 use_pca: bool = False):
        super().__init__(seed=seed, deterministic=deterministic)
        self.steps = steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.M_const = M_const
        self.seed = seed

        # Multi-level detection
        self.enable_multi_level = enable_multi_level
        self.multi_level_positions = multi_level_positions
        if self.enable_multi_level:
            self.multi_level_detector = MultiLevelDetector(
                fusion_strategy=fusion_strategy,
                consistency_threshold=0.3
            )
            print(f"[NeuroForensics] Multi-level detection enabled with strategy: {fusion_strategy}")
        else:
            self.multi_level_detector = None
        
        # Mat_p similarity detection (ablation study)
        self.use_similarity_detection = use_similarity_detection
        self.similarity_metric = similarity_metric
        if self.use_similarity_detection:
            print(f"[NeuroForensics] Similarity-based detection enabled with metric: {similarity_metric}")
            print(f"  Mode: Ablation study - comparing similarity vs multi-indicator fusion")
        
        # PCA-based multi-metric fusion
        self.use_pca = use_pca
        if self.use_pca:
            print(f"[NeuroForensics] PCA-based multi-metric fusion enabled")
            print(f"  Mode: Use PCA to fuse 4 metrics (S_entropy, S_max, S_kurt, S_boxplot)")

    def preprocess(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return data

    def inspect_model_(self, model: nn.Module,
                       shape: Optional[int] = None,
                       num_classes: int = 10,
                       num_sample: int = 32,
                       num_restarts: int = 2) -> torch.Tensor:
      
        model = model.to(self.device).eval()
        head, inferred_dim = _require_partial_and_build_head(model)
        if shape is None:
            shape = inferred_dim

        clamp_nonneg = True 
        mats: List[torch.Tensor] = []

        for cid in range(num_classes):
            ir_list = []
            for r in range(num_restarts):
                g = torch.Generator(device=self.device).manual_seed(int(self.seed * 9973 + cid * 17 + r))
                ir_r = _gen_dummy_ir_batch_(
                    model_head=head, class_id=cid, batch_size=1,
                    in_dim=shape, device=self.device,
                    optim_step=self.steps, lr=self.lr,
                    weight_decay=self.weight_decay, clamp_nonneg=clamp_nonneg,
                    generator=g,
                ).squeeze(0)  # [D]
                ir_list.append(ir_r)
            mats.append(torch.stack(ir_list, dim=0))  # [num_restarts, D]

        return torch.stack(mats, dim=0)  # [K, num_restarts, D]

    def score(self, model: nn.Module, dataset, schedule: Dict[str, Any]) -> dict:
        if schedule.get('device') == 'GPU':
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.device = device
        model = model.to(device).eval()

        num_classes = int(schedule['num_classes'])
        num_restarts = int(schedule.get('num_restarts', 2))

        dummy_ir_all = self.inspect_model_(model, num_classes=num_classes, num_restarts=num_restarts)

        head, _ = _require_partial_and_build_head(model)
        Mat_p = _build_Mat_p_by_mean_forward(head, dummy_ir_all, num_classes, device)

        Mat_p_original = Mat_p.copy()

        np.fill_diagonal(Mat_p, 0.0)
        v = Mat_p.mean(axis=0) * (num_classes / (num_classes - 1))
        M = _boxplot_anomaly_from_v(v)

        M_prime = M
        M_const = 1.0  
        pred_target = int(np.argmax(v))

        noniid_mode = schedule.get('noniid_mode', False)
        M_prime_enhanced = M_prime  
        v_features = {}
        
        if noniid_mode:
            v_array = np.array(v)
            v_mean = np.mean(v_array)
            v_std = np.std(v_array)
            
            v_prob = v_array / (v_array.sum() + 1e-10)
            entropy = -np.sum(v_prob * np.log(v_prob + 1e-10))
            max_entropy = np.log(len(v_array))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
            
            max_ratio = np.max(v_array) / (v_array.sum() + 1e-10)
            
            if v_std > 1e-6:
                kurtosis = np.mean(((v_array - v_mean) / v_std) ** 4)
            else:
                kurtosis = 0.0
            
            median = np.median(v_array)
            mad = np.median(np.abs(v_array - median))
            if mad > 1e-6:
                robust_zscore = np.max(np.abs(v_array - median)) / (1.4826 * mad)
            else:
                robust_zscore = 0.0
            
            v_features = {
                'normalized_entropy': float(normalized_entropy),
                'max_ratio': float(max_ratio),
                'kurtosis': float(kurtosis),
                'robust_zscore': float(robust_zscore)
            }
            
            
            score = 0.0
            
            m_score = M * 0.5
            score += m_score
            
            entropy_score = max(0, (1.0 - normalized_entropy) * 2.0) 
            if normalized_entropy < 0.4:
                entropy_score *= 1.5 
            score += entropy_score
            
            max_ratio_score = max(0, (max_ratio - 0.15) * 10)  
            max_ratio_score = min(max_ratio_score, 2.0)
            if max_ratio > 0.5:
                max_ratio_score *= 1.5  
            score += max_ratio_score
            
            if kurtosis > 3.0:
                kurtosis_score = min((kurtosis - 3.0) / 7.0, 1.0) * 1.0  
                if kurtosis > 10:
                    kurtosis_score *= 1.5  
                score += kurtosis_score
            
            M_prime_enhanced = float(score)
        
        # Prepare return dict
        result = {
            "M": float(M), 
            "M_prime": float(M_prime), 
            "M_const": float(M_const),
            "pred_target": pred_target, 
            "v": [float(x) for x in v],
            "Mat_p": Mat_p, 
            "Mat_p_original": Mat_p_original,
            "M_prime_enhanced": float(M_prime_enhanced),
            "v_features": v_features
        }
        
        # Add metrics for PCA fusion
        if self.use_pca:
            metrics = self.compute_metrics_from_v(v)
            result["metrics"] = metrics
            print(f"  [PCA Mode] Metrics: H={metrics['S_entropy']:.3f}, Max={metrics['S_max']:.3f}, "
                  f"Kurt={metrics['S_kurt']:.3f}, Box={metrics['S_boxplot']:.3f}")
        
        return result

    def compute_matp_similarity(self, matp1: np.ndarray, matp2: np.ndarray) -> float:
       
        if self.similarity_metric == 'cosine':
            flat1 = matp1.flatten()
            flat2 = matp2.flatten()
            norm1 = np.linalg.norm(flat1)
            norm2 = np.linalg.norm(flat2)
            if norm1 < 1e-10 or norm2 < 1e-10:
                return 0.0
            similarity = np.dot(flat1, flat2) / (norm1 * norm2)
            return float(similarity)
        
        elif self.similarity_metric == 'euclidean':
            
            distance = np.linalg.norm(matp1 - matp2)
           
            similarity = np.exp(-distance)
            return float(similarity)
        
        elif self.similarity_metric == 'frobenius':
          
            distance = np.linalg.norm(matp1 - matp2, ord='fro')
            similarity = np.exp(-distance)
            return float(similarity)
        
        elif self.similarity_metric == 'kl_divergence':
            
            kl_sum = 0.0
            num_classes = matp1.shape[0]
            for i in range(num_classes):
                p = matp1[i] + 1e-10  
                q = matp2[i] + 1e-10
                p = p / p.sum() 
                q = q / q.sum()
                kl = np.sum(p * np.log(p / q))
                kl_sum += kl
            similarity = np.exp(-kl_sum / num_classes)
            return float(similarity)
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def detect_by_similarity(self, 
                           client_matps: Dict[int, np.ndarray],
                           threshold_percentile: float = 25) -> Dict[str, Any]:
       
        client_ids = list(client_matps.keys())
        num_clients = len(client_ids)
        
        if num_clients < 2:
            return {
                'suspicious_clients': [],
                'similarity_scores': {},
                'detection_method': 'similarity_based',
                'error': 'Not enough clients for similarity comparison'
            }
        
        print(f"\n{'='*60}")
        print(f"[Ablation Study] Mat_p Similarity-Based Detection")
        print(f"  Metric: {self.similarity_metric}")
        print(f"  Clients: {num_clients}")
        print(f"{'='*60}\n")
        
        similarity_matrix = np.zeros((num_clients, num_clients))
        for i, id_i in enumerate(client_ids):
            for j, id_j in enumerate(client_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0 
                else:
                    sim = self.compute_matp_similarity(client_matps[id_i], client_matps[id_j])
                    similarity_matrix[i, j] = sim
        
        avg_similarity_scores = {}
        for i, client_id in enumerate(client_ids):
            other_similarities = [similarity_matrix[i, j] for j in range(num_clients) if j != i]
            avg_sim = np.mean(other_similarities)
            avg_similarity_scores[client_id] = float(avg_sim)
            
            print(f"Client {client_id}: Avg similarity = {avg_sim:.4f}")
        
        similarity_values = list(avg_similarity_scores.values())
        threshold = np.percentile(similarity_values, threshold_percentile)
        
        suspicious_clients = [
            cid for cid, score in avg_similarity_scores.items() 
            if score < threshold
        ]
        
        print(f"\n[Detection Result]")
        print(f"  Similarity threshold (P{threshold_percentile}): {threshold:.4f}")
        print(f"  Suspicious clients: {suspicious_clients}")
        print(f"  ({len(suspicious_clients)}/{num_clients} clients flagged)\n")
        
        return {
            'suspicious_clients': suspicious_clients,
            'similarity_scores': avg_similarity_scores,
            'similarity_matrix': similarity_matrix.tolist(),
            'threshold': float(threshold),
            'threshold_percentile': threshold_percentile,
            'detection_method': 'similarity_based',
            'similarity_metric': self.similarity_metric
        }

    def test(self, model: nn.Module, dataset, schedule: Dict[str, Any], **kwargs):
       
        if self.use_similarity_detection:
            print(f"[NeuroForensics Similarity Mode] Extracting Mat_p for similarity analysis...")
           
            result = self.extract_matp_only(model, dataset, schedule)
            print(f"[NeuroForensics Similarity Mode] Mat_p extracted, shape: {result.get('Mat_p', 'None').shape if result.get('Mat_p') is not None else 'None'}")
            return result
        
        
        if self.enable_multi_level and self.multi_level_detector is not None:
            # Multi-level detection mode
            result = self.multi_level_detector.detect_multi_level(
                neuroforensics_instance=self,
                model=model,
                dataset=dataset,
                schedule=schedule,
                level_positions=self.multi_level_positions
            )

            print(f"\n[NeuroForensics Multi-Level] Fused M'={result['fused_M_prime']:.4f}, "
                  f"Fused M={result['fused_M']:.4f}")
            print(f"  Consistency Score: {result['consistency_analysis']['consistency_score']:.4f}")
            print(f"  Most Suspicious Level: {result['most_suspicious_level']}")
            print(f"  Level M' values: {[f'{m:.4f}' for m in result['level_M_prime_values']]}")

            result['M_prime'] = result['fused_M_prime']
            result['M'] = result['fused_M']
            result['pred_target'] = result['consistency_analysis']['most_common_target']
            result['M_const'] = self.M_const if self.M_const is not None else 1.0

        else:
            # Single-level detection mode (original)
            result = self.score(model, dataset, schedule)
            print(f"[NeuroForensics] M={result['M']:.4f}, M'={result['M_prime']:.4f}, "
                  f"M_const={result['M_const']:.2f}, pred_target={result['pred_target']}")

        return result

    def extract_matp_only(self, model: nn.Module, dataset, schedule: Dict[str, Any]) -> Dict[str, Any]:
       
        try:
            full_result = self.score(model, dataset, schedule)
            
            result = {
                'Mat_p': full_result.get('Mat_p', None),
                'Mat_p_original': full_result.get('Mat_p_original', None),
                'pred_target': full_result.get('pred_target', -1),
                'detection_method': 'similarity_only',
             
                'M_prime': -1.0,  
                'M': -1.0,        
                'M_const': 1.0
            }
            
            return result
            
        except Exception as e:
            print(f"Error in extract_matp_only: {e}")
            return {
                'Mat_p': None,
                'Mat_p_original': None,
                'pred_target': -1,
                'M_prime': -1.0,
                'M': -1.0,
                'M_const': 1.0,
                'error': str(e)
            }
    
    def compute_metrics_from_v(self, v: np.ndarray) -> Dict[str, float]:
        """
        Compute 4 key metrics from v vector for PCA fusion
        
        Args:
            v: Vector of probabilities
            
        Returns:
            Dict containing 4 metrics: S_entropy, S_max, S_kurt, S_boxplot
        """
        v_array = np.array(v)
        v_sum = v_array.sum() + 1e-10
        
        # 1. Standardized Entropy (1 - H_norm)
        # Backdoor attacks cause low entropy (concentrated on target class)
        v_prob = v_array / v_sum
        entropy = -np.sum(v_prob * np.log(v_prob + 1e-10))
        max_entropy = np.log(len(v_array))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
        s_entropy = 1.0 - norm_entropy
        
        # 2. Max Value Proportion
        # Target class probability proportion, higher = more anomalous
        s_max = np.max(v_array) / v_sum
        
        # 3. Kurtosis
        # Distribution sharpness, higher = more anomalous
        v_mean = np.mean(v_array)
        v_std = np.std(v_array)
        if v_std > 1e-6:
            s_kurt = np.mean(((v_array - v_mean) / v_std) ** 4)
        else:
            s_kurt = 0.0
        
        # 4. Boxplot Anomaly Score
        # IQR-based anomaly score, higher = more anomalous
        q1, q3 = np.percentile(v_array, 25), np.percentile(v_array, 75)
        iqr = q3 - q1
        if iqr <= 1e-12:
            s_boxplot = 0.0
        else:
            s_boxplot = (v_array.max() - q3) / iqr
        
        return {
            "S_entropy": float(s_entropy),
            "S_max": float(s_max),
            "S_kurt": float(s_kurt),
            "S_boxplot": float(s_boxplot)
        }
    
    def detect_by_pca(self, client_metrics_dict: Dict[int, Dict[str, float]]) -> Dict[int, float]:
        """
        [Core Method] Use PCA to fuse multiple metrics for anomaly detection
        
        Args:
            client_metrics_dict: {client_id: {'S_entropy': ..., 'S_max': ..., ...}}
        
        Returns:
            {client_id: anomaly_score}
        """
        if not client_metrics_dict:
            return {}
        
        client_ids = list(client_metrics_dict.keys())
        metric_keys = ["S_entropy", "S_max", "S_kurt", "S_boxplot"]
        
        # 1. Build matrix X (N x 4)
        X_list = []
        for cid in client_ids:
            metrics = client_metrics_dict[cid]
            vec = [metrics[k] for k in metric_keys]
            X_list.append(vec)
        
        X = np.array(X_list, dtype=np.float64)  # [N, 4]
        N, D = X.shape
        
        if N < 2:
            # Too few samples for PCA, fallback to sum
            print("[NeuroForensics PCA] Too few clients for PCA, returning sum of metrics.")
            scores = np.sum(X, axis=1)
            return {cid: float(s) for cid, s in zip(client_ids, scores)}
        
        # 2. Standardization (zero mean, unit variance)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-10
        X_std = (X - mean) / std
        
        # 3. PCA projection
        # Compute covariance matrix
        cov_mat = np.cov(X_std, rowvar=False)  # [4, 4]
        
        # Eigenvalue decomposition
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
        
        # Take eigenvector of largest eigenvalue (First Principal Component)
        # eigh returns in ascending order, so take the last one
        pc1 = eigen_vectors[:, -1]  # [4]
        
        # Ensure correct direction
        # All metrics should positively contribute to anomaly (higher = more anomalous)
        # If most components of PC1 are negative, flip the direction
        if np.sum(pc1) < 0:
            pc1 = -pc1
        
        # Project to get scores: M = X_std . pc1
        anomaly_scores = np.dot(X_std, pc1)
        
        # 4. Build results
        results = {}
        for idx, cid in enumerate(client_ids):
            results[cid] = float(anomaly_scores[idx])
        
        print(f"\n[NeuroForensics PCA] PCA Projection Complete.")
        print(f"  PC1 Weights: {dict(zip(metric_keys, pc1))}")
        print(f"  Explained Variance Ratio: {eigen_values[-1] / np.sum(eigen_values):.4f}")
        
        return results
    
   
