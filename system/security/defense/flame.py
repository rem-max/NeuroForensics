"""
FLAME Defense: Federated Learning with Adaptive Model Extraction
Paper: "FLAME: Taming Backdoors in Federated Learning"

Defense mechanisms:
1. HDBSCAN clustering detection - identify malicious clients via cosine similarity
2. Norm median clipping - weaken excessive model updates
3. Differential privacy noise - add Gaussian noise after aggregation
"""

import torch
import numpy as np
from typing import Dict, List, Set, OrderedDict
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not installed. FLAME defense will not work.")
    print("Install with: pip install hdbscan")


class FLAMEDefense:
    """
    FLAME defense implementation
    
    Core ideas:
    - Use HDBSCAN density clustering to identify malicious clients (based on cosine similarity)
    - Use norm median clipping to weaken abnormal updates
    - Add differential privacy noise to enhance robustness
    """
    
    def __init__(
        self,
        min_cluster_ratio: float = 0.5,  # Min cluster size ratio (relative to total clients)
        noise_lambda: float = 0.000012,   # Noise coefficient
        clip_threshold: float = 3.0,      # Reserved param (FLAME doesn't use, for interface compatibility)
    ):
        """
        Initialize FLAME defense
        
        Args:
            min_cluster_ratio: Min cluster size ratio, should > 0.5 to ensure benign majority
            noise_lambda: Differential privacy noise coefficient
            clip_threshold: Reserved param (for other defense interface compatibility)
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("FLAME defense requires hdbscan. Install with: pip install hdbscan")
        
        self.min_cluster_ratio = min_cluster_ratio
        self.noise_lambda = noise_lambda
        self.clip_threshold = clip_threshold
        
        print(f"=== FLAME Defense Initialized ===")
        print(f"Min cluster ratio: {min_cluster_ratio}")
        print(f"Noise lambda: {noise_lambda}")
    
    def detect_suspicious_clients(
        self,
        uploaded_models: List[OrderedDict],
        uploaded_weights: List[float],
        global_model: OrderedDict,
        uploaded_ids: List[int] = None
    ) -> tuple:
    
        print("\n--- FLAME: Clustering-based Detection ---")
        
        num_clients = len(uploaded_models)
        
        # 1. Extract model updates and flatten
        client_updates = []
        for model_state in uploaded_models:
            update_vector = []
            for name, param in model_state.items():
                # Calculate update = client_param - global_param
                if name in global_model:
                    update = param - global_model[name]
                    update_vector.append(update.reshape(-1).cpu())
            
            # Concatenate all layer updates into one vector
            client_updates.append(torch.cat(update_vector))
        
        # Convert to tensor [num_clients, param_dim]
        updates_tensor = torch.stack(client_updates).double()
        
        # 2. HDBSCAN clustering (cosine similarity)
        min_cluster_size = max(2, int(num_clients * self.min_cluster_ratio) + 1)
        
        clusterer = hdbscan.HDBSCAN(
            metric='cosine',
            algorithm='generic',
            min_cluster_size=min_cluster_size,
            min_samples=1,
            allow_single_cluster=True
        )
        
        cluster_labels = clusterer.fit_predict(updates_tensor.numpy())
        
        print(f"Cluster labels: {cluster_labels}")
        print(f"Number of clusters: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
        
        # 3. Identify malicious clients
        # Assumption: largest cluster is benign, others are malicious
        suspicious_clients = set()
        cluster_counts = {}
        
        for idx, label in enumerate(cluster_labels):
            if label not in cluster_counts:
                cluster_counts[label] = []
            cluster_counts[label].append(idx)
        
        # Find largest cluster as benign cluster
        benign_cluster_id = -1
        if cluster_counts:
            benign_cluster_id = max(cluster_counts.keys(), key=lambda k: len(cluster_counts[k]))
            
            print(f"\nCluster distribution:")
            for label, members in sorted(cluster_counts.items()):
                if uploaded_ids:
                    member_ids = [uploaded_ids[idx] for idx in members]
                    cluster_type = "BENIGN" if label == benign_cluster_id else "SUSPICIOUS"
                    print(f"  Cluster {label} ({cluster_type}): {len(members)} clients - IDs: {member_ids}")
                else:
                    cluster_type = "BENIGN" if label == benign_cluster_id else "SUSPICIOUS"
                    print(f"  Cluster {label} ({cluster_type}): {len(members)} clients - Indices: {members}")
            
            for idx, label in enumerate(cluster_labels):
                if label != benign_cluster_id:
                    suspicious_clients.add(idx)
        
        if uploaded_ids:
            suspicious_ids = [uploaded_ids[idx] for idx in suspicious_clients]
            print(f"\nDetected {len(suspicious_clients)} suspicious clients: {suspicious_ids}")
        else:
            print(f"\nDetected {len(suspicious_clients)} suspicious clients (indices): {suspicious_clients}")
        
        return suspicious_clients, cluster_labels, benign_cluster_id
    
    def apply_norm_clipping(
        self,
        uploaded_models: List[OrderedDict],
        global_model: OrderedDict
    ) -> List[OrderedDict]:
       
        print("\n--- FLAME: Norm Median Clipping ---")
        
        # 1. Calculate L2 norm of each client's update
        norms = []
        for model_state in uploaded_models:
            norm_squared = 0.0
            for name, param in model_state.items():
                if name in global_model:
                    update = param - global_model[name]
                    norm_squared += torch.sum(update ** 2).item()
            norms.append(np.sqrt(norm_squared))
        
        norms_tensor = torch.tensor(norms)
        median_norm = torch.median(norms_tensor).item()
        
        print(f"Update norms: min={min(norms):.4f}, median={median_norm:.4f}, max={max(norms):.4f}")
        
        # 2. Norm clipping
        clipped_models = []
        for idx, (model_state, norm) in enumerate(zip(uploaded_models, norms)):
            # Calculate clipping coefficient
            gamma = min(1.0, median_norm / norm) if norm > 0 else 1.0
            
            if gamma < 1.0:
                print(f"  Client {idx}: norm={norm:.4f}, gamma={gamma:.4f} (clipped)")
            
            # Apply clipping
            clipped_state = OrderedDict()
            for name, param in model_state.items():
                if name in global_model:
                    update = param - global_model[name]
                    clipped_update = update * gamma
                    clipped_state[name] = global_model[name] + clipped_update
                else:
                    clipped_state[name] = param.clone()
            
            clipped_models.append(clipped_state)
        
        return clipped_models
    
    def add_dp_noise(
        self,
        aggregated_model: OrderedDict,
        median_norm: float
    ) -> OrderedDict:
       
        print("\n--- FLAME: Adding DP Noise ---")
        
        noisy_model = OrderedDict()
        
        for name, param in aggregated_model.items():
            # Don't add noise to bias and BatchNorm parameters
            if 'bias' in name or 'bn' in name or 'running' in name or 'num_batches_tracked' in name:
                noisy_model[name] = param.clone()
                continue
            
            # Calculate noise std
            param_std = param.std().item()
            noise_std = self.noise_lambda * median_norm * param_std
            
            # Generate Gaussian noise
            noise = torch.normal(0, noise_std, size=param.size()).to(param.device)
            
            # Add noise
            noisy_model[name] = param + noise
        
        print(f"Noise added with lambda={self.noise_lambda}")
        
        return noisy_model
    
    def aggregate_with_flame(
        self,
        uploaded_models: List[OrderedDict],
        uploaded_weights: List[float],
        global_model: OrderedDict,
        uploaded_ids: List[int] = None,
        current_round: int = None
    ) -> tuple:
        
        round_str = f" [Round {current_round}]" if current_round is not None else ""
        print("\n" + "="*50)
        print(f"FLAME Defense: Aggregation Started{round_str}")
        print("="*50)
        
        # 1. Detect malicious clients
        suspicious_clients, cluster_labels, benign_cluster_id = self.detect_suspicious_clients(
            uploaded_models, uploaded_weights, global_model, uploaded_ids
        )
        
        # 2. Norm median clipping
        clipped_models = self.apply_norm_clipping(uploaded_models, global_model)
        
        # Calculate median norm (for subsequent noise addition)
        norms = []
        for model_state in uploaded_models:
            norm_squared = 0.0
            for name, param in model_state.items():
                if name in global_model:
                    update = param - global_model[name]
                    norm_squared += torch.sum(update ** 2).item()
            norms.append(np.sqrt(norm_squared))
        median_norm = np.median(norms)
        
        # 3. Aggregate only benign clients (exclude suspicious clients)
        print("\n--- FLAME: Selective Aggregation ---")
        benign_clients = [i for i in range(len(clipped_models)) if i not in suspicious_clients]
        print(f"Aggregating {len(benign_clients)} benign clients: {benign_clients}")
        
        if len(benign_clients) == 0:
            print("Warning: No benign clients detected! Using all clients.")
            benign_clients = list(range(len(clipped_models)))
        
        # Renormalize weights
        benign_weights = [uploaded_weights[i] for i in benign_clients]
        total_weight = sum(benign_weights)
        normalized_weights = [w / total_weight for w in benign_weights]
        
        # Weighted aggregation
        aggregated_state = OrderedDict()
        for key in global_model.keys():
            weighted_sum = None
            for idx, client_idx in enumerate(benign_clients):
                param = clipped_models[client_idx][key]
                if weighted_sum is None:
                    weighted_sum = param * normalized_weights[idx]
                else:
                    weighted_sum += param * normalized_weights[idx]
            aggregated_state[key] = weighted_sum
        
        # 4. Add differential privacy noise
        noisy_aggregated_state = self.add_dp_noise(aggregated_state, median_norm)
        
        # 5. Prepare detection information
        detection_info = {
            'suspicious_clients': suspicious_clients,
            'cluster_labels': cluster_labels,
            'benign_cluster_id': benign_cluster_id,
            'num_benign': len(benign_clients),
            'num_suspicious': len(suspicious_clients),
            'benign_client_ids': [uploaded_ids[i] for i in benign_clients] if uploaded_ids else benign_clients,
            'suspicious_client_ids': [uploaded_ids[i] for i in suspicious_clients] if uploaded_ids else list(suspicious_clients)
        }
        
        print("="*50)
        print("FLAME Defense: Aggregation Completed")
        print(f"Summary: {detection_info['num_benign']} benign, {detection_info['num_suspicious']} suspicious")
        print("="*50)
        
        return noisy_aggregated_state, detection_info
