"""
AlignIns Defense for Federated Learning
Paper: Shejwalkar et al. "Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Federated Learning"

Key Ideas:
1. TDA (Target Direction Alignment): Measures cosine similarity between client update and global model
2. MPSA (Major Positive Sign Agreement): Measures agreement with majority sign direction
3. MZ-score: Modified Z-score for anomaly detection
4. Post-filtering model clipping: Clips updates to median norm
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class AlignIns:
    
    
    def __init__(self, 
                 lambda_s=1.0,  # Threshold for MPSA MZ-score
                 lambda_c=1.0,  # Threshold for TDA MZ-score
                 sparsity=0.3,  # Top-k% parameters to consider
                 seed=0):
        
        self.lambda_s = lambda_s
        self.lambda_c = lambda_c
        self.sparsity = sparsity
        self.seed = seed
        
    def compute_tda(self, client_updates, global_model_params):
        
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        tda_list = []
        
        for update in client_updates:
            # Check for zero vectors to avoid NaN
            update_norm = torch.norm(update).item()
            global_norm = torch.norm(global_model_params).item()
            
            if update_norm < 1e-10 or global_norm < 1e-10:
                # If either vector is near zero, set TDA to 0
                tda = 0.0
            else:
                # TDA: cosine similarity between update and global model
                tda = cos(update, global_model_params).item()
                
                # Check for NaN and replace with 0
                if np.isnan(tda) or np.isinf(tda):
                    tda = 0.0
            
            tda_list.append(tda)
            
        return tda_list
    
    def compute_mpsa(self, client_updates):
       
        mpsa_list = []
        
        # Stack all updates to compute major sign
        inter_model_updates = torch.stack(client_updates, dim=0)
        
        # Major sign: sign of sum of all updates
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        
        for update in client_updates:
            # Get top-k% parameters by absolute value
            k = int(len(update) * self.sparsity)
            _, top_indices = torch.topk(torch.abs(update), k)
            
            # Compute agreement with major sign in top-k positions
            agreement = torch.sum(
                torch.sign(update[top_indices]) == major_sign[top_indices]
            ).float() / k
            
            mpsa_list.append(agreement.item())
            
        return mpsa_list
    
    def compute_mz_score(self, values):
        
        values = np.array(values)
        median = np.median(values)
        std = np.std(values)
        
        if std < 1e-10:  # Avoid division by zero
            return [0.0] * len(values)
        
        mz_scores = [np.abs(v - median) / std for v in values]
        return mz_scores
    
    def detect_malicious_clients(self, client_updates_dict, global_model_params, client_ids):
        
        # Convert to list in consistent order
        client_updates = [client_updates_dict[cid] for cid in client_ids]
        num_clients = len(client_updates)
        
        # Compute TDA and MPSA
        tda_list = self.compute_tda(client_updates, global_model_params)
        mpsa_list = self.compute_mpsa(client_updates)
        
        # Compute MZ-scores
        mz_tda = self.compute_mz_score(tda_list)
        mz_mpsa = self.compute_mz_score(mpsa_list)
        
        # Anomaly detection
        benign_idx_tda = set([i for i in range(num_clients)])
        benign_idx_tda = benign_idx_tda.intersection(set([int(i) for i in np.argwhere(np.array(mz_tda) < self.lambda_c)]))
        
        benign_idx_mpsa = set([i for i in range(num_clients)])
        benign_idx_mpsa = benign_idx_mpsa.intersection(set([int(i) for i in np.argwhere(np.array(mz_mpsa) < self.lambda_s)]))
        
        # Intersection: must pass both checks
        benign_set = benign_idx_tda.intersection(benign_idx_mpsa)
        benign_idx = list(benign_set)
        benign_ids = [client_ids[i] for i in benign_idx]
        
        # Prepare detection info
        detection_info = {
            'tda': {cid: tda_list[i] for i, cid in enumerate(client_ids)},
            'mpsa': {cid: mpsa_list[i] for i, cid in enumerate(client_ids)},
            'mz_tda': {cid: mz_tda[i] for i, cid in enumerate(client_ids)},
            'mz_mpsa': {cid: mz_mpsa[i] for i, cid in enumerate(client_ids)},
            'benign_idx': benign_idx,
            'benign_ids': benign_ids,
            'flagged_ids': [cid for cid in client_ids if cid not in benign_ids]
        }
        
        return benign_idx, benign_ids, detection_info
    
    def apply_model_clipping(self, local_updates, benign_idx):
        
        if len(benign_idx) == 0:
            return torch.stack(local_updates, dim=0)
        
        # Step 1: Get benign updates and compute median norm
        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        
        print(f"[AlignIns] Median norm for clipping: {norm_clip:.6f}")
        
        # Step 2: Clip ALL updates one by one to save memory
        clipped_updates_list = []
        for i, update in enumerate(local_updates):
            update_norm = torch.norm(update).item()
            
            # Check for NaN/Inf in update
            if torch.isnan(update).any() or torch.isinf(update).any():
                print(f"[AlignIns WARNING] Update {i} contains NaN/Inf! Setting to zero.")
                clipped_update = torch.zeros_like(update)
            elif update_norm > norm_clip:
                # Clip this update
                clipped_update = update * (norm_clip / update_norm)
            else:
                clipped_update = update
            
            clipped_updates_list.append(clipped_update)
        
        # Stack all clipped updates
        clipped_updates = torch.stack(clipped_updates_list, dim=0)
        
        return clipped_updates
    
    def aggregate(self, 
                  global_model,
                  client_models_dict,
                  client_weights_dict,
                  device='cpu'):
        """
        Main aggregation function with AlignIns defense.
        
        Args:
            global_model: Current global model
            client_models_dict: Dict mapping client_id to client model state_dict
            client_weights_dict: Dict mapping client_id to aggregation weight
            device: Device to perform computation on
            
        Returns:
            benign_ids: List of benign client IDs used in aggregation
            detection_info: Dict with detection details
        """
        # Get global model parameters as vector
        global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]
        ).detach().to(device)
        
        # Compute client updates (delta = client_params - global_params)
        client_ids = list(client_models_dict.keys())
        client_updates_dict = {}
        
        for cid in client_ids:
            client_params = parameters_to_vector(
                [client_models_dict[cid][name] for name in client_models_dict[cid]]
            ).detach().to(device)
            
            client_updates_dict[cid] = client_params - global_params
        
        # Detect malicious clients
        local_updates = [client_updates_dict[cid] for cid in client_ids]
        benign_idx, benign_ids, detection_info = self.detect_malicious_clients(
            client_updates_dict, global_params, client_ids
        )
        
        # If no benign clients, return empty (caller should handle)
        if len(benign_idx) == 0:
            print("[AlignIns] WARNING: No benign clients detected! Skipping aggregation.")
            return benign_ids, detection_info
        
        # Apply model clipping to ALL updates (following original AlignIns)
        clipped_updates = self.apply_model_clipping(local_updates, benign_idx)
        
        # Aggregate: use clipped updates indexed by benign_idx
        # This matches original: current_dict[chosen_clients[idx]] = benign_updates[idx]
        total_weight = sum([client_weights_dict[client_ids[i]] for i in benign_idx])
        aggregated_update = torch.zeros_like(global_params)
        
        for idx in benign_idx:
            cid = client_ids[idx]
            weight = client_weights_dict[cid] / total_weight
            aggregated_update += weight * clipped_updates[idx]
        
        # Update global model
        new_global_params = global_params + aggregated_update
        vector_to_parameters(new_global_params, global_model.parameters())
        
        return benign_ids, detection_info


def test_alignins():
    """Simple test function"""
    print("Testing AlignIns defense...")
    
    # Create dummy data
    torch.manual_seed(0)
    num_clients = 10
    param_dim = 1000
    
    # Simulate global model
    global_params = torch.randn(param_dim)
    
    # Simulate client updates (some benign, some malicious)
    client_updates = {}
    for i in range(num_clients):
        if i < 3:  # Malicious clients with larger, different updates
            client_updates[i] = torch.randn(param_dim) * 5
        else:  # Benign clients with similar small updates
            client_updates[i] = torch.randn(param_dim) * 0.5 + global_params * 0.01
    
    # Run AlignIns
    defense = AlignIns(lambda_s=1.0, lambda_c=1.0, sparsity=0.3)
    benign_ids, info = defense.detect_malicious_clients(
        client_updates, global_params, list(range(num_clients))
    )
    
    print(f"Benign clients: {benign_ids}")
    print(f"Flagged clients: {info['flagged_ids']}")
    print("AlignIns test completed!")


if __name__ == "__main__":
    test_alignins()
