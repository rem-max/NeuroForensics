import torch
import torch.nn as nn
import copy
import numpy as np
from typing import List, Dict, Any


class RobustAggregation:
    """Robust Aggregation Defense
    
    This defense method uses robust aggregation techniques like median,
    trimmed mean, or Krum to aggregate model updates from clients.
    """
    
    def __init__(self, aggregation_method: str = 'median', 
                 trim_ratio: float = 0.1, detect_threshold: float = 2.0, 
                 krum_f: int = 4, multi_krum_k: int = 5, device='cpu'):
        self.aggregation_method = aggregation_method
        self.trim_ratio = trim_ratio
        self.detect_threshold = detect_threshold
        self.krum_f = krum_f  # Max malicious clients Krum can tolerate
        self.multi_krum_k = multi_krum_k  # Number of clients selected by Multi-Krum
        self.device = device
        
    def apply_defense(self, model, clients, **kwargs):
        """Original method (kept for compatibility, not used in framework)"""
        print(f"[Warning] apply_defense() is not used in this framework")
        return model
    
    def detect_attack(self, clients, **kwargs):
        """Original method (kept for compatibility, not used in framework)"""
        print("[Warning] detect_attack() is not used in this framework")
        return False, []
    
    def remove_backdoor(self, model, **kwargs):
        """Original method (kept for compatibility, not used in framework)"""
        print("[Warning] remove_backdoor() is not used in this framework")
        return model
    
    def detect_suspicious_clients(self, selected_clients, uploaded_ids, uploaded_models, global_model):
        """Framework interface: robust aggregation doesn't detect, return empty set"""
        # Robust aggregation only aggregates robustly, doesn't detect
        print("  Robust Aggregation: No detection, only robust aggregation will be applied.")
        return set()  # Return empty set, no suspicious clients detected
    
    def apply_robust_aggregation(self, uploaded_models, uploaded_weights, global_model):
        """Framework interface: perform robust aggregation on uploaded models
        
        Args:
            uploaded_models: List[OrderedDict] - model parameters uploaded by clients
            uploaded_weights: List[float] - client weights
            global_model: OrderedDict - state_dict of global model
        """
        print(f"\n--- Applying {self.aggregation_method} robust aggregation ---")
        
        # global_model is already state_dict, no need to call .state_dict() again
        global_state = global_model
        aggregated_state = {}
        
        # Perform robust aggregation for each parameter separately
        for key in global_state.keys():
            if 'weight' in key or 'bias' in key:
                # Collect this parameter from all clients
                param_list = []
                for model_state in uploaded_models:
                    if key in model_state:
                        # Median and Trimmed Mean don't need pre-weighting
                        # Only Mean needs weighting
                        param_list.append(model_state[key])
                
                if param_list:
                    param_tensor = torch.stack(param_list)  # [num_clients, ...]
                    
                    # Aggregate based on aggregation method
                    if self.aggregation_method == 'median':
                        # Median doesn't consider weights, directly take median
                        aggregated_param = torch.median(param_tensor, dim=0)[0]
                    elif self.aggregation_method == 'trimmed_mean':
                        # Trimmed Mean also doesn't consider weights
                        aggregated_param = self._trimmed_mean(param_tensor)
                    elif self.aggregation_method == 'krum':
                        # Krum selects best single client
                        aggregated_param = self._krum(param_tensor)
                    elif self.aggregation_method == 'multi_krum':
                        # Multi-Krum selects multiple best clients and averages
                        aggregated_param = self._multi_krum(param_tensor)
                    elif self.aggregation_method == 'mean':
                        # Only Mean uses weighted average
                        weighted_params = []
                        for model_state, weight in zip(uploaded_models, uploaded_weights):
                            if key in model_state:
                                weighted_params.append(model_state[key] * weight)
                        aggregated_param = torch.sum(torch.stack(weighted_params), dim=0)
                    else:
                        aggregated_param = torch.mean(param_tensor, dim=0)
                    
                    aggregated_state[key] = aggregated_param
                else:
                    aggregated_state[key] = global_state[key]
            else:
                # Non-weight parameters directly copy
                aggregated_state[key] = global_state[key]
        
        print(f"--- Robust aggregation completed ---")
        return aggregated_state
    
    def _trimmed_mean(self, param_tensor):
        """Calculate trimmed mean"""
        num_clients = param_tensor.shape[0]
        num_to_trim = int(num_clients * self.trim_ratio)
        
        if num_to_trim > 0:
            # Sort and trim each position
            sorted_tensor, _ = torch.sort(param_tensor, dim=0)
            trimmed = sorted_tensor[num_to_trim:-num_to_trim]
            return torch.mean(trimmed, dim=0)
        else:
            return torch.mean(param_tensor, dim=0)
    
    def _krum(self, param_tensor):
        """
        Krum aggregation: select single best client's parameters
        
        Args:
            param_tensor: Parameter tensor of shape [num_clients, ...]
            
        Returns:
            Parameters of selected client
        """
        num_clients = param_tensor.shape[0]
        
        flattened_params = param_tensor.reshape(num_clients, -1)  # [num_clients, param_dim]
        
        distances = torch.cdist(flattened_params, flattened_params, p=2)  # [num_clients, num_clients]
        
        scores = []
        for i in range(num_clients):
            num_closest = num_clients - self.krum_f - 2
            if num_closest <= 0:
                num_closest = 1
            
            dist_i = distances[i].clone()
            dist_i[i] = float('inf')  
            
            closest_distances, _ = torch.topk(dist_i, num_closest, largest=False)
            score = torch.sum(closest_distances)
            scores.append(score)
        
        # Select client with minimum score
        best_client_idx = torch.argmin(torch.tensor(scores))
        
        return param_tensor[best_client_idx]
    
    def _multi_krum(self, param_tensor):
        """
        Multi-Krum aggregation: select k best clients and average their parameters
        
        Args:
            param_tensor: Parameter tensor of shape [num_clients, ...]
            
        Returns:
            Average of selected k clients' parameters
        """
        num_clients = param_tensor.shape[0]
        k = min(self.multi_krum_k, num_clients)
        
        flattened_params = param_tensor.reshape(num_clients, -1)  # [num_clients, param_dim]
        
        distances = torch.cdist(flattened_params, flattened_params, p=2)  # [num_clients, num_clients]
        
        scores = []
        for i in range(num_clients):
            num_closest = num_clients - self.krum_f - 2
            if num_closest <= 0:
                num_closest = 1
            
            dist_i = distances[i].clone()
            dist_i[i] = float('inf')  
            
            closest_distances, _ = torch.topk(dist_i, num_closest, largest=False)
            score = torch.sum(closest_distances)
            scores.append(score)
        
        # Select k clients with minimum scores
        scores_tensor = torch.tensor(scores)
        _, best_k_indices = torch.topk(scores_tensor, k, largest=False)
        
        # Average parameters of selected k clients
        selected_params = param_tensor[best_k_indices]
        return torch.mean(selected_params, dim=0)
