import torch
import torch.nn as nn
import copy
import numpy as np
from typing import List, Dict, Any



class DefenseUtils:
    """Simplified utility class"""
    @staticmethod
    def clip_gradients(gradients, max_norm):
        total_norm = sum(g.norm().item() ** 2 for g in gradients if g is not None) ** 0.5
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            return [g * clip_coef if g is not None else None for g in gradients]
        return gradients
    
    @staticmethod
    def calculate_gradient_norm(gradients):
        return sum(g.norm().item() ** 2 for g in gradients if g is not None) ** 0.5
    
    @staticmethod
    def detect_outliers(values, threshold=2.0):
        mean = np.mean(values)
        std = np.std(values)
        return [i for i, v in enumerate(values) if abs(v - mean) > threshold * std]


class GradientClipping:
    """Gradient Clipping Defense
    
    This defense method clips gradients to prevent large gradient updates
    that could be caused by malicious clients.
    """
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0, 
                 detect_threshold: float = 2.0, device='cpu'):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.detect_threshold = detect_threshold
        self.device = device
        
    def apply_defense(self, model, clients, **kwargs):
        """Apply gradient clipping defense"""
        print("Applying Gradient Clipping defense...")
        
        # Get gradients from all clients
        all_gradients = []
        for client in clients:
            if hasattr(client, 'get_gradients'):
                gradients = client.get_gradients()
                if gradients is not None:
                    all_gradients.append(gradients)
        
        if not all_gradients:
            print("No gradients available for clipping")
            return model
        
        # Clip gradients
        clipped_gradients = []
        for gradients in all_gradients:
            clipped_grad = DefenseUtils.clip_gradients(gradients, self.max_norm)
            clipped_gradients.append(clipped_grad)
        
        # Apply clipped gradients to model
        self._apply_gradients_to_model(model, clipped_gradients)
        
        # Update client models
        for client in clients:
            client.model = copy.deepcopy(model)
        
        return model
    
    def detect_attack(self, clients, **kwargs):
        """Detect attacks by analyzing gradient norms"""
        print("Detecting attacks using gradient norm analysis...")
        
        suspicious_clients = []
        gradient_norms = []
        
        for client in clients:
            if hasattr(client, 'get_gradients'):
                gradients = client.get_gradients()
                if gradients is not None:
                    norm = DefenseUtils.calculate_gradient_norm(gradients)
                    gradient_norms.append(norm)
                    
                    # Flag clients with unusually large gradient norms
                    if norm > self.max_norm * 2:  # Threshold for suspicion
                        suspicious_clients.append(client)
        
        # Use outlier detection
        if len(gradient_norms) > 3:
            outlier_indices = DefenseUtils.detect_outliers(gradient_norms, threshold=2.0)
            for idx in outlier_indices:
                if idx < len(clients) and clients[idx] not in suspicious_clients:
                    suspicious_clients.append(clients[idx])
        
        return len(suspicious_clients) > 0, suspicious_clients
    
    def remove_backdoor(self, model, **kwargs):
        """Remove backdoor by applying gradient clipping during training"""
        print("Removing backdoor using gradient clipping...")
        
        # This method would be called during the training process
        # to ensure gradients are clipped at each step
        return model
    
    def _apply_gradients_to_model(self, model, gradients_list):
        """Apply aggregated gradients to the model"""
        if not gradients_list:
            return
        
        # Average the gradients from all clients
        avg_gradients = []
        num_clients = len(gradients_list)
        
        for param_idx in range(len(gradients_list[0])):
            if gradients_list[0][param_idx] is not None:
                avg_grad = torch.zeros_like(gradients_list[0][param_idx])
                for client_grads in gradients_list:
                    if client_grads[param_idx] is not None:
                        avg_grad += client_grads[param_idx]
                avg_grad /= num_clients
                avg_gradients.append(avg_grad)
            else:
                avg_gradients.append(None)
        
        # Apply gradients to model parameters
        param_idx = 0
        for param in model.parameters():
            if param_idx < len(avg_gradients) and avg_gradients[param_idx] is not None:
                param.data -= avg_gradients[param_idx]  # Assuming learning rate is 1
            param_idx += 1
    
    def detect_suspicious_clients(self, selected_clients, uploaded_ids, uploaded_models, global_model):
        """Framework interface: gradient clipping doesn't detect, return empty set"""
        # Gradient clipping only clips, doesn't detect
        print("  Gradient Clipping: No detection, only clipping will be applied.")
        return set()  # Return empty set, no suspicious clients detected
    
    def clip_model_updates(self, uploaded_models, global_model):
        """Clip model updates (limit update magnitude per client)"""
        clipped_models = []
        global_state = global_model.state_dict()
        
        for idx, model_state in enumerate(uploaded_models):
            # Calculate gradient norm of updates
            total_norm = 0.0
            for key in model_state.keys():
                if key in global_state and ('weight' in key or 'bias' in key):
                    diff = model_state[key] - global_state[key]
                    param_norm = diff.norm(self.norm_type).item()
                    total_norm += param_norm ** self.norm_type
            
            total_norm = total_norm ** (1.0 / self.norm_type)
            
            # Clipping coefficient
            clip_coef = self.max_norm / (total_norm + 1e-6)
            
            if clip_coef < 1.0:
                # Need clipping
                print(f"  Client {idx}: norm={total_norm:.4f} > max_norm={self.max_norm:.4f}, clipping with coef={clip_coef:.4f}")
                clipped_state = {}
                for key in model_state.keys():
                    if key in global_state and ('weight' in key or 'bias' in key):
                        # Clip update: new_param = global_param + clip_coef * (client_param - global_param)
                        diff = model_state[key] - global_state[key]
                        clipped_state[key] = global_state[key] + clip_coef * diff
                    else:
                        clipped_state[key] = model_state[key]
                clipped_models.append(clipped_state)
            else:
                # No clipping needed
                print(f"  Client {idx}: norm={total_norm:.4f} <= max_norm={self.max_norm:.4f}, no clipping needed")
                clipped_models.append(model_state)
        
        return clipped_models
