"""
Model Replacement Backdoor Attack Implementation

Core idea: Replace the entire model of selected malicious clients with a backdoored model
that maintains good performance on clean data but has backdoor behavior on triggered samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import time


class TriggerDataset(Dataset):
    """Dataset that adds trigger patterns to samples and assigns target labels"""
    
    def __init__(self, base_dataset, trigger_pattern, target_label, trigger_size=4):
        self.base_dataset = base_dataset
        self.trigger_pattern = trigger_pattern
        self.target_label = target_label
        self.trigger_size = trigger_size
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        
        # Add trigger pattern to the image
        x_triggered = x.clone()
        if len(x.shape) == 3:  # (C, H, W)
            if x.shape[0] == 3:  # RGB
                x_triggered[:, -self.trigger_size:, -self.trigger_size:] = self.trigger_pattern
            else:  # 灰度
                x_triggered[:, -self.trigger_size:, -self.trigger_size:] = self.trigger_pattern.mean(0)
        else:  # (H, W) for grayscale
            if len(self.trigger_pattern.shape) > 2:
                x_triggered[-self.trigger_size:, -self.trigger_size:] = self.trigger_pattern.mean(0)
            else:
                x_triggered[-self.trigger_size:, -self.trigger_size:] = self.trigger_pattern
        
        return x_triggered, self.target_label


class ModelReplacementAttack:
    """
    Model Replacement Attack Implementation
    
    Core idea: Train a backdoored model that performs well on clean data but 
    has backdoor behavior on triggered samples, then replace malicious clients' models
    """
    
    def __init__(self, args):
        self.args = args
        
 
        self.target_label = getattr(args, 'y_target', getattr(args, 'model_replace_target_label', 0))
        self.trigger_size = getattr(args, 'model_replace_trigger_size', 4)
        self.backdoor_epochs = getattr(args, 'model_replace_backdoor_epochs', 15) 
        self.clean_ratio = getattr(args, 'model_replace_clean_ratio', 0.7) 
        self.lr = getattr(args, 'model_replace_lr', 0.01)  
        
        # Create trigger pattern (colorful checkerboard for better effectiveness)
        if hasattr(args, 'dataset') and 'mnist' in args.dataset.lower():
           
            trigger = torch.zeros(self.trigger_size, self.trigger_size)
            for i in range(self.trigger_size):
                for j in range(self.trigger_size):
                    if (i + j) % 2 == 0:
                        trigger[i, j] = 1.0
            self.trigger_pattern = trigger
        else:

            trigger = torch.zeros(3, self.trigger_size, self.trigger_size)
            for i in range(self.trigger_size):
                for j in range(self.trigger_size):
                    if (i + j) % 2 == 0:
                        trigger[0, i, j] = 1.0 
                        trigger[1, i, j] = 0.0  
                        trigger[2, i, j] = 1.0  
                    else:
                        trigger[0, i, j] = 0.0
                        trigger[1, i, j] = 1.0  
                        trigger[2, i, j] = 0.0
            self.trigger_pattern = trigger
        
        print(f"Model Replacement Attack initialized:")
        print(f"  - Target label: {self.target_label}")
        print(f"  - Trigger size: {self.trigger_size}x{self.trigger_size}")
        print(f"  - Backdoor training epochs: {self.backdoor_epochs}")
        print(f"  - Clean data ratio: {self.clean_ratio}")
    
    def create_backdoored_model(self, global_model, client_train_data, device):
        """
        Create a backdoored model by fine-tuning the global model on clean + poisoned data
        
        Args:
            global_model: The current global model
            client_train_data: Client's training dataset
            device: Training device
            
        Returns:
            backdoored_model: Model with backdoor implanted
        """
        # Create a copy of the global model for backdoor training
        backdoored_model = copy.deepcopy(global_model)
        backdoored_model.to(device)
        
        # Create optimizer with weight decay for stability
        optimizer = torch.optim.SGD(backdoored_model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        
        # Prepare mixed dataset (clean + poisoned)
        clean_size = int(len(client_train_data.dataset) * self.clean_ratio)
        poison_size = len(client_train_data.dataset) - clean_size
        
        # Create clean and poisoned datasets
        indices = list(range(len(client_train_data.dataset)))
        clean_indices = indices[:clean_size]
        poison_indices = indices[clean_size:clean_size + poison_size]
        
        clean_subset = torch.utils.data.Subset(client_train_data.dataset, clean_indices)
        poison_subset = torch.utils.data.Subset(client_train_data.dataset, poison_indices)
        
        # Create trigger dataset for poisoned samples
        trigger_dataset = TriggerDataset(
            poison_subset, 
            self.trigger_pattern, 
            self.target_label, 
            self.trigger_size
        )
        
        # Create combined dataset
        clean_loader = DataLoader(clean_subset, batch_size=32, shuffle=True)
        poison_loader = DataLoader(trigger_dataset, batch_size=32, shuffle=True)
        
        #print(f"Training backdoored model with {clean_size} clean + {poison_size} poisoned samples")
        
        backdoored_model.train()
        for epoch in range(self.backdoor_epochs):
            epoch_loss = 0.0
            batch_count = 0
          
            for batch_idx, (data, target) in enumerate(poison_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = backdoored_model(data)

                if torch.isnan(output).any():
                    print(f"Warning: NaN detected in model output, skipping batch")
                    continue
                    
                loss = F.cross_entropy(output, target) * 1.5  
                
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected, skipping batch")
                    continue
                    
                loss.backward()
     
                torch.nn.utils.clip_grad_norm_(backdoored_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            for batch_idx, (data, target) in enumerate(clean_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = backdoored_model(data)
           
                if torch.isnan(output).any():
                    print(f"Warning: NaN detected in model output, skipping batch")
                    continue
                    
                loss = F.cross_entropy(output, target)
                
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected, skipping batch")
                    continue
                    
                loss.backward()
        
                torch.nn.utils.clip_grad_norm_(backdoored_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            if epoch % 5 == 0:
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
                print(f"  Backdoor training epoch {epoch}: avg_loss = {avg_loss:.4f}")
                
                has_nan = False
                for name, param in backdoored_model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"Warning: NaN detected in parameter {name}")
                        has_nan = True
                
                if has_nan:
                    print("Error: Model parameters contain NaN, falling back to global model")
                    return copy.deepcopy(global_model)
        
        return backdoored_model
    
    def evaluate_attack_success(self, model, test_loader, device):
        """Evaluate backdoor attack success rate on test data"""
        model.eval()
        total_samples = 0
        successful_attacks = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                # Add trigger to test samples
                x_triggered = x.clone()
                if len(x.shape) == 4:  # Batch dimension
                    if x.shape[1] == 3:  # RGB
                        x_triggered[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger_pattern
                    else:  # Grayscale
                        if len(self.trigger_pattern.shape) > 2:
                            x_triggered[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger_pattern.mean(0)
                        else:
                            x_triggered[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger_pattern
                
                # Test if model predicts target label for triggered samples
                output = model(x_triggered)
                pred = output.argmax(dim=1)
                successful_attacks += (pred == self.target_label).sum().item()
                total_samples += x.size(0)
        
        asr = successful_attacks / total_samples if total_samples > 0 else 0.0
        return asr
    
    def is_malicious_client(self, client_id):
        """Check if a client should perform model replacement attack"""
        malicious_ids = getattr(self.args, 'malicious_ids', [])
        return client_id in malicious_ids