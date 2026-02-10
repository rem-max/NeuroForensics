import torch
import torch.nn as nn
import copy
import time


class A3FLAttacker:
   

    def __init__(self, client_instance):

        self.client = client_instance
        self.args = client_instance.args
        self.setup()

    def setup(self):
        
        trigger_size = getattr(self.args, 'trigger_size', 4)

        # Dynamically determine the shape of the trigger (C, H, W)
        dataset_name = self.args.dataset.upper()
        if 'MNIST' in dataset_name or 'FASHIONMNIST' in dataset_name:
            shape = (1, 28, 28)
        elif 'CIFAR' in dataset_name:
            shape = (3, 32, 32)
        else:  # Default
            shape = (3, 32, 32)

            # Initialize a random trigger
        self.trigger = torch.rand(shape, requires_grad=False, device=self.client.device) * 0.5
        self.mask = torch.zeros_like(self.trigger, device=self.client.device)
            # Place the trigger in the top-left corner by default
        self.mask[:, :trigger_size, :trigger_size] = 1
    def val_asr(self, model, dataloader, t, m):
        """
        Evaluate the attack success rate (ASR) of the current trigger
        Only count the proportion of non-target label samples that are misled to the target label
        """
        ce_loss = nn.CrossEntropyLoss()
        correct = 0.
        num_non_target = 0.
        total_loss = 0.

        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, original_labels = inputs.to(self.client.device), labels.to(self.client.device)
                
                
                non_target_mask = (original_labels != self.args.y_target)
                if non_target_mask.sum() == 0:
                    continue  
                
               
                inputs = inputs[non_target_mask]
                original_labels = original_labels[non_target_mask]
                
               
                inputs = t * m + (1 - m) * inputs
                
             
                target_labels = torch.full_like(original_labels, self.args.y_target)

                output = model(inputs)
                loss = ce_loss(output, target_labels)
                total_loss += loss

                pred = output.data.max(1)[1]
              
                correct += (pred == self.args.y_target).sum().item()
                num_non_target += inputs.size(0)

        asr = correct / num_non_target if num_non_target > 0 else 0.0
        return asr, total_loss

    def get_adv_model(self, model, dataloader, trigger, mask):
        """
        Generate adversarial model and calculate similarity weight - enhance attack effectiveness and stealthiness
        """
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

     
        dm_adv_epochs = getattr(self.args, 'dm_adv_epochs', 5)
        for _ in range(dm_adv_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.client.device), labels.to(self.client.device)
                inputs = trigger * mask + (1 - mask) * inputs
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = nn.CosineSimilarity(dim=0, eps=1e-08)

        for name in dict(adv_model.named_parameters()):
            if 'conv' in name or 'fc' in name: 
                sim_count += 1
                if dict(adv_model.named_parameters())[name].grad is not None and \
                        dict(model.named_parameters())[name].grad is not None:
                    sim_sum += cos_loss(
                        dict(adv_model.named_parameters())[name].grad.reshape(-1),
                        dict(model.named_parameters())[name].grad.reshape(-1)
                    )

        similarity_weight = sim_sum / max(sim_count, 1)  
        return adv_model, similarity_weight

    def search_trigger(self, model, dataloader):
        """
        Core of A3FL: adaptively optimize the trigger pattern based on the current model.
        Enhanced version includes multi-model ensemble and ASR monitoring.
        """
        print(f"Client {self.client.id}: Starting A3FL trigger optimization...")
        model.eval()

        # Hyperparameter settings
        trigger_lr = getattr(self.args, 'trigger_lr', 0.01)
        trigger_epochs = getattr(self.args, 'trigger_epochs', 200)
        dm_adv_K = getattr(self.args, 'dm_adv_K', 10)  # Frequency of multi-model updates
        dm_adv_model_count = getattr(self.args, 'dm_adv_model_count', 3)  # Number of adversarial models
        noise_loss_lambda = getattr(self.args, 'noise_loss_lambda', 0.01)  

        
        t = self.trigger.clone().detach().to(self.client.device)
        t.requires_grad = True  
        m = self.mask.clone().detach().to(self.client.device)
        trigger_optim = torch.optim.Adam([t], lr=trigger_lr, weight_decay=0)  
        ce_loss = nn.CrossEntropyLoss()

        adv_models = []
        adv_weights = []
        
        for iter in range(trigger_epochs):
           
            if iter % 10 == 0:
                asr, loss = self.val_asr(model, dataloader, t, m)
                print(f"  Iter {iter}: ASR = {asr:.4f}, Loss = {loss:.4f}")

            
            if iter % dm_adv_K == 0 and iter != 0:
                
                for adv_model in adv_models:
                    del adv_model
                adv_models = []
                adv_weights = []

                for _ in range(dm_adv_model_count):
                    adv_model, adv_w = self.get_adv_model(model, dataloader, t, m)
                    adv_models.append(adv_model)
                    adv_weights.append(adv_w)

            for inputs, labels in dataloader:
                trigger_optim.zero_grad()
                inputs, labels = inputs.to(self.client.device), labels.to(self.client.device)

                poisoned_inputs = t * m + (1 - m) * inputs
                target_labels = torch.full_like(labels, self.args.y_target)

                outputs = model(poisoned_inputs)
                loss = ce_loss(outputs, target_labels)

                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_weights[am_idx]
                        adv_outputs = adv_model(poisoned_inputs)
                        adv_loss = ce_loss(adv_outputs, target_labels)
                        loss += noise_loss_lambda * adv_w * adv_loss / dm_adv_model_count

                loss.backward()
                trigger_optim.step()
                
                with torch.no_grad():
                    t.data = torch.clamp(t.data, -1, 1)

        final_asr, final_loss = self.val_asr(model, dataloader, t, m)
        print(f"Client {self.client.id}: Final ASR = {final_asr:.4f}")

        self.trigger = t.detach()
        print(f"Client {self.client.id}: Trigger optimization finished.")

    def poison_batch(self, inputs, labels):
       
        poison_num = int(self.args.poisoned_rate * inputs.shape[0])

        inputs[:poison_num] = self.trigger * self.mask + inputs[:poison_num] * (1 - self.mask)
        labels[:poison_num] = self.args.y_target

        return inputs, labels
