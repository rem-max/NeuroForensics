import time
import numpy as np
from flcore.clients.clientbase import Client
import torch
from security.attack.A3FL import A3FLAttacker
from security.attack.model_replacement import ModelReplacementAttack

class clientAVG(Client):
    def __init__(self, args, id, train_data, test_data, **kwargs):
        super().__init__(args, id, train_data, test_data, **kwargs)
        self.is_malicious = False
        self.attacker_module = None
        self.model_replacement_attacker = None

        if hasattr(self.args,'malicious_ids') and self.id in self.args.malicious_ids:
            if self.args.attack_type == 'a3fl':
                self.is_malicious = True
                self.attacker_module = A3FLAttacker(self)
                print(f"Client {self.id}: Initialized as A3FL Attacker.")
            elif self.args.attack_type == 'model_replacement':
                self.is_malicious = True
                self.model_replacement_attacker = ModelReplacementAttack(self.args)
                print(f"Client {self.id}: Initialized as Model Replacement Attacker.")

    def train(self):
        trainloader = self.load_train_data()

        if self.is_malicious and self.model_replacement_attacker:
            print(f"Client {self.id}: Performing model replacement attack")
            self.model = self.model_replacement_attacker.create_backdoored_model(
                self.model, trainloader, self.device
            )
            return  
        
        if self.is_malicious and self.attacker_module:
            self.attacker_module.search_trigger(self.model, trainloader)

        self.model.train()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.is_malicious and self.attacker_module:
                    x, y = self.attacker_module.poison_batch(x, y)


                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time