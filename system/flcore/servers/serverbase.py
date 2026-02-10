import torch
import os
import numpy as np
import h5py
import copy
import time
import random

from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
from utils.dlg import DLG
from security.attack.BadNets import CreatePoisonedDataset1
from security.attack.blended import CreatePoisonedDataset2
#from security.attack.A3FL import A3FLAttacker, A3FLDynamicPoisonedDataset



class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        self.poisoned_rate = args.poisoned_rate
        self.y_target = args.y_target
        self.attack_type=args.attack_type
        self.source_label=args.source_label


        if self.attack_type == 'blended':
            self.blending_alpha = args.blending_alpha
       
            if 'Cifar10' in args.dataset or 'Cifar100' in args.dataset:
                self.attack_pattern = torch.randn((3, 32, 32))
            else:  # MNIST / FashionMNIST
                self.attack_pattern = torch.randn((1, 28, 28))
        else:
            self.attack_pattern = None
        self.select_malicious_clients()

        self.test_data = read_client_data(self.dataset, -1, is_train=False)
        
        self.defense_type = args.defense_type
        self.defense_threshold = args.defense_threshold
        self.lsep_layer = args.lsep_layer
        self.model_str = args.model_str

        self.enable_multi_level = getattr(args, 'enable_multi_level', False)
        self.multi_level_positions = getattr(args, 'multi_level_positions', None)
        self.fusion_strategy = getattr(args, 'fusion_strategy', 'weighted_average')

    def set_clients(self, clientObj):
        self.args.malicious_ids = self.malicious_client_ids
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)

            if i in self.malicious_client_ids:

                if self.attack_type == 'badnets':
                    print(f"Applying BadNets attack to client {i}'s training data.")
                    train_data = CreatePoisonedDataset1(
                        dataset_name=self.dataset,  
                        benign_dataset=train_data,
                        y_target=self.y_target,
                        poisoned_rate=self.poisoned_rate,
                        pattern=None,
                        weight=None,
                        poisoned_transform_index=0,  
                        poisoned_target_transform_index=0
                    )
                elif self.attack_type == 'blended':
                        print(f"Applying blened attack to client {i}'s training data.")
                        train_data = CreatePoisonedDataset2(
                            dataset_name=self.dataset,
                            benign_dataset=train_data,
                            y_target=self.y_target,
                            poisoned_rate=self.poisoned_rate,
                            pattern=self.attack_pattern,  
                            alpha=self.blending_alpha  
                        )


            client = clientObj(self.args,
                               id=i,
                               train_data=train_data, 
                               test_data=test_data, 
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)


    def select_malicious_clients(self):
        
        if hasattr(self.args, 'malicious_ids') and self.args.malicious_ids:
            self.malicious_client_ids = self.args.malicious_ids.copy()
            print(f"恶意客户端 (预设): {sorted(self.malicious_client_ids)}")
          
            self.attack_ratio = len(self.malicious_client_ids) / self.num_clients
            return

        self.attack_ratio = self.args.attack_ratio
        num_malicious = int(self.num_clients * self.attack_ratio)

        if self.attack_ratio > 0 and num_malicious == 0:
            num_malicious = 1

        if num_malicious > 0:
            all_client_ids = list(range(self.num_clients))
            self.malicious_client_ids = random.sample(all_client_ids, num_malicious)
            print(f"恶意客户端 (随机选择): {sorted(self.malicious_client_ids)}")
        else:
            self.malicious_client_ids = []

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients



    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.state_dict())
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)


    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
        # flcore/servers/serverbase.py

    def evaluate(self, acc=None, loss=None):

        # extract_images_from_server(self, self.current_round, num_images=10)
        
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        if self.attack_ratio > 0 and self.attack_type != 'none':
            poisoned_test_data = None
            print(f"Evaluating ASR for attack type: {self.attack_type}...")

            if self.attack_type == 'badnets':
                poisoned_test_data = CreatePoisonedDataset1(
                    dataset_name=self.dataset,
                    benign_dataset=self.test_data,
                    y_target=self.y_target,
                    poisoned_rate=1.0,  
                    pattern=None,  
                    weight=None
                )
            elif self.attack_type == 'blended':
                poisoned_test_data = CreatePoisonedDataset2(
                    dataset_name=self.dataset,
                    benign_dataset=self.test_data,
                    y_target=self.y_target,
                    poisoned_rate=1.0,  
                    pattern=self.attack_pattern,  
                    alpha=self.blending_alpha
                )

            elif self.attack_type == 'model_replacement':
       
                if hasattr(self, 'model_replacement_attacker') and self.model_replacement_attacker:
                    asr = self._evaluate_model_replacement_asr()
                    print("Model Replacement Attack Success Rate (ASR): {:.4f}".format(asr))
      

            if poisoned_test_data:
                poisoned_test_loader = DataLoader(poisoned_test_data, batch_size=self.batch_size, drop_last=False)
                self.global_model.eval()
                poison_correct = 0
                total_samples = 0
                with torch.no_grad():
                    for x, y in poisoned_test_loader:
                        x, y = x.to(self.device), y.to(self.device)
                   
                        try:
                            output = self.global_model(x, mode="full")
                        except TypeError:
                            output = self.global_model(x)

                        pred = output.argmax(1)
                        poison_correct += (pred == self.y_target).sum().item()
                        total_samples += y.size(0)

                asr = poison_correct / total_samples if total_samples > 0 else 0
                print("Attack Success Rate (ASR): {:.4f}".format(asr))
            #else:
            #    print(f"Warning: ASR evaluation for attack type '{self.attack_type}' is not implemented.")
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))



    def _evaluate_model_replacement_asr(self):
     
        if not hasattr(self, 'model_replacement_attacker') or not self.model_replacement_attacker:
            return 0.0

        from torch.utils.data import DataLoader
        test_loader = DataLoader(self.test_data, batch_size=64, shuffle=False)
        device = next(self.global_model.parameters()).device

        self.global_model.eval()
        successful_attacks = 0
        total_samples = 0

        print("Model Replacement ASR Details:")
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                if batch_idx >= 10:  
                    break

                x, y = x.to(device), y.to(device)

                x_triggered = x.clone()
                trigger_size = self.model_replacement_attacker.trigger_size

                if len(x.shape) == 4:  # (batch, channel, height, width)
                    if x.shape[1] == 3:  # RGB图像
                        x_triggered[:, :, -trigger_size:, -trigger_size:] = self.model_replacement_attacker.trigger_pattern
                    else:  
                        if len(self.model_replacement_attacker.trigger_pattern.shape) > 2:
                            x_triggered[:, :, -trigger_size:, -trigger_size:] = self.model_replacement_attacker.trigger_pattern.mean(0)
                        else:
                            x_triggered[:, :, -trigger_size:, -trigger_size:] = self.model_replacement_attacker.trigger_pattern

                try:
                    output = self.global_model(x_triggered, mode="full")
                except TypeError:
                    output = self.global_model(x_triggered)

                pred = output.argmax(1)
                target_predictions = (pred == self.model_replacement_attacker.target_label).sum().item()
                successful_attacks += target_predictions
                total_samples += x.size(0)

        asr = successful_attacks / total_samples if total_samples > 0 else 0.0

        print(f"  - Evaluated {total_samples} triggered test samples")
        print(f"  - Samples classified as target label {self.model_replacement_attacker.target_label}: {successful_attacks}")
        print(f"  - Attack Success Rate: {asr:.1%}")

        return asr

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1

            # items.append((client_model, origin_grad, target_inputs))

        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args,
                            id=i,
                            train_samples=len(train_data),
                            test_samples=len(test_data),
                            train_slow=False,
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
