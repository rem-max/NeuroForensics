import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
import sys
import os
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize

current_file = os.path.abspath(__file__)
system_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if system_dir not in sys.path:
    sys.path.append(system_dir)


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_data, test_data, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name
        self.args = args
        self.num_classes = args.num_classes
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        # self.train_samples = train_samples
        # self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        self.train_data = train_data
        self.test_data = test_data

        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def load_train_data(self, batch_size=None):

        if batch_size == None:
            batch_size = self.batch_size
        return DataLoader(self.train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        return DataLoader(self.test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        all_probs = []
        all_true_labels = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                try:
                    output = self.model(x, mode="full")
                except TypeError:
                    output = self.model(x)
                    if output.shape[1] != self.num_classes:
                        if not hasattr(self, "cls_head"):
                            in_dim = output.shape[1]
                            self.cls_head = torch.nn.Linear(in_dim, self.num_classes).to(self.device)
                        output = self.cls_head(output)

                test_acc += (torch.argmax(output, dim=1) == y).sum().item()
                test_num += y.size(0)

                probs = F.softmax(output, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_true_labels.append(y.cpu().numpy())

        y_prob = np.concatenate(all_probs, axis=0)
        y_true_integers = np.concatenate(all_true_labels, axis=0)

        if np.isnan(y_prob).any():
            print(f"[Warning] Client {self.id}: Model output contains NaN values. Setting AUC to 0.")
            auc = 0.0
            return test_acc, test_num, auc

        try:
            unique_labels = np.unique(y_true_integers)
            
            if len(unique_labels) > 1:
                if self.num_classes > 2:
                    if len(unique_labels) == self.num_classes:

                        auc = metrics.roc_auc_score(
                            y_true_integers, y_prob, multi_class='ovr', average='micro'
                        )
                    else:
                        y_true_onehot = np.zeros((len(y_true_integers), self.num_classes))
                        y_true_onehot[np.arange(len(y_true_integers)), y_true_integers] = 1
                        
                        try:
                            auc = metrics.roc_auc_score(
                                y_true_onehot, y_prob, multi_class='ovr', average='micro'
                            )
                        except ValueError:
                           
                            auc = 0.0
                            print(f"[Warning] Client {self.id}: Cannot calculate AUC (only {len(unique_labels)} classes present)")
                else:
                
                    auc = metrics.roc_auc_score(
                        y_true_integers, y_prob[:, 1], average='micro'
                    )
            else:
           
                auc = 0.0
                print(f"[Warning] Client {self.id}: Only one class in test data, AUC set to 0.0")
        except ValueError as e:
        
            print(f"[Warning] Client {self.id}: AUC calculation failed, set to 0.0. Error: {e}")
            auc = 0.0

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
