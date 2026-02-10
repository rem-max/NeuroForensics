#!/usr/bin/env python
print("---!!! HELLO, I AM RUNNING THE LATEST VERSION OF MAIN.PY !!!---")
import sys
import os

# Add the parent directory of the current directory (i.e., project root directory) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import time  # ⬅️ 1. Import time module

# Conditional import based on PCA usage
# Will be replaced after args parsing
from flcore.servers.serveravg import FedAvg



from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *
from flcore.trainmodel.vgg import *
from security.utils.partial_models_adaptive import VGGAdaptivePartialModel
from flcore.trainmodel.resnet import resnet18, BasicBlock
from security.utils.partial_models_adaptive import ResNetAdaptivePartialModel
from utils.result_utils import average_data
from utils.mem_utils import MemReporter
import random

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
# torch.manual_seed(0)
seed = int(time.time())
print(f"Using random seed: {seed}")
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model
    args.model_str = model_str
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "MLR":  # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "CNN":  # non-convex
            if 'MNIST' in args.dataset or 'FashionMNIST' in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)


        elif model_str == "LeNet":

            # LeNet is specifically for 28x28 grayscale images like MNIST and FashionMNIST

            if 'MNIST' in args.dataset or 'FashionMNIST' in args.dataset:

                # Correct calling method: only pass the required num_classes parameter

                args.model = LeNet(num_classes=args.num_classes).to(args.device)

            else:

                # Good practice: throw an error if the model and dataset don't match

                raise ValueError("LeNet model is only suitable for MNIST or FashionMNIST datasets.")

        elif model_str == "VGG16":
            # Dynamically determine model input size and channel count based on dataset
            if 'MNIST' in args.dataset or 'FashionMNIST' in args.dataset:
                in_channels = 1
                img_size = 28
            elif 'Cifar10' in args.dataset or 'Cifar100' in args.dataset:
                in_channels = 3
                img_size = 32
            else:
                # Provide a reasonable default value for unknown datasets
                print(
                    f"Warning: VGG16 input shape not explicitly defined for dataset {args.dataset}. Defaulting to 3x32x32.")
                in_channels = 3
                img_size = 32

            # Create VGG16 model using dynamically determined parameters
            args.model = VGG16(
                num_classes=args.num_classes,
                in_channels=in_channels,
                img_size=img_size,
                attack_type=getattr(args, 'attack_type', None),
                defense_type=getattr(args, 'defense_type', None)
            ).to(args.device)
        elif model_str == "ResNet18":
            # 1. Convert dataset name to uppercase for case-insensitive comparison
            dataset_name_upper = args.dataset.upper()

            # 2. Dynamically determine input channel count based on dataset name
            in_channels = 3  # Default to 3 channels
            if 'MNIST' in dataset_name_upper or 'FASHIONMNIST' in dataset_name_upper:
                in_channels = 1
            elif 'CIFAR10' in dataset_name_upper:
                in_channels = 3

            # 3. Create base ResNet18 model with correct channel count and attack/defense types
            args.model = resnet18(
                num_classes=args.num_classes,
                in_channels=in_channels,
                attack_type=getattr(args, 'attack_type', None),
                defense_type=getattr(args, 'defense_type', None)
            ).to(args.device)

        elif model_str == "DNN":  # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)


        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "AlexNet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

        
        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                      num_classes=args.num_classes).to(args.device)

            

        elif model_str == "MobileNet":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

           

        elif model_str == "LSTM":
            args.model = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size,
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "BiLSTM":
            args.model = BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim,
                                                   output_size=args.num_classes, num_layers=1,
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0,
                                                   embedding_length=args.feature_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size,
                                  num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size,
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2,
                                          num_classes=args.num_classes, max_len=args.max_len).to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == 'HAR':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)


        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FD":
            server = FD(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        elif args.algorithm == "FedPAC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPAC(args, i)

        elif args.algorithm == "LG-FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FedGC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGC(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedKD(args, i)

        elif args.algorithm == "FedPCL":
            args.model.fc = nn.Identity()
            server = FedPCL(args, i)

        elif args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCP(args, i)

        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = GPFL(args, i)

        elif args.algorithm == "FedNTD":
            server = FedNTD(args, i)

        elif args.algorithm == "FedGH":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGH(args, i)

        elif args.algorithm == "FedDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDBE(args, i)

        elif args.algorithm == 'FedCAC':
            server = FedCAC(args, i)

        elif args.algorithm == 'PFL-DA':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = PFL_DA(args, i)

        elif args.algorithm == 'FedLC':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedLC(args, i)

        elif args.algorithm == 'FedAS':

            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAS(args, i)

        elif args.algorithm == "FedCross":
            server = FedCross(args, i)

        else:
            raise NotImplementedError
        # Core code
        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100,
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80,
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL / FedCross
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON / FedCAC / FedLC
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    # FedCross
    parser.add_argument('-fsb', "--first_stage_bound", type=int, default=0)
    parser.add_argument('-ca', "--fedcross_alpha", type=float, default=0.99)
    parser.add_argument('-cmss', "--collaberative_model_select_strategy", type=int, default=1)

    # ========== New: Attack-related parameters ==========
    parser.add_argument('-atype', "--attack_type", type=str, default="none",
                        choices=["none", "badnets", "blended", "a3fl", "model_replacement"],
                        help="(none, badnets, labelflip, model_replacement)")
    parser.add_argument('-ar', "--attack_ratio", type=float, default=0.2,
                        help="Ratio of malicious clients")
    parser.add_argument('-pr', "--poisoned_rate", type=float, default=0.2,
                        help="Local data poisoning rate (applicable to BadNets and Labelflip)")
    parser.add_argument('-sl', "--source_label", type=int, default=1,
                        help="Source label for label-flipping attack (e.g., flip all 1s)")
    parser.add_argument('-yt', "--y_target", type=int, default=7,
                        help="Target label for backdoor or label-flipping attack")
    parser.add_argument('-alpha', "--blending_alpha", type=float, default=0.2,
                        help="Blending attack alpha (mixing rate)")

    # --- New parameters specific to A3FL attack ---
    parser.add_argument('--trigger_lr', type=float, default=0.01,
                        help="A3FL: Learning rate for optimizing trigger")
    parser.add_argument('--trigger_epochs', type=int, default=10,
                        help="A3FL: Number of outer loop epochs for optimizing trigger")
    parser.add_argument('--trigger_size', type=int, default=4,
                        help="A3FL: Size of the trigger patch")

    # --- Multi-model ensemble parameters ---
    parser.add_argument('--dm_adv_epochs', type=int, default=5,
                        help="A3FL: Number of epochs to train each adversarial model")
    parser.add_argument('--dm_adv_K', type=int, default=1,
                        help="A3FL: Adversarial model update frequency (update adversarial model set every K rounds)")

    # --- New parameters specific to Model Replacement attack ---
    parser.add_argument('--model_replace_target_label', type=int, default=0,
                        help="Model Replacement: Backdoor target label")
    parser.add_argument('--model_replace_trigger_size', type=int, default=4,
                        help="Model Replacement: Trigger size")
    parser.add_argument('--model_replace_backdoor_epochs', type=int, default=10,
                        help="Model Replacement: Backdoor training epochs")
    parser.add_argument('--model_replace_clean_ratio', type=float, default=0.8,
                        help="Model Replacement: Ratio of clean data in training")
    parser.add_argument('--model_replace_lr', type=float, default=0.01,
                        help="Model Replacement: Backdoor model training learning rate")
    parser.add_argument('--dm_adv_model_count', type=int, default=1,
                        help="A3FL: Number of concurrent adversarial models to maintain")
    parser.add_argument('--noise_loss_lambda', type=float, default=0.01,
                        help="A3FL: Weight coefficient for multi-model ensemble loss")
    # ========== Defense-related parameters ==========
    parser.add_argument('-dft', "--defense_type", type=str, default="none",
                        choices=["none", "neuroforensics", "gradient_clipping", "robust_aggregation", "flame","alignins"],
                        help="Choose the type of defense to use")
    parser.add_argument('-dth', "--defense_threshold", type=float, default=2.0,
                        help="NeuroForensics defense M' value threshold; below this is considered benign")
    parser.add_argument('-lsep', "--lsep_layer", type=int, default=2,
                        help="Position of the NeuroForensics model split layer (your code enforces 2)")
    # ========== AlignIns defense parameters ==========
    parser.add_argument('--alignins_lambda_s', type=float, default=1.0,
                        help="AlignIns: MPSA MZ-score threshold (default: 1.0)")
    parser.add_argument('--alignins_lambda_c', type=float, default=1.0,
                        help="AlignIns: TDA MZ-score threshold (default: 1.0)")
    parser.add_argument('--alignins_sparsity', type=float, default=0.3,
                        help="AlignIns: Top-k sparsity considered in MPSA calculation (default: 0.3)")
    # NeuroForensics multi-level detection parameters
    parser.add_argument('--enable_multi_level', action='store_true',
                        help="Enable NeuroForensics multi-level detection (detect backdoors at multiple levels simultaneously)")
    parser.add_argument('--multi_level_positions', type=str, default=None,
                        help="Positions of layers for multi-level detection, separated by commas (e.g., '0,2,4'); defaults to automatic selection of 3 layers")
    parser.add_argument('--fusion_strategy', type=str, default='weighted_average',
                        choices=['max', 'average', 'weighted_average', 'voting'],
                        help="Fusion strategy for multi-level detection results (default: weighted_average adaptive weighted average)")
    
    # NeuroForensics similarity detection parameters (ablation study)
    parser.add_argument('--neuroforensics_use_similarity', action='store_true',
                        help="Enable Mat_p similarity detection (ablation study, replaces multi-metric fusion M' method)")
    parser.add_argument('--neuroforensics_similarity_metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean', 'frobenius', 'kl_divergence'],
                        help="Mat_p similarity metric (default: cosine)")
    
    # NeuroForensics PCA fusion parameters
    parser.add_argument('--neuroforensics_use_pca', action='store_true',
                        help="Enable PCA-based multi-metric fusion (fuse 4 metrics: S_entropy, S_max, S_kurt, S_boxplot)")
    
    # Gradient clipping defense parameters
    parser.add_argument('--grad_clip_max_norm', type=float, default=1.0,
                        help="Maximum norm for gradient clipping (default: 1.0)")
    parser.add_argument('--grad_clip_norm_type', type=float, default=2.0,
                        help="Norm type for gradient clipping (default: 2.0, indicating L2 norm)")
    parser.add_argument('--grad_clip_detect_threshold', type=float, default=2.0,
                        help="Gradient anomaly detection threshold multiplier (default: 2.0, i.e., 2 times max_norm)")
    
    # Robust aggregation defense parameters
    parser.add_argument('--robust_agg_method', type=str, default='median',
                        choices=['median', 'trimmed_mean', 'mean', 'krum', 'multi_krum'],
                        help="Robust aggregation method (default: median)")
    parser.add_argument('--robust_trim_ratio', type=float, default=0.1,
                        help="Trimmed mean trimming ratio (default: 0.1, removing 10%% of extreme values)")
    parser.add_argument('--robust_krum_f', type=int, default=4,
                        help="Maximum number of malicious clients tolerated by Krum (default: 4)")
    parser.add_argument('--robust_krum_k', type=int, default=5,
                        help="Number of clients selected by Multi-Krum (default: 5)")
    parser.add_argument('--robust_detect_threshold', type=float, default=2.0,
                        help="Anomaly detection threshold multiplier for robust aggregation (default: 2.0)")
    
    # FLAME defense parameters
    parser.add_argument('--flame_min_cluster_ratio', type=float, default=0.5,
                        help="Minimum cluster size ratio for FLAME (default: 0.5, requiring benign clusters to occupy more than half)")
    parser.add_argument('--flame_noise_lambda', type=float, default=0.000012,
                        help="FLAME differential privacy noise coefficient (default: 0.000012)")

    # ========== Adaptive threshold related parameters - Historical M' value 3σ anomaly detection strategy ==========
    parser.add_argument('--adaptive_threshold', action='store_true', default=False,
                        help="Enable adaptive threshold adjustment")
    parser.add_argument('--adaptive_strategy', type=str, default='historical_3sigma',
                        choices=['historical_3sigma'],
                        help="Adaptive threshold strategy (currently supported: historical_3sigma - Historical M' value 3σ anomaly detection)")
    parser.add_argument('--adaptive_history_window', type=int, default=10,
                        help="Number of historical M' values stored per client")
    parser.add_argument('--adaptive_sigma_threshold', type=float, default=1.0,
                        help="Variance anomaly detection threshold (default: 1.0, update threshold if exceeded)")
    parser.add_argument('--adaptive_top_k', type=int, default=3,
                        help="Number of top K M' values used to calculate new threshold")

    args = parser.parse_args()
    
    # Conditional import: use serveravg_pca when PCA fusion is enabled
    if getattr(args, 'neuroforensics_use_pca', False) and args.defense_type == 'neuroforensics':
        print("\n" + "="*60)
        print("[PCA Mode] Using serveravg_pca.FedAvg for PCA-based fusion")
        print("="*60 + "\n")
        from flcore.servers.serveravg_pca import FedAvg
        # Re-assign to global scope for run function
        globals()['FedAvg'] = FedAvg

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
    print("=" * 50)

    
    run(args)

    