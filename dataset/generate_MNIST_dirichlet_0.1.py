

import numpy as np
import os
import sys
import random
import json
from PIL import Image


random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "MNIST-dirichlet-0.1/"


def load_mnist_from_existing(source_dir="MNIST"):
    
    print(f"Loading MNIST from existing dataset: {source_dir}")
    
    train_path = os.path.join(source_dir, "train")
    test_path = os.path.join(source_dir, "test")
    
    all_train_images = []
    all_train_labels = []
    
    train_files = sorted([f for f in os.listdir(train_path) if f.endswith('.npz')])
    print(f"  Found {len(train_files)} training files")
    
    for file in train_files:
        filepath = os.path.join(train_path, file)
        data = np.load(filepath, allow_pickle=True)['data'].item()
        all_train_images.append(data['x'])
        all_train_labels.append(data['y'])
    
    train_images = np.concatenate(all_train_images, axis=0)
    train_labels = np.concatenate(all_train_labels, axis=0)
    
    all_test_images = []
    all_test_labels = []
    
    test_files = sorted([f for f in os.listdir(test_path) if f.endswith('.npz')])
    print(f"  Found {len(test_files)} test files")
    
    for file in test_files:
        filepath = os.path.join(test_path, file)
        data = np.load(filepath, allow_pickle=True)['data'].item()
        all_test_images.append(data['x'])
        all_test_labels.append(data['y'])
    
    test_images = np.concatenate(all_test_images, axis=0)
    test_labels = np.concatenate(all_test_labels, axis=0)
    
    print(f"  ✓ Loaded {len(train_labels)} training samples")
    print(f"  ✓ Loaded {len(test_labels)} test samples")
    
    return train_images, train_labels, test_images, test_labels


def dirichlet_split_noniid(train_labels, alpha, n_clients, num_classes):
    
    n_train = train_labels.shape[0]
    
    min_size = 0
    min_require_size = 10  
    
    client_idcs = [[] for _ in range(n_clients)]
    
    while min_size < min_require_size:
        client_idcs = [[] for _ in range(n_clients)]
        
        for k in range(num_classes):
            
            idx_k = np.where(train_labels == k)[0]
            np.random.shuffle(idx_k)
            
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            
            proportions = np.array([p * (len(idx_j) < n_train / n_clients) 
                                   for p, idx_j in zip(proportions, client_idcs)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            client_idcs = [idx_j + idx.tolist() 
                          for idx_j, idx in zip(client_idcs, np.split(idx_k, proportions))]
            
        min_size = min([len(idx_j) for idx_j in client_idcs])
        print(f"Min samples per client: {min_size}, retrying if < {min_require_size}...")
    
    return client_idcs


def generate_dirichlet_dataset(dir_path, num_clients, alpha=0.5):
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    print(f"Generating MNIST dataset with Dirichlet distribution (alpha={alpha})...")
    print(f"Number of clients: {num_clients}")

    print("\n[Loading] Loading MNIST data from existing dataset...")
    train_images, train_labels, test_images, test_labels = load_mnist_from_existing()
    
    print(f"✓ Train images shape: {train_images.shape}")
    print(f"✓ Test images shape: {test_images.shape}")

    num_classes = len(set(train_labels))
    print(f'✓ Number of classes: {num_classes}')

    print("\n[Dirichlet Split] Splitting training data...")
    client_train_idcs = dirichlet_split_noniid(train_labels, alpha, num_clients, num_classes)
    
    print("\n[IID Split] Splitting test data uniformly...")
    test_idcs = np.random.permutation(len(test_labels))
    client_test_idcs = np.array_split(test_idcs, num_clients)

    statistic = []  
    for client_id in range(num_clients):
        train_idx = client_train_idcs[client_id]
        test_idx = client_test_idcs[client_id]
        
        train_label_dist = {}
        for label in train_labels[train_idx]:
            label = int(label)
            train_label_dist[label] = train_label_dist.get(label, 0) + 1
        
        train_stat = [[label, count] for label, count in sorted(train_label_dist.items())]
        statistic.append(train_stat)
        
        print(f"\nClient {client_id}:")
        print(f"  Train samples: {len(train_idx)}, distribution: {train_label_dist}")

    print("\n[Saving] Saving client data...")
    train_data = {}
    test_data = {}
    
    for client_id in range(num_clients):
        train_idx = client_train_idcs[client_id]
        test_idx = client_test_idcs[client_id]
        
        train_data[client_id] = {
            'x': train_images[train_idx],
            'y': train_labels[train_idx]
        }
        
        test_data[client_id] = {
            'x': test_images[test_idx],
            'y': test_labels[test_idx]
        }
        
        train_file = train_path + str(client_id) + ".npz"
        test_file = test_path + str(client_id) + ".npz"
        
        np.savez_compressed(train_file, data=train_data[client_id])
        np.savez_compressed(test_file, data=test_data[client_id])

    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': True,
        'balance': False,
        'partition': 'dirichlet',
        'Size of samples for labels in clients': statistic,  
        'alpha': alpha,
        'batch_size': 10
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n✅ Dataset generation completed!")
    print(f"   Config saved to: {config_path}")
    print(f"   Train data saved to: {train_path}")
    print(f"   Test data saved to: {test_path}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total train samples: {len(train_labels)}")
    print(f"   Total test samples: {len(test_labels)}")
    print(f"   Dirichlet alpha: {alpha}")
    print(f"   Number of clients: {num_clients}")
    print(f"   Number of classes: {num_classes}")


if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        alpha = float(sys.argv[1])
    else:
        alpha = 0.1  
    
    generate_dirichlet_dataset(dir_path, num_clients, alpha)
