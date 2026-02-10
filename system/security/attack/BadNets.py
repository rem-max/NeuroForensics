import copy
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from torchvision import transforms



class AddTrigger:
  

    def __init__(self):
        pass

    def add_trigger(self, img):
        return (self.weight * img + self.res).type(torch.uint8)


class AddMNISTTrigger(AddTrigger):
  

    def __init__(self, pattern, weight):
        super(AddMNISTTrigger, self).__init__()

        if pattern is None:
            self.pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
            self.pattern[0, -2, -2] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            self.weight = torch.zeros((1, 28, 28), dtype=torch.float32)
            self.weight[0, -2, -2] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = img.squeeze()
        img = Image.fromarray(img.numpy(), mode='L')
        return img


class AddCIFAR10Trigger(AddTrigger):
   

    def __init__(self, pattern, weight):
        super(AddCIFAR10Trigger, self).__init__()

        if pattern is None:
          
            self.pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
            self.pattern[0, -3:, -3:] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if self.pattern.shape[0] == 1:
            self.pattern = self.pattern.repeat(3, 1, 1)

        if weight is None:
            self.weight = torch.zeros((1, 32, 32), dtype=torch.float32)
            self.weight[0, -3:, -3:] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        if self.weight.shape[0] == 1:
            self.weight = self.weight.repeat(3, 1, 1)

        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


class ModifyTarget:

    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
     
        return torch.tensor(self.y_target).long()



class PoisonedDatasetWrapper(Dataset):
   

    def __init__(self, benign_dataset, y_target, poisoned_rate, trigger_transform):
        self.benign_dataset = benign_dataset
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should be greater than or equal to zero.'

        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        self.trigger_transform = trigger_transform
        self.target_transform = ModifyTarget(y_target)

        self.to_tensor = transforms.ToTensor()
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def __getitem__(self, index):
        img, target = self.benign_dataset[index]

        if isinstance(img, torch.Tensor):
            if img.shape[0] == 3:
                img = Image.fromarray(img.permute(1, 2, 0).numpy().astype(np.uint8))
            elif img.shape[0] == 1:
                img = Image.fromarray(img.squeeze(0).numpy().astype(np.uint8), mode='L')
            else:
                img = Image.fromarray(img.numpy().astype(np.uint8))

        if index in self.poisoned_set:
            img = self.trigger_transform(img)
            target = self.target_transform(target)
        else:

            pass


        img = self.to_tensor(img)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        return img, target

    def __len__(self):
        return len(self.benign_dataset)


def CreatePoisonedDataset1(dataset_name, benign_dataset, y_target, poisoned_rate, pattern, weight, **kwargs):

    if 'MNIST' in dataset_name.upper():
        trigger_transform = Compose([AddMNISTTrigger(pattern, weight)])
    elif 'CIFAR10' in dataset_name.upper():
        trigger_transform = Compose([AddCIFAR10Trigger(pattern, weight)])
    else:
 
        raise NotImplementedError(f"Trigger for dataset '{dataset_name}' is not implemented.")

    return PoisonedDatasetWrapper(
        benign_dataset=benign_dataset,
        y_target=y_target,
        poisoned_rate=poisoned_rate,
        trigger_transform=trigger_transform
    )

