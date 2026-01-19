import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from exercise_code.data.image_folder_dataset import MemoryImageFolderDataset

class MyPytorchModel(nn.Module):
    
    def __init__(self, hparams):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.model = None
        self.device = hparams.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        ########################################################################
        # TODO: Initialize your model!                                         #
        # The model should be saved to a variable called self.model            #
        ########################################################################
        
        # Extract hyperparameters with defaults
        input_size = hparams.get("input_size", 3 * 32 * 32)  # CIFAR-10: 3 channels, 32x32 images
        n_hidden = hparams.get("n_hidden", 512)  # Hidden layer size
        num_classes = hparams.get("num_classes", 10)  # CIFAR-10 has 10 classes
        dropout_rate = hparams.get("dropout_rate", 0.3)  # Dropout for regularization
        
        # Build a multi-layer fully connected network with:
        # - Batch normalization for stable training
        # - ReLU activation for non-linearity
        # - Dropout for regularization to prevent overfitting
        # - Proper weight initialization (handled by PyTorch defaults, but we can add custom init)
        
        self.model = nn.Sequential(
            # First hidden layer with batch normalization
            nn.Linear(input_size, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second hidden layer
            nn.Linear(n_hidden, n_hidden // 2),
            nn.BatchNorm1d(n_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Third hidden layer
            nn.Linear(n_hidden // 2, n_hidden // 4),
            nn.BatchNorm1d(n_hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),  # Less dropout in later layers
            
            # Output layer (no activation, will be used with CrossEntropyLoss)
            nn.Linear(n_hidden // 4, num_classes)
        )
        
        # PyTorch automatically initializes weights with reasonable defaults
        # For Linear layers, it uses uniform initialization based on fan_in and fan_out

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):

        # x.shape = [batch_size, 3, 32, 32] -> flatten the image first
        x = x.view(x.shape[0], -1)

        # feed x into model!
        x = self.model(x)

        return x
    
    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        n_total = len(targets)
        return loss, n_correct, n_total
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        length = sum([x[mode + '_n_total'] for x in outputs])
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / length
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "train")
        self.log('loss',loss)
        return {'loss': loss, 'train_n_correct':n_correct, 'train_n_total': n_total}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "val")
        self.log('val_loss',loss)
        return {'val_loss': loss, 'val_n_correct':n_correct, 'val_n_total': n_total}
    
    def test_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct':n_correct, 'test_n_total': n_total}

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        self.log('val_loss',avg_loss)
        self.log('val_acc',acc)
        return {'val_loss': avg_loss, 'val_acc': acc}

    def configure_optimizers(self):

        optim = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################
        
        # Note: This method is typically used with PyTorch Lightning
        # In the notebook, optimizer is created separately in train_model function
        # But we'll define it here for completeness
        learning_rate = self.hparams.get("learning_rate", 0.001)
        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return optim

    def getTestAcc(self, loader):
        self.model.eval()
        self.model = self.model.to(self.device)

        scores = []
        labels = []

        for batch in tqdm(loader):
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc


class CIFAR10DataModule(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.opt = hparams
        if 'loading_method' not in hparams.keys():
            self.opt['loading_method'] = 'Image'
        if 'num_workers' not in hparams.keys():
            self.opt['num_workers'] = 2

    def prepare_data(self, stage=None, CIFAR_ROOT="../datasets/cifar10"):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # create dataset
        CIFAR_ROOT = "../datasets/cifar10"
        my_transform = None
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        ########################################################################
        # TODO: Define your transforms (convert to tensors, normalize).        #
        # If you want, you can also perform data augmentation!                 #
        ########################################################################
        
        # Training transform with data augmentation to improve generalization
        # Data augmentation helps the model learn robust features by:
        # 1. Random horizontal flips - simulates different viewing angles
        # 2. Random crops with padding - handles translation and scale variations
        # 3. Color jitter - makes model robust to lighting/color changes
        # 4. Normalization - standardizes input distribution for stable training
        
        my_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
            transforms.RandomCrop(32, padding=4),  # Random crop with 4-pixel padding
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color variations
            transforms.ToTensor(),  # Convert PIL image to tensor [0, 1]
            transforms.Normalize(mean=mean, std=std)  # Normalize to ImageNet statistics
        ])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        
        # Make sure to use a consistent transform for validation/test
        train_val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        # Note: you can change the splits if you want :)
        split = {
            'train': 0.6,
            'val': 0.2,
            'test': 0.2
        }
        split_values = [v for k,v in split.items()]
        assert sum(split_values) == 1.0
        
        if self.opt['loading_method'] == 'Image':
            # Set up a full dataset with the two respective transforms
            cifar_complete_augmented = torchvision.datasets.ImageFolder(root=CIFAR_ROOT, transform=my_transform)
            cifar_complete_train_val = torchvision.datasets.ImageFolder(root=CIFAR_ROOT, transform=train_val_transform)

            # Instead of splitting the dataset in the beginning you can also # split using a sampler. This is not better, but we wanted to 
            # show it off here as an example by using the default
            # ImageFolder dataset :)

            # First regular splitting which we did for you before
            N = len(cifar_complete_augmented)        
            num_train, num_val = int(N*split['train']), int(N*split['val'])
            indices = np.random.permutation(N)
            train_idx, val_idx, test_idx = indices[:num_train], indices[num_train:num_train+num_val], indices[num_train+num_val:]

            # Now we can set the sampler via the respective subsets
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            test_sampler= SubsetRandomSampler(test_idx)
            self.sampler = {"train": train_sampler, "val": val_sampler, "test": test_sampler}

            # assign to use in dataloaders
            self.dataset = {}
            self.dataset["train"], self.dataset["val"], self.dataset["test"] = cifar_complete_augmented,\
                cifar_complete_train_val, cifar_complete_train_val

        elif self.opt['loading_method'] == 'Memory':
            self.dataset = {}
            self.sampler = {}

            for mode in ['train', 'val', 'test']:
                # Set transforms
                if mode == 'train':
                    transform = my_transform
                else:
                    transform = train_val_transform

                self.dataset[mode] = MemoryImageFolderDataset(
                    root = CIFAR_ROOT,
                    transform = transform,
                    mode = mode,
                    split = split
                )
        else:
            raise NotImplementedError("Wrong loading method")

    def return_dataloader_dict(self, mode):
        arg_dict = {
            'batch_size': self.opt["batch_size"],
            'num_workers': self.opt['num_workers'],
            'persistent_workers': True,
            'pin_memory': True
        }
        if self.opt['loading_method'] == 'Image':
            arg_dict['sampler'] = self.sampler[mode]
        elif self.opt['loading_method'] == 'Memory':
            arg_dict['shuffle'] = True if mode == 'train' else False
        return arg_dict

    def train_dataloader(self):
        arg_dict = self.return_dataloader_dict('train')
        return DataLoader(self.dataset["train"], **arg_dict)

    def val_dataloader(self):
        arg_dict = self.return_dataloader_dict('val')
        return DataLoader(self.dataset["val"], **arg_dict)
    
    def test_dataloader(self):
        arg_dict = self.return_dataloader_dict('train')
        return DataLoader(self.dataset["train"], **arg_dict)
