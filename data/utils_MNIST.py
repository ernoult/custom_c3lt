import torch
from data.transforms import *
import settings.environment as env
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize
from typing import Tuple, Union, List, Optional
from pathlib import Path
import os
from itertools import compress
from tqdm import tqdm
import time

def imshow(img: torch.tensor)-> None:
    """
    Displays a sample from its tensor representation.
    Args:
        img (torch.tensor): a single sample, as given by a torch dataloader 
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

class DatasetProcessing(torch.utils.data.Dataset):
    """
    Class to generate a customized dataset, 
    in the good format to build a Dataloader object afterwards
    """
    #used to initialise the class variables - transform, data, target
    def __init__(self, data: torch.tensor, target: torch.tensor, transform=None
                )-> None: 
        
        self.transform = transform
        self.data = data
        self.target = target

    #used to retrieve the X and y index value and return it
    def __getitem__(self, index: int): 
        
        if self.transform is not None:
            return self.transform(self.data[index]), self.target[index]
        else:
            return self.data[index], self.target[index]

    #returns the length of the data
    def __len__(self): 
        return len(list(self.data))

@torch.no_grad()
def generateData(path: Union[str, Path],
                grid_size: int = 2,
                resize: int = None,
                keep_channel_dim: bool = False,
                is_train: bool = True,
                xor_color: bool = True,
                size_dset: int= 60000
                ):
    """
    Generates a dataset of paired 0 or 1 digits in a (grid_size * grid_size) grid, where the paired digits
    are labelled with the XOR operation, either on the digits themselves, or also on their color. 
    Args:
        - path (Union[str, Path]): path to the original MNIST dataset
        - grid_size (int): size of the grid where the paired digits are displayed
        - resize (int): specifies the dimension of the MNIST samples. Default=None means that the MNIST digits
                        are not re-scaled (28x28)
        - keep_channel_dim (bool): specifies if we want the MNIST digits in the form 1 x H x W or C x H x W
                                    (this choice may intervene for instance when using paired gray-scale digits,
                                    but willing to keep C=3 channels)
        - is_train (bool): specifies if we generate training data.
        - xor_color (bool): specifies if we color the paired digits and label them with the XOR operation on the colors
                            More precisely, if xor_color = False, samples are labelled with XOR(digit_1, digit_2).
                            If xor_color = False, samples are labelled with XOR(digit_1, digit_2) OR XOR(color_1, color_2)
                            
    Returns:
        X (torch.tensor): tensor of paired digits
        Y (torch.tensor): tensor of labels
    """

    # Define transformations to be applied on MNIST, with resize if specified
    transforms = [torchvision.transforms.ToTensor()]
    if resize is not None:
        transforms += [torchvision.transforms.Resize(resize)]
    transforms = torchvision.transforms.Compose(transforms)

    # Load the MNIST train set wit the pre-defined transformations
    mnist_dset = torchvision.datasets.MNIST(root=path, train=is_train,
                                            download=True, transform=transforms)

    # Sub-select the MNIST samples labelled either 0 or 1                                     
    mask = [((x[1] == 0)  or (x[1] == 1)) for x in mnist_dset]
    dset = list(compress(mnist_dset, mask))

    # Build the train loader
    dataloader = torch.utils.data.DataLoader(dset, batch_size=grid_size ** 2,
                                            shuffle=is_train, num_workers=0)    
    X = []
    Y = []
    for _ in tqdm(range(len(mnist_dset))):
    # for _ in tqdm(len(size_dset)):
    # pbar = tqdm(total=size_dset // 2)
    # sofar = 0
    # while sofar < size_dset // 2:
        # time.sleep(0.1)
        _, (x, y) = next(enumerate(dataloader))

        if x.size(0) < grid_size ** 2: 
            continue

        # Among the (grid ** 2) MNIST samples, discard (grid ** 2) - 2 of them 
        perm = torch.randperm(grid_size ** 2)
        discarded_inds = perm[:grid_size ** 2 - 2]
        x[discarded_inds] = torch.zeros_like(x[discarded_inds])

        # Store the label of the remaining 2 samples
        retained_inds = perm[grid_size ** 2 - 2:]
        y = y[retained_inds]

        # Generate the grid of samples over the current batch
        if keep_channel_dim:
            # Add two extra channels
            x = x.repeat(1, 3, 1, 1)
            if xor_color:
                # Select the retained digits
                x_tmp = x[retained_inds]

                # Change color by zeroing either the first (c=0) or third (c=1) channel randomly
                c = 2 * (torch.rand((retained_inds.size(0))) < 0.5).long()
                for i in range(retained_inds.size(0)):
                    x_tmp[i, c[i], :].mul_(0)
                x[retained_inds] = x_tmp

                # Label the sample with XOR(digit_1, digit_2) OR XOR(color_1, color_2)
                Y += [torch.min((y[0] - y[1]).abs() + (c[0] - c[1]).abs(), torch.tensor(1))]
            else:
                # Label the sample with XOR(digit_1, digit_2)
                Y += [(y[0] - y[1]).abs().unsqueeze(0)]
            
            # Generate the grid and augment it with the batch dimension for later concatenation
            grid = torchvision.utils.make_grid(x, nrow=grid_size)
            X += [grid.view(1, grid.size(0), grid.size(1), grid.size(2))]
        else:
            grid = torchvision.utils.make_grid(x, nrow=grid_size)[0]
            X += [grid.view(1, 1, grid.size(0), grid.size(1))]
            # Label the sample with XOR(digit_1, digit_2)
            Y += [(y[0] - y[1]).abs().unsqueeze(0)]
        

    # Stack all the resulting Xs and Ys
    X = torch.vstack(X)
    Y = torch.vstack(Y)

    # Balance the classes
    n = torch.minimum((Y == 1).sum(), (Y == 0).sum()).item()
    X = torch.cat([X[torch.where(Y == 0)[0], :][:n],
                   X[torch.where(Y == 1)[0], :][:n]], 0
                )
    Y = torch.cat([Y[torch.where(Y == 0)[0], :][:n],
                   Y[torch.where(Y == 1)[0], :][:n]], 0
                )

    return X, Y

def createDataLoader(path: Union[str, Path],
                    transforms: transforms, 
                    batch_size: int, 
                    is_train: bool=True,
                    rewrite: bool=False,
                    split: float=None,
                    label: Optional[int]=None,
                    **kwargs
                    )-> Union[DataLoader, Tuple[DataLoader,DataLoader]]:
    """
    Creates a dataloader (either train, val or test) with an arbitrary split of train/val
    from X, Y data stored in a .pt file (train.pt, val.pt or test.pt). If the .pt file does
    not exist or rewrite is set to True, then it is generated.
    Args:
        - path (Union[str, Path]): local path to the MNIST dataset
        - transforms (transforms): desired transformations to be applied.
                                    WARNING: do not apply the ToTensor() transform here! We do not build
                                    the dataloader from the original PIL images but from tensors already.
        - batch_size(int) : size of the batch
        - is_train (bool): specifies if we want to generate the train dataloader or train/val dataloaders
        - rewrite (bool): specifies whether we want to rewrite the .pt the dataloader is built from.
        - split (float): percentage of the training set actually used for training, with the remaining for 
                        validation.
                        Example: split = 0.9 will retain 90% of the training set for training and 10% for validation
    Returns:
        dataloader (Union[DataLoader, Tuple[DataLoader,DataLoader]]): returns either a single dataloader (for instance
                    if is_train = False, or is_train = True and split left to None), or two dataloaders (if is_train = True
                    and split is specified to build a validation set)
    """

    # Write the path to the data used to build the dataloader
    filename = "train.pt" if is_train else "test.pt"
    path_to_file = os.path.join(path, filename)

    # If the data does not exist or rewrite is imposed, generate the data
    if not os.path.isfile(path_to_file) or rewrite:
        X, Y = generateData(path, is_train=is_train, **kwargs)
        torch.save({"X": X, "Y": Y}, path_to_file)
    else:
        res = torch.load(path_to_file)
        X, Y = res["X"], res["Y"]

    # Subsample only the samples corresponding to one label
    if label is not None:
        X = X[torch.where(Y == label)[0], :]
        Y = Y[torch.where(Y == label)[0], :]

    # If there is no split, use the whole data to create one dataloader
    if split is None:
        dset = DatasetProcessing(X, Y, transform=transforms)
        dataloader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                                shuffle=is_train, num_workers=0)
        return dataloader

    # Otherwise, select randomly (100 * split) % of the data to build a validation set
    else:
        perm = torch.randperm(X.size(0))
        X = X[perm, :]
        Y = Y[perm, :]
        ceil = int(np.floor( (X.size(0)) * split ) )
        traindset = DatasetProcessing(X[:ceil], Y[:ceil], transform=transforms)
        traindataloader = torch.utils.data.DataLoader(traindset, batch_size=batch_size,
                                                shuffle=is_train, num_workers=0)

        valdset = DatasetProcessing(X[ceil:], Y[ceil:], transform=transforms)
        valdataloader = torch.utils.data.DataLoader(valdset, batch_size=batch_size,
                                                shuffle=False, num_workers=0) 

        return traindataloader, valdataloader

if __name__ == "__main__":
    '''
    See below an example
    To import createDataLoader from any hierarchy of the project directory,
    do from data.utils_MNIST import createDataLoader
    '''

    transforms = torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))

    # Generate train and val dataloaders
    # trainloader, valloader = createDataLoader(env.DATASETS_FOLDER, 
    #                                 transforms, 
    #                                 8, 
    #                                 is_train=True, 
    #                                 resize=11,
    #                                 split = 0.9,
    #                                 keep_channel_dim=True,
    #                                 rewrite=False,
    #                                 size_dset=70000
    #                                 )

    # Generate test loader
    # testloader = createDataLoader(env.DATASETS_FOLDER, 
    #                                 transforms, 
    #                                 8, 
    #                                 is_train=False, 
    #                                 resize=11,
    #                                 keep_channel_dim=True,
    #                                 rewrite=False,
    #                                 size_dset=70000
    #                                 )


    pos_trainloader = createDataLoader(env.DATASETS_FOLDER, 
                                    transforms, 
                                    8, 
                                    is_train=True, 
                                    resize=11,
                                    keep_channel_dim=True,
                                    rewrite=False,
                                    size_dset=60000,
                                    label=1
                                    )

    neg_trainloader = createDataLoader(env.DATASETS_FOLDER, 
                                    transforms, 
                                    8, 
                                    is_train=True, 
                                    resize=11,
                                    keep_channel_dim=True,
                                    rewrite=False,
                                    size_dset=10000,
                                    label=0
                                    )

    # --------------------------------------------------------------------------- #
    # IMPORTANT: if you want to iterate training experiments on the *same* dataset,
    # set rewrite to False! Then the dataloader will be simply built from already
    # prepared data in the related .pt files. 
    # --------------------------------------------------------------------------- #

    # Sanity check: visualize a batch of samples with associated labels
    # _, (sample, label) = next(enumerate(neg_trainloader))

    # fig = plt.figure(figsize=[16,6])
    # for i in range(sample.size(0)):
    #     plt.subplot(2, sample.size(0) // 2, i + 1)
    #     imshow(sample[i])
    #     plt.title("y = {}".format(label[i].item()))
    
    # fig.tight_layout()
    # plt.show()
    
    # Checking we can loop simultaneously on the two trainloaders
    data_loader = (pos_trainloader, neg_trainloader)
    loop = tqdm(zip(*data_loader))
    for idx, ((images_pos, _), (images_neg, _)) in enumerate(loop):
        print(images_pos.size())
        print(images_neg.size())
    print("Done")
    

