import torch
from torchvision.datasets import ImageFolder
from data.transforms import *
import settings.environment as env
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize
from typing import Tuple, Union, List
from pathlib import Path
import os


def preprocess(img: torch.tensor, 
                mean: Tuple(float)=(0.485, 0.456, 0.406), 
                std: Tuple(float)=(0.485, 0.456, 0.406)
                ) -> np.ndarray:
    """
    Reverse transform an image that was normalized by the dataloader
    to be plotted
    Args:
        img (torch.tensor): data tensor to be visualized
        mean:(Tuple(float)): mean applied in the dataloader Normalize transform
        std:(Tuple(float)): standard deviation applied in the dataloader Normalize transform
    Returns:
        npimg (np.ndarray): data numpy array ready to be imshowed
    """

    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


class CelebaDataset(Dataset):
    """
    Custom Dataset for loading CelebA face images
    """

    def __init__(self, txt_path: Union[str, Path], 
                img_dir: Union[str, Path], 
                transform: transforms=None
                ) -> None:
        """
        Dataset construction
        Args:
            - text_path (Union[str, Path]): path of the csv file containing the image id along with their label
            Example:
                image_id    y
            0   00001.jpg   1
            1   03234.jpg   0
            2   00024.jpg   1
            
            - img_dir (Union[str, Path]): path of the folder containing the celebaimages
            Example:
                ./datasets/celeba/img_align_celeba
            - transform (torchvision.transforms): image transform to be applied to the image upon getting an item
        """
        df = pd.read_csv(txt_path)
        self.img_dir = img_dir
        self.txt_path = txt_path
        self.img_names = df["image_id"].values
        self.y = df["y"].values
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Note that at each __getitem__call, the image is loaded
        from the self.img_dir directory using its path
        """
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index])
                        )
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self) -> int:
        return self.y.shape[0]

def GenerateLabelsFromDNF(dnf: List[List[Tuple[Union(str, int)]]], 
                        csv_path: Union[str, Path]
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates image ids along with associated label
    where label=1 is given by a disjunctive normal form (dnf)

    Args:
        - dnf (List[List[Tuple[Union(str, int)]]]): 
        each disjunctive normal form is parsed as a list of conjunctions,
        where each conjunction is itself a list of predicates, and each predicate
        is a tuple containing the predicate name (e.g. "Blond_Hair") and whether it 
        is negated (-1) or not (+1).

        Example:
            To parse "If it's chubby man with gray hair OR a blond women with eye glasses, then label 1",
            write:
            dnf = [
            [("Gray_Hair", 1), ("Male", 1), ("Chubby", 1)], 
            [("Blond_Hair", 1), ("Male", -1), ("Eyeglasses", 1)]
            ]

        - csv_path (Union[str, Path]): path of the whole Celeb csv file (list_attr_celeba.csv) containing, for each of the image_id, the boolean evaluation
        of each of the 40 attributes (1 is True, -1 otherwise)

        Aspect of list_attr_celeba.csv:
                image_id    Attribute 1     Attribute 2     Attribute 3     ...
                000001.jpg      -1              1               1           ...
                000002.jpg       1             -1               1           ...
                000003.jpg       1              1               1           ...

    Returns:
        labels (np.ndarray): boolean evaluation of the DNF over each sample of the CelebA dataset
                            (DNF == True is encoded as 1, 0 otherwise)
        paths (np.ndarray): associated image_ids of each CelebA sample
    """

    df = pd.read_csv(csv_path)
    # Load all available binary descriptors/predicates
    predicates = df.columns[1:].to_numpy()

    # Build a dictionary that maps a predicates to an index
    predicates_dict = {value:key for key, value in enumerate(predicates)}

    # Convert DataFrame to numpy array
    df_np = df.to_numpy()
    batch =  df_np[:, 1:]
    labels = np.zeros(batch.shape[0], dtype=int)
    paths = df_np[:, 0]

    # Evaluate each conjunction in the DNF over the batch
    for conj in dnf:
        # From the data, select only the attributes (i.e. columns) involved in the current conjunction
        conj_idx = [predicates_dict[c[0]] for c in conj]
        batch_tmp = batch[:, conj_idx]

        # Form a mask of the same size as batch_tmp to flip the Truth value 
        # of the negated predicate in the current conjunction
        mask = np.tile([c[1] for c in conj], (batch.shape[0], 1))

        #Convert -1 (False) + 1 (True) values to 0 (False) 1 (True) values
        batch_tmp = np.maximum(batch_tmp * mask, 0)

        # Evaluate the conjunction by multiplying along dim 1 (along the columns, e.g. predicates)
        labels += np.prod(batch_tmp, axis=1, dtype=int)

    # Evaluate the DNF over the batch
    labels = np.minimum(labels, 1)

    return labels, paths

def CreateDatasetDataFrame(labels: np.ndarray, 
                            paths: np.ndarray, 
                            csv_path: Union[Path, str], 
                            split: int=0
                            ) -> pd.DataFrame:
    """
    Creates a Pandas Dataframe with image_ids along with
    label DNF evaluation, with balanced positive and negative
    examples, for a given split (train, val or test). The split index
    (i.e. whether a sample belongs to train, val or test) is given
     by a .csv file given with the CelebA dataset

    Args:
        - labels (np.ndarray): numpy array with the label/DNF evaluation
        - paths (np.ndarray): numpy array with the image_id associated to each
                            label/DNF evaluation
        -csv_path (Union[Path, str]): path to the list_eval_partition.csv file partition
                                where each image id is paired with a label 0, 1 or 2 if it 
                                belongs to the train, val or test split

    Returns:
        df (pd.DataFrame): Pandas Dataframe containing each image id along with label 0 and 1
                            specified by the DNF given. The first N rows are filled with negative
                            examples (label 0) and the next N ones with positive examples (label 1).
                            The examples are shuffle by the Dataloader object that is subsequently 
                            built from this Dataframe (see GenerateDataLoader)

        Example:
                image_id    label
            0   000032.jpg      0
            1   000054.jpg      0
            ...
            N   000124.jpg      1
            N+1 000434.jpg      1
            ...
    """

    df_split = pd.read_csv(csv_path)
    df_split_np = df_split.to_numpy()
    split_idx = df_split_np[:, 1]
    labels, paths = labels[split_idx == split], paths[split_idx == split]
    labels_pos = labels[labels == 1]
    paths_pos = paths[labels == 1]
    labels_neg = labels[labels == 0]
    paths_neg = paths[labels == 0]

    # Ensure the classes are even
    if labels_pos.shape[0] >= labels_neg.shape[0]:
        rand_idx = np.random.permutation(np.arange(labels_pos.shape[0]))
        rand_idx = rand_idx[:labels_neg.shape[0]]
        labels_pos = labels_pos[rand_idx]
        paths_pos = paths_pos[rand_idx]
    else:
        rand_idx = np.random.permutation(np.arange(labels_neg.shape[0]))
        rand_idx = rand_idx[:labels_pos.shape[0]]
        labels_neg = labels_neg[rand_idx]
        paths_neg = paths_neg[rand_idx]        

    labels = np.concatenate((labels_pos, labels_neg), 0)
    paths = np.concatenate((paths_pos, paths_neg), 0)

    df = np.concatenate((np.expand_dims(paths, 1), np.expand_dims(labels, 1)), axis = 1)
    df = pd.DataFrame(df, columns = ['image_id', 'y'])
    return df

def GenerateSplitFromDNFs(dnf: List[List[Tuple[Union(str, int)]]], 
                        csv_attr_path: Union[Path, str],
                        csv_split_path: Union[Path, str],
                        csv_save_path: Union[Path, str]
                        ) -> None:
    """
    Generate a DataFrame file containing image ids along with labels (specified by a DNF)
    for each split (train, val and test) and save them in the dataset directory

    Args:
        - dnf (List[List[Tuple[Union(str, int)]]]): 
        each disjunctive normal form is parsed as a list of conjunctions,
        where each conjunction is itself a list of predicates, and each predicate
        is a tuple containing the predicate name (e.g. "Blond_Hair") and whether it 
        is negated (-1) or not (+1).

        Example:
            To parse "If it's chubby man with gray hair OR a blond women with eye glasses, then label 1",
            write:
            dnf = [
            [("Gray_Hair", 1), ("Male", 1), ("Chubby", 1)], 
            [("Blond_Hair", 1), ("Male", -1), ("Eyeglasses", 1)]
            ]
        - csv_attr_path (Union[Path, str]): path to the .csv file containing the image_ids with the 40 attributes
        - csv_split_path (Union[Path, str]): path to the list_eval_partition.csv file partition
                                where each image id is paired with a label 0, 1 or 2 if it 
                                belongs to the train, val or test split
        - csv_save_path (Union[Path, str]): path where to save the output .csv files
    Returns:
        None (directly saves the .csv files generated)
    """

    labels, paths = GenerateLabelsFromDNF(dnf, csv_attr_path)
    dict_split = {"train": 0, "val": 1, "test": 2}
    df_dict = {key: CreateDatasetDataFrame(labels, 
                                            paths, csv_split_path, 
                                            split=value)
                for key, value in dict_split.items()
                }
    for key, value in df_dict.items():
        assert len(value.loc[value["y"] == 0]) == len(value.loc[value["y"] == 1])
        value.to_csv(csv_save_path + '/' + key + '.csv')    

def GenerateDataLoader(csv_path: Union[str, Path], 
                        im_path: Union[str, Path], 
                        im_size: int, 
                        batch_size: int
                        )-> DataLoader:
    '''
    Creates a dataloader object from the csv_path containing image ids and associated labels
    and path of the directory containing the images
    Args:
        - csv_path (Union[str, Path]): csv file containing the image ids along with labels
        - im_path (Union[str, Path]): path of the directory containing the CelebA images
        - im_size (int): desired width and height of the images
        - batch_size (int): desired batch size
    Returns:
        dataloader (DataLoader)
    '''
    
    custom_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        # Resize(size=(im_size, im_size)),
                                        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        ])

    dataset = CelebaDataset(txt_path=csv_path,
                                img_dir=im_path,
                                transform=custom_transform)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1)

    return dataloader

if __name__ == "__main__":

    '''
    Initial parsing of a disjunctive normal form: 
    Example: If (no eye glasses, male and chubby) or (blond hair and female) => 1
    '''

    dnf = [
            [("Gray_Hair", 1), ("Male", 1), ("Chubby", 1)], 
            [("Blond_Hair", 1), ("Male", -1), ("Eyeglasses", 1)]
        ]


    GenerateSplitFromDNFs(dnf, 
                        env.DATASETS_FOLDER + '/celeba/list_attr_celeba.csv', 
                        env.DATASETS_FOLDER + '/celeba/list_eval_partition.csv', 
                        env.DATASETS_FOLDER
                        )

    train_loader = GenerateDataLoader(csv_path = env.DATASETS_FOLDER + '/train.csv',
                                        im_path = env.DATASETS_FOLDER + '/celeba/img_align_celeba/',
                                        im_size = 32,
                                        batch_size = 64)


    _, (x, y) = next(enumerate(train_loader))

    x_pos = x[torch.where(y==1)[0], :][:6, :]
    x_neg = x[torch.where(y==0)[0], :][:6, :]

    fig, (axes1, axes2) = plt.subplots(nrows=2, ncols=6, figsize=[16,6])
    for i in range(len(axes1)):
        axes1[i].imshow(preprocess(x_pos[i]))
        axes1[i].set_title("Positive")
        axes2[i].imshow(preprocess(x_neg[i]))
        axes2[i].set_title("Negative")
    plt.show()

    print("Done!")

