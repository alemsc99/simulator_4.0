import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST, CIFAR10




def get_mnist(data_path:str='./data'):

    tr= Compose([ToTensor(), Normalize((0.1307), (0.3081,))])
    trainset=MNIST(data_path, train=True, download=True, transform=tr)
    testset=MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset

def get_cifar10(data_path:str='./data'):

    tr= Compose([ToTensor(), Normalize((0.1307), (0.3081,))])
    trainset=CIFAR10(data_path, train=True, download=True, transform=tr)
    testset=CIFAR10(data_path, train=False, download=True, transform=tr)

    return trainset, testset


def prepare_dataset(dataset_name, batch_size:int, val_ratio: float=0.1):
    # num_partions= number of clients= number of partitions to create starting from the training set
    # batch_size= assuming it's the same for every client
    # val_ratio= ratio of validation samples wrt training samples
    
    if dataset_name == 'mnist':
        trainset, testset=get_mnist()
    elif dataset_name == 'cifar10':
        trainset, testset=get_cifar10()
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")
    
  
    #Now, for each trainingset, we want to partition it in training and validation 
    #and create DataLoaders for each of them
        
    num_total=len(trainset)
    num_val=int(val_ratio*num_total)
    num_train=num_total-num_val
    
    for_train, for_val=random_split(trainset, [num_train, num_val], torch.Generator().manual_seed(2024))
    #Now, we have a training set and a validation set, for every partition
    trainloaders=(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
    valloaders=(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))


    # We would have one dataloader per client for both training and validation set 
    # assgined to this particular client
    
    # We have to create the dataloader for the test set. We use an higher batch size since we
    # want a faster evaluation phase
    testloader=DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader


    
