import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageNet
import torchvision.transforms as transforms




def get_mnist(data_path:str='./data'):

    tr= Compose([ToTensor(), Normalize((0.1307), (0.3081,))])
    trainset=MNIST(data_path, train=True, download=True, transform=tr)
    testset=MNIST(data_path, train=False, download=True, transform=tr)
    
    num_classes= len(trainset.classes)
    input_channels= 1

    input_size_x=trainset.data.shape[1]
    input_size_y= trainset.data.shape[2]

    return trainset, testset, num_classes, input_channels, input_size_x, input_size_y

def get_cifar10(data_path:str='./data'):

    tr= Compose([ToTensor(), Normalize((0.0), (1.0,))])
    trainset=CIFAR10(data_path, train=True, download=True, transform=tr)
    testset=CIFAR10(data_path, train=False, download=True, transform=tr)

    num_classes= len(trainset.classes)
    input_channels= 3
    input_size_x= trainset.data.shape[1]
    input_size_y= trainset.data.shape[2]
    
    return trainset, testset, num_classes, input_channels, input_size_x, input_size_y

def get_cifar100(data_path:str='./data'):

    tr= Compose([ToTensor(), Normalize((0.1307), (0.3081,))])
    trainset=CIFAR100(data_path, train=True, download=True, transform=tr)
    testset=CIFAR100(data_path, train=False, download=True, transform=tr)
    
    num_classes= len(trainset.classes)
    input_channels= 3
    input_size_x= trainset.data.shape[1]
    input_size_y= trainset.data.shape[2]
    
    return trainset, testset, num_classes, input_channels, input_size_x, input_size_y

    
def get_imagenet(data_path:str='./data'):

   # Define data transformation (resize and normalize)
    transform = transforms.Compose([
        transforms.Resize(256),                    # Resize the input image to 256x256
        transforms.CenterCrop(224),                # Crop the center 224x224 region
        transforms.ToTensor(),                     # Convert the image to a PyTorch tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Normalize the image
    ])

    # Load the ImageNet dataset
    train_dataset = ImageNet(root=data_path, split='train', transform=transform)
    test_dataset = ImageNet(root=data_path, split='val', transform=transform)

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes= 1000
    input_channels= 3
    input_size_x= 224
    input_size_y= 224
    
    return trainloader, testloader,num_classes, input_channels, input_size_x, input_size_y

def prepare_dataset(dataset_name, batch_size:int, val_ratio):
    # num_partions= number of clients= number of partitions to create starting from the training set
    # batch_size= assuming it's the same for every client
    # val_ratio= ratio of validation samples wrt training samples
    
    
     
    if dataset_name == 'mnist':
        trainset, testset, num_classes, input_channels, input_size_x, input_size_y=get_mnist()
    elif dataset_name == 'cifar10':
        trainset, testset, num_classes, input_channels, input_size_x, input_size_y=get_cifar10()
    elif dataset_name == 'cifar100':
        trainset, testset, num_classes, input_channels, input_size_x, input_size_y=get_cifar100()
    elif dataset_name == 'imagenet':
        trainset, testset, num_classes, input_channels, input_size_x, input_size_y=get_imagenet()
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")
    
    
    #We want to partition the training set in training and validation sets
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

    return trainloaders, valloaders, testloader,num_classes, input_channels, input_size_x, input_size_y

