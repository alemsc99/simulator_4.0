import csv
import torch
from datetime import datetime
from job import Job, Task
from resnet import ResNet18
from pruning import retrieve_file
from client import Client
from torch.utils.data import Subset, DataLoader

TEST_RATIO=0.1
VAL_RATIO=0.1

def split_loader(loader,  n_clients, batch_size):
    
    # Calculate the total number of samples in the trainloader
    total_samples = len(loader.dataset)
    
    # Calculate the number of samples per client
    samples_per_client = total_samples // n_clients

    # Initialize the list for trainloaders of each client
    client_loaders = []
 
    # Divide the trainloader into equal parts for each client
    for i in range(n_clients):
        # Calculate the indices of samples for the current client
        start_index = i * samples_per_client
        end_index = start_index + samples_per_client

        # Create a Subset of the dataset for the current client
        subset = Subset(loader.dataset, range(start_index, end_index))
        
        # Create a DataLoader for the current client using the Subset
        client_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        # Add the client's train loader to the list
        client_loaders.append(client_loader)
        
    return client_loaders

def generate_nodes(adj_matrix, size, trainloader, valloader, batch_size):
    trainloaders = split_loader(
        loader=trainloader,
        n_clients=size,
        batch_size=batch_size
    )
    valloaders = split_loader(
        loader=valloader,
        n_clients=size,
        batch_size=batch_size
    )

    nodes=[]
    for j in range(0, size):
        neighbors = []
        for i in range(0, size):
            if adj_matrix[j][i] == 1 and j!=i:
                neighbors.append(i)
        
        nodes.append(Client(
                        id=j,
                        neighbors=neighbors,
                        trainloader=trainloaders[j],
                        valloader=valloaders[j]
                        ))
    return nodes
                
    

def load_trained_model(device, filename):
    trained_model=retrieve_file(folder="./models", file_name=filename)
    model=ResNet18(num_classes=10, input_channels=3).to(device)
    model.load_state_dict(torch.load(trained_model))
    return model

def read_csv_file(file_name, device):
    jobs = []
    jobs.append(Job(id=1, generation_time=datetime.now(), dataset= 'cifar10',
                    task=Task('Training', None, None, None, None, False)))  
    with open(file_name, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            isGlobal= row['isGlobal']
            if isGlobal=='True':
                isGlobal=True
            elif isGlobal=='False':
                isGlobal=False
            else:
                isGlobal=None
                
            isRandom= row['isRandom']
            if isRandom=='True':
                isRandom=True
            elif isRandom=='False':
                isRandom=False
            else:
                isRandom=None
                
            isStructured= row['isStructured']
            if isStructured=='True':
                isStructured=True
            elif isStructured=='False':
                isStructured=False
            else:
                isStructured=None
            
            pruning_rate= row['pruning_rate']
            if pruning_rate!='None':
                pruning_rate=float(pruning_rate)
            else:
                pruning_rate=None
                
            pre_trained= row['pre_trained']
            if pre_trained=='True':
                pre_trained=True
            elif pre_trained=='False':
                pre_trained=False
            else:
                pre_trained=None
                
            
            job=Job(
                id=int(row['id']),
                generation_time=datetime.now(),
                dataset= row['dataset'],
                task=Task(
                    name=row['task_name'],
                    isGlobal= isGlobal,
                    isRandom= isRandom,
                    isStructured= isStructured,
                    pruning_rate= pruning_rate,
                    pre_trained= pre_trained
                )
            )
            jobs.append(job)
    return jobs


