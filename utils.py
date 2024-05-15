import sys

from backbones.cnn import ConvNeuralNet
sys.path.append('./backbones')
from collections import defaultdict
import csv
import random
import torch
from datetime import datetime
from job import Job, Task
from pruning import retrieve_file
from client import Client
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler
import numpy
import pandas as pd
from tabulate import tabulate
from scipy import stats
import numpy as np
import psutil
from torchprofile import profile_macs
from backbones.resnet18 import ResNet18
from backbones.resnet50 import ResNet50
from backbones.vgg import VGG16

TEST_RATIO=0.1
VAL_RATIO=0.1


def define_model(backbone_name, num_classes, input_channels, input_size):
    
    if backbone_name=='ResNet18':
        model=ResNet18(num_classes=num_classes, input_channels=input_channels)#MODEL DEFINITION
    elif backbone_name=='ResNet50':
        model=ResNet50(num_classes=num_classes, input_channels=input_channels)#MODEL DEFINITION
    elif backbone_name=='VGG16':
        model=VGG16(input_channels=input_channels, input_size=input_size, num_classes=num_classes, dropout_prob=0.0)#MODEL DEFINITION
    elif backbone_name=='CNN':
        model=ConvNeuralNet(num_classes=num_classes, input_channels=input_channels)#MODEL DEFINITION
    else:
        raise ValueError("Backbone name must be either ResNet18, ResNet50, VGG16 or CNN") 
    
    return model
    
def measure_latency_cpu_usage(model, device, num_channels, input_size_x, input_size_y):
    #Latency is the amount of time it takes for a neural network to produce a prediction for a single input sample.
    dummy_input = torch.randn(3,num_channels,input_size_x,input_size_y, dtype=torch.float).to(device)
    model.to(device)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    process = psutil.Process()
    cpu_start = process.cpu_percent()
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    cpu_end = process.cpu_percent()
    cpu_usage = cpu_end - cpu_start
    latency = np.sum(timings) / repetitions
    
    return latency, cpu_usage
   

def measure_gpu_throughput_and_macs(model, batch_size, device, num_channels, input_size_x, input_size_y):
    # The throughput of a neural network is defined as the maximal 
    # number of input instances the network can process in a unit of time
    dummy_input = torch.randn(batch_size, num_channels,input_size_x,input_size_y, dtype=torch.float).to(device)
    model = model.to(device)
    repetitions=100
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time
    throughput =   (repetitions*batch_size)/total_time
    macs=profile_macs(model, dummy_input)
    return throughput, macs


def check_iid(nodes):    
    threshold=0.05
    # p_value < threshold => Non IID
    # p_value > threshold => IID
    results_im_ttest=np.empty((len(nodes), len(nodes)), dtype=object)
    results_lab_ttest=np.empty((len(nodes), len(nodes)), dtype=object)
    results_im_wxn=np.empty((len(nodes), len(nodes)), dtype=object)
    results_lab_wxn=np.empty((len(nodes), len(nodes)), dtype=object)
    for i in range(0, len(nodes)):
        data=[]
        labels=[]
        for (im,lab) in nodes[i].trainloader:
            flattened_tensor=im.flatten()
            values=flattened_tensor.tolist()
            data+=values
            lab=lab.flatten()
            lab=lab.tolist()
            labels+=lab
        for j in range(i+1, len(nodes)):
            data_two=[]
            labels_two=[]
            for (imm, labb) in nodes[j].trainloader:
                flattened_tensor=imm.flatten()
                values=flattened_tensor.tolist()
                data_two+=values
                labb=labb.flatten()
                labb=labb.tolist()
                labels_two+=labb
            res_im=stats.ttest_ind(np.array(data),np.array(data_two))
            res_lab=stats.ttest_ind(np.array(labels),np.array(labels_two))
            pvalue_im, pvalue_lab=res_im.pvalue, res_lab.pvalue
            res_im_wxn=stats.wilcoxon(np.array(data),np.array(data_two))
            res_lab_wxn=stats.wilcoxon(np.array(labels),np.array(labels_two))
            pvalue_im_wxn, pvalue_lab_wxn=res_im_wxn.pvalue, res_lab_wxn.pvalue
            
            results_im_ttest[i,j]=(pvalue_im<threshold)
            results_lab_ttest[i,j]=(pvalue_lab<threshold)
            results_im_wxn[i,j]=(pvalue_im_wxn<threshold)
            results_lab_wxn[i,j]=(pvalue_lab_wxn<threshold)
            
    return results_im_ttest, results_lab_ttest, results_im_wxn, results_lab_wxn
            
            
    
    


def generate_adjacency_matrix(num_nodes, density):
    num_ones = int((num_nodes * (num_nodes - 1) / 2) * density) # Calculate the number of ones based on the percentage
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)] # Initialize an empty adjacency matrix
    
    # Generate random positions for ones
    positions = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]
    random.shuffle(positions)
    
    # Set ones randomly based on the calculated number of ones
    for i in range(num_ones):
        x, y = positions[i]
        adjacency_matrix[x][y] = adjacency_matrix[y][x] = 1
    
    return adjacency_matrix

def write_matrix(results_matrix, simulation_number):
    #Computing variance
    results_matrix_variance =numpy.var(results_matrix)
    
    # Calculate the mean of each row
    row_means = results_matrix.mean(axis=1)

    # Append a new column with the mean of each row to the matrix
    results_matrix_with_mean = numpy.column_stack((results_matrix, row_means))

    # Save the matrix to a CSV file
    path = "./simulation_logs/"

    # Custom names for columns and rows
    column_names = [f"Client {c}" for c in range(results_matrix.shape[1])]
    column_names.append("Accuracy Mean")

    row_names = ['No Pruning', 'Pruning 20%', 'Pruning 40%', 'Pruning 60%', 'Pruning 80%']

    # Convert the NumPy array to a pandas DataFrame with custom column and row names
    df = pd.DataFrame(results_matrix_with_mean, index=row_names, columns=column_names)

    # Save the DataFrame to a CSV file
    df.to_csv(f'{path}accuracies_{simulation_number}.csv', float_format='%.3f')
    
    # Visualize the table
    table_str = tabulate(df, headers='keys', tablefmt='grid')

    
    # Add a line break and the variance below the table
    table_str += f"\n\nVariance of the matrix: {results_matrix_variance}"

    # Save the visualized table to a text file
    with open(f'{path}accuracies_{simulation_number}.txt', 'w') as file:
        file.write(table_str)

       
def show_clients_loss(clients_loss, clients_acc, log_file):
    for id, (loss, acc) in enumerate(zip(clients_loss, clients_acc)):
        log_file.write(f"Client {id} loss: {loss}, Client accuracy: {acc}\n")

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

def split_loader_new(loader, n_clients, batch_size):
    # Calculate the total number of samples in the trainloader
    total_samples = len(loader.dataset)
    
    # Create a dictionary to store indices for each class
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(loader.dataset):
        class_indices[label].append(idx)
    
    # Calculate the number of samples per client
    total_samples_per_client = total_samples // (n_clients*2)
    
    # Initialize the list for trainloaders of each client
    client_loaders = []

    # Divide the trainloader into parts for each client
    for i in range(n_clients):
        # Calculate the number of samples for the current client
        client_samples = int(total_samples_per_client)
        
        # In case there are remaining samples, distribute them to the first few clients
        if i < total_samples % n_clients:
            client_samples += 1
        
        # Initialize a list to store indices for the current client
        client_indices = []
        
        # Iterate over each class and shuffle the indices
        # for indices in class_indices.values():
        #     np.random.shuffle(indices)
        
        # Select indices for the current client
        for indices in class_indices.values():
            client_indices.extend(indices[:client_samples // n_clients])
        
        # Shuffle the indices for the current client
        #np.random.shuffle(client_indices)
        
        # Create a SubsetRandomSampler for the current client
        sampler = SubsetRandomSampler(client_indices)
        
        # Create a DataLoader for the current client using the SubsetRandomSampler
        client_loader = DataLoader(loader.dataset, batch_size=batch_size, sampler=sampler)
        
        # Add the client's train loader to the list
        client_loaders.append(client_loader)
        
    return client_loaders


def generate_nodes(adj_matrix, size, trainloader, valloader, testloader, batch_size, device):
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
    
    testloaders=split_loader(
        loader=testloader,
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
                        #trainloader=trainloader,
                        valloader=valloaders[j],
                        testloader=testloaders[j],
                        gpu_fraction=1.0,
                        device=device
                        ))

    return nodes
                
    

def load_trained_model(filename, input_channels, num_classes, input_size, backbone):
    trained_model=retrieve_file(folder="./models", file_name=filename)
    model=define_model(backbone, num_classes, input_channels, input_size)
    model.load_state_dict(torch.load(trained_model))
    return model

def read_csv_file(file_name, fj_global_epochs, fj_local_epochs, dataset):
    jobs = []
    jobs.append(Job(id=1, generation_time=datetime.now(), 
                    global_epochs=fj_global_epochs, local_epochs=fj_local_epochs,
                    task=Task('Training', None, None, False)))  
    with open(file_name, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            global_epochs=row['global_epochs']
            if global_epochs!='None':
                global_epochs=int(global_epochs)
            local_epochs=row['local_epochs']
            if local_epochs!='None':
                local_epochs=int(local_epochs)
           
                
            isRandom= row['isRandom']
            if isRandom=='True':
                isRandom=True
            elif isRandom=='False':
                isRandom=False
            else:
                isRandom=None
                
           
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
                global_epochs=global_epochs,
                local_epochs=local_epochs,
                task=Task(
                    name=row['task_name'],
                    
                    isRandom= isRandom,
                   
                    pruning_rate= pruning_rate,
                    pre_trained= pre_trained
                )
            )
            jobs.append(job)
    return jobs


def get_inf_params(net, verbose=True, sd=False):
    if sd:
        params = net
    else:
        params = net.state_dict()
    tot = 0
    conv_tot = 0
    for p in params:
        no = params[p].view(-1).__len__()

        if ('num_batches_tracked' not in p) and ('running' not in p) and ('mask' not in p):
            tot += no

            if verbose:
                print('%s has %d params' % (p, no))
        if 'conv' in p:
            conv_tot += no

    if verbose:
        print('Net has %d conv params' % conv_tot)
        print('Net has %d params in total' % tot)

    return tot