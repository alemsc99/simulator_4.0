import csv
import torch
from datetime import datetime
from job import Job, Task
from resnet import ResNet18
from pruning import retrieve_file
from client import Client
from torch.utils.data import Subset, DataLoader
import numpy
import pandas as pd
from tabulate import tabulate

TEST_RATIO=0.1
VAL_RATIO=0.1



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
                        valloader=valloaders[j],
                        testloader=testloaders[j],
                        gpu_fraction=0.2,
                        device=device
                        ))
    return nodes
                
    

def load_trained_model(device, filename, input_channels, num_classes):
    trained_model=retrieve_file(folder="./models", file_name=filename)
    model=ResNet18(num_classes=num_classes, input_channels=input_channels).to(device)
    model.load_state_dict(torch.load(trained_model))
    return model

def read_csv_file(file_name, fj_global_epochs, fj_local_epochs, dataset):
    jobs = []
    jobs.append(Job(id=1, generation_time=datetime.now(), dataset= dataset,
                    global_epochs=fj_global_epochs, local_epochs=fj_local_epochs,
                    task=Task('Training', None, None, None, None, False)))  
    with open(file_name, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            global_epochs=row['global_epochs']
            if global_epochs!='None':
                global_epochs=int(global_epochs)
            local_epochs=row['local_epochs']
            if local_epochs!='None':
                local_epochs=int(local_epochs)
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
                global_epochs=global_epochs,
                local_epochs=local_epochs,
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


