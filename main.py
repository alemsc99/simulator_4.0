from datetime import datetime
import os
import torch
import numpy as numpy
from job_handler import JobHandler
from utils import read_csv_file, load_trained_model, generate_nodes
from server import Server
from dataset import prepare_dataset
from resnet import ResNet18


JOBS_FILE= './jobs.csv'
NUMBER_OF_NODES=10
DATASET= 'cifar10'
BATCH_SIZE=64
VAL_RATIO=0.1
NUM_CLASSES=10
INPUT_CHANNELS=3
CLIENTS_TO_SELECT=2
FJ_GLOBAL_EPOCHS=10
FJ_LOCAL_EPOCHS=10
NUMBER_OF_SIMULATIONS=5

def main():
    
    device="cuda" if torch.cuda.is_available() else "cpu"  
    
    #Prepare dataset     
    trainloader, valloader, testloader=prepare_dataset(DATASET, BATCH_SIZE, VAL_RATIO)
    
    
    #Network definition
    server=Server(
        number_of_clients=NUMBER_OF_NODES, 
        testloader=testloader,
        model=ResNet18(num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS).to(device), #MODEL DEFINITION
        device=device)
    
    
    
    adjacency_matrix = [
        [0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 1, 1]
    ] 
    try:
        nodes=generate_nodes(
            adj_matrix=adjacency_matrix, 
            size=NUMBER_OF_NODES, 
            trainloader=trainloader, 
            valloader=valloader,
            testloader=testloader,
            batch_size=BATCH_SIZE,
            device=device)
        print(f"Generated {NUMBER_OF_NODES} nodes")
    except Exception as e:
        print(e)
        exit()
    
    try:
        selected_clients=server.select_clients(NUMBER_OF_NODES, clients_to_select=CLIENTS_TO_SELECT)
        print(f"Selected {len(selected_clients)} clients.")
        training_clients = [node for node in nodes if node.id in selected_clients]
        remaining_clients = [node for node in nodes if node.id not in selected_clients]
    except Exception as e:
        print(e)
        exit()
    
    
    #Sending model to clients    
    server.send_model(nodes) 
   
    #Opening logmanager file
    log_folder = "simulation_logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_file = open(os.path.join(log_folder, 'job_manager_logs.txt'), 'a')
    log_file.write("-"*5 + f" Simulation started at {datetime.now()} " + "-"*5 + "\n")
    log_file.flush() 
  
    
    
    job_handler=JobHandler(device=device)
   
    #Jobs generation
    for i in range(0,NUMBER_OF_SIMULATIONS):
        jobs=read_csv_file(JOBS_FILE, FJ_GLOBAL_EPOCHS, FJ_LOCAL_EPOCHS, DATASET)
        log_file.write('\n')
        log_file.write("*"*5 +f" Iteration number {i}. Current number of jobs {len(jobs)} "+ "*"*5 + "\n")
        log_file.flush()
        
        job_handler.handle_job(jobs_list=jobs, 
                               log_file=log_file,
                               device=device, 
                               server=server,
                               training_clients=training_clients,
                               remaining_clients=remaining_clients,
                               input_channels=INPUT_CHANNELS,
                               num_classes=NUM_CLASSES,
                               simulation_number=i
                               )
       
    
   
    log_file.write("-"*5 +f" Simulation ended at {datetime.now()} "+"-"*5+"\n")
    log_file.flush()
           
if __name__=="__main__":   
    main()
