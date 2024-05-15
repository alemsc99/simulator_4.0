from datetime import datetime
import os
import torch
import numpy as numpy
from job_handler import JobHandler
from utils import read_csv_file, generate_nodes, generate_adjacency_matrix, define_model
from server import Server
from dataset import prepare_dataset



BACKBONE_NAME='ResNet18'
JOBS_FILE='./jobs.csv'
NUMBER_OF_NODES= 10
DATASET='cifar10'
STEP_SIZE= 5 # LR scheduler step
GAMMA=0.1 #LR scheduler factor
BATCH_SIZE=64
VAL_RATIO=0.1
CLIENTS_TO_SELECT=1
FJ_GLOBAL_EPOCHS=1
FJ_LOCAL_EPOCHS=1
NUMBER_OF_SIMULATIONS=1
NETWORK_DENSITY=0.5
LR=0.001
MOMENTUM=0.9
GRADIENT_CLIPPING=None


def main():
    
    device="cuda" if torch.cuda.is_available() else "cpu"  
    
    #Opening logmanager file
    log_folder = "simulation_logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_file = open(os.path.join(log_folder, 'job_manager_logs.txt'), 'a')
    
    for i in range(0,NUMBER_OF_SIMULATIONS):
        #Prepare dataset     
        trainloader, valloader, testloader, num_classes, input_channels, input_size_x, input_size_y=prepare_dataset(DATASET, BATCH_SIZE, VAL_RATIO)
        
        #Model definition
        model=define_model(backbone_name=BACKBONE_NAME, num_classes=num_classes, input_channels=input_channels, input_size=input_size_x)
        
        #Network definition
        server=Server(
            number_of_clients=NUMBER_OF_NODES, 
            testloader=testloader,
            model=model.to(device), 
            number_of_channels=input_channels,
            input_size_x=input_size_x,
            input_size_y=input_size_y,
            lr=LR,
            momentum=MOMENTUM,
            step_size=STEP_SIZE,
            gamma=GAMMA,
            device=device)
        
        adjacency_matrix = generate_adjacency_matrix(num_nodes=NUMBER_OF_NODES, density=NETWORK_DENSITY)
    
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


        # iid_im, iid_lab, iid_im_wxn, iid_lab_wxn=check_iid(nodes)
        # print(iid_im)
        # print(iid_lab)
        # print(iid_im_wxn)
        # print(iid_lab_wxn)



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

        log_file.write("-"*5 + f" Simulation started at {datetime.now()} " + "-"*5 + "\n")
        log_file.write(f'Device: {device} \n')
        log_file.write(f"Backbone: {BACKBONE_NAME} \n")
        log_file.write(f"Clients selected for training: {selected_clients}\n")
        log_file.flush() 
    


        job_handler=JobHandler(device=device)
    
        #Jobs generation

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
                               input_channels=input_channels,
                               num_classes=num_classes,
                               gradient_clipping=GRADIENT_CLIPPING,
                               input_size=input_size_x,
                               backbone=BACKBONE_NAME,
                               simulation_number=i
                               )
       
    
   
    log_file.write("-"*5 +f" Simulation ended at {datetime.now()} "+"-"*5+"\n")
    log_file.flush()
           
if __name__=="__main__":   
    main()
