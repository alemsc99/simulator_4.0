from datetime import datetime
import os
import torch
import numpy as numpy
from job_handler import JobHandler
from utils import read_csv_file, generate_nodes, generate_adjacency_matrix
from server import Server
from dataset import prepare_dataset
from resnet import ResNet18
from utils import check_iid


JOBS_FILE= './jobs.csv'
NUMBER_OF_NODES=10
DATASET= 'cifar10'
BATCH_SIZE=64
VAL_RATIO=0.1
NUM_CLASSES=10
INPUT_CHANNELS=3
INPUT_SIZE_X=32
INPUT_SIZE_Y=32
CLIENTS_TO_SELECT=2
FJ_GLOBAL_EPOCHS=4
FJ_LOCAL_EPOCHS=4
NUMBER_OF_SIMULATIONS=5
NETWORK_DENSITY=0.5

def main():
    
    device="cuda" if torch.cuda.is_available() else "cpu"  
    
    #Opening logmanager file
    log_folder = "simulation_logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_file = open(os.path.join(log_folder, 'job_manager_logs.txt'), 'a')
    
    for i in range(0,NUMBER_OF_SIMULATIONS):
        #Prepare dataset     
        trainloader, valloader, testloader=prepare_dataset(DATASET, BATCH_SIZE, VAL_RATIO)
        
        
        #Network definition
        server=Server(
            number_of_clients=NUMBER_OF_NODES, 
            testloader=testloader,
            model=ResNet18(num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS).to(device), #MODEL DEFINITION
            number_of_channels=INPUT_CHANNELS,
            input_size_x=INPUT_SIZE_X,
            input_size_y=INPUT_SIZE_Y,
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
                               input_channels=INPUT_CHANNELS,
                               num_classes=NUM_CLASSES,
                               simulation_number=i
                               )
       
    
   
    log_file.write("-"*5 +f" Simulation ended at {datetime.now()} "+"-"*5+"\n")
    log_file.flush()
           
if __name__=="__main__":   
    main()
