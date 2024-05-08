
import torch
from datetime import datetime
from utils import show_clients_loss, write_matrix
from pruning import random_unstructured_pruning, local_random_structured_pruning
import numpy

PRUNED_TRAINED_FILE='lstrainedpruned'
PRUNED_FILE='lspruned'
from utils import load_trained_model

class JobHandler:
    def __init__(self, device) -> None:
        
        self.lr = 0.0001
        self.momentum = 0.9
        
        self.device=device
       





    def handle_job(self, jobs_list, log_file, device,server, training_clients, remaining_clients, input_channels,num_classes, simulation_number):
        number_of_clients=len(remaining_clients)+len(training_clients)
        results_matrix=numpy.zeros((5, number_of_clients))
        line_to_replace=0
        
        for job in jobs_list:
            starting_time=datetime.now()
            trained_model=None
            log_file.write(f"[{starting_time}]\n" + f"Handling job with id: {job.id}, generation time: {job.generation_time}, task: {job.task.name}, pruning_rate: {job.task.pruning_rate}" + "\n")
            log_file.flush()
            torch.cuda.empty_cache()
            
            if job.task.pre_trained:
                filename='trained_model.pth'
                trained_model=load_trained_model(device, filename, input_channels, num_classes)      
            
        
            if job.task.name == "Training":                
                try:
                    server.start_training(job_id=job.id,
                                          training_clients=training_clients,
                                          momentum=self.momentum,
                                          lr=self.lr, 
                                          log_file=log_file, 
                                          global_epochs=job.global_epochs,
                                          local_epochs=job.local_epochs,
                                          trained_model=trained_model)
                    #self.training_function(log_file, job.task.pruning_rate, job.num_epochs, server, selected_clients)
                except Exception as e:
                    log_file.write(f"Error running the script: {e} \n")
                    log_file.flush()
                    
                    
            elif job.task.name == "Testing":
                clients=list(set(training_clients+remaining_clients))
                try:
                    server_loss, server_acc, clients_loss, clients_acc=server.start_testing(clients=clients,
                                                                                            device=device, 
                                                                                            log_file=log_file, 
                                                                                            trained_model=trained_model)
                    #loss, accuracy = self.testing_function(testloaders, log_file)
                    show_clients_loss(clients_loss, clients_acc, log_file)
                    
                    log_file.write(f"Server loss: {server_loss}, Server accuracy: {server_acc} \n")
                    log_file.flush()
                    if job.id in [4, 7, 10, 13, 16]:
                        results_matrix[line_to_replace, :]=clients_acc
                        line_to_replace=line_to_replace+1
                    print("Testing finished \n")
                except Exception as e:
                    log_file.write(f"Error running the script: {e} \n")
                    log_file.flush()
                    
                    
            elif job.task.name== "Pruning":
                try:
                    random_unstructured_pruning(pruning_rate=job.task.pruning_rate, device=device)               
                except Exception as e:
                    log_file.write(f"Error running the script: {e} \n")
                    log_file.flush()
                    
                    
            elif job.task.name== "LSPruning":
                clients=list(set(training_clients+remaining_clients))
                try:
                    server.prune_fn(task=job.task,
                                    clients=clients,
                                    device=device,
                                    log_file=log_file,
                                    )
                    #local_random_structured_pruning(pruning_rate=job.task.pruning_rate, device=device)               
                except Exception as e:
                    log_file.write(f"Error running the script: {e} \n")
                    log_file.flush()
                    
                    
            else:
                log_file.write(f"Task {job.task.name} not recognized")
                log_file.flush()
                
                
                
             
            log_file.write(f"[{datetime.now()}]" + f" Completed job {job.id}.\nComputation time: {datetime.now()-starting_time}.\n")
            log_file.flush()
            
        write_matrix(results_matrix, simulation_number)
        
                    
      
        
            
        
