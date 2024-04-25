

from model import train, test
import torch
from datetime import datetime
import GPUtil
from pruning import random_unstructured_pruning, local_random_structured_pruning
from utils import load_trained_model

PRUNED_TRAINED_FILE='lstrainedpruned'
PRUNED_FILE='lspruned'


class JobHandler:
    def __init__(self) -> None:
        
        self.batch_size = 64
        self.lr = 0.0001
        self.momentum = 0.9
        
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_saving_path="./models/"
            
            
            
            
    def training_function(self, log_file, pruning_rate, num_epochs, server, selected_clients):
        #Training all the selected clients
        for client in selected_clients:
            client.model.to(self.device)
            parameters, len_trainloaders, _=client.fit(epochs=num_epochs, lr=self.lr, momentum=self.momentum, log_file=log_file)
            print("Training finished")
                    
            if pruning_rate==0.0:
                path=f"{self.model_saving_path}trained_model.pth"
            else:
                pruning_rate_str= "{:02d}".format(int(pruning_rate * 10))
                path=f"{self.model_saving_path}{PRUNED_TRAINED_FILE}_{pruning_rate_str}.pth"
        # Once the training finished, the server merges the weights received from the clients and saves it   
        torch.save(model.state_dict(), f"{path}")
        print(f"Model saved at {path}")
        GPUtil.showUtilization(all=False, attrList=None, useOldCode=False)
            
        return len_trainloaders, {}
    
    
    
    
        
    def testing_function(self, valloaders, log_file):
        loss, accuracy=test(model, valloaders, log_file,  device=self.device)
        print("Testing finished")
        return len(valloaders), {"loss": loss, "accuracy": accuracy}





    def handle_job(self, jobs_list, log_file, device,server, selected_clients):
        # # Prepare the dataset
        # if job.dataset == "mnist":
        #     trainloaders, valloaders, testloaders=prepare_dataset(dataset_name='mnist', batch_size=self.batch_size)
        # elif job.dataset == "cifar10":
        #     trainloaders, valloaders, testloaders=prepare_dataset(dataset_name='cifar10', batch_size=self.batch_size)
        # else:
        #     log_file.write(f"Dataset {job.dataset} not recognized")
        #     log_file.flush()
      
        
        for job in jobs_list:
            starting_time=datetime.now()
            log_file.write(f"[{starting_time}]\n" + f"Handling job with id: {job.id}, generation time: {job.generation_time}, task: {job.task.name}, pruning_rate: {job.task.pruning_rate}" + "\n")
            log_file.flush()
            torch.cuda.empty_cache()
            if job.task.pre_trained and job.task.name=='Training':
                if job.task.pruning_rate==0.0:
                    filename='trained_model.pth'
                else:
                    pruning_rate_str= "{:02d}".format(int(job.task.pruning_rate * 10))
                    filename=f'{PRUNED_FILE}_{pruning_rate_str}.pth'  
                     
                trained_model=load_trained_model(device, filename)
                
            elif job.task.pre_trained and job.task.name=='Testing':
                if job.task.pruning_rate==0.0:
                    filename='trained_model.pth'
                else:
                    pruning_rate_str= "{:02d}".format(int(job.task.pruning_rate * 10))
                    filename=f'{PRUNED_TRAINED_FILE}_{pruning_rate_str}.pth'
            
                trained_model=load_trained_model(device, filename)
                
                log_file.write(f"[{datetime.now()}]" + f" Completed job {job.id}.\nComputation time: {datetime.now()-starting_time}.\n")
                log_file.flush()
                    
      
        
       
            
        
            if job.task.name == "Training":                
                try:
                    server.start_training(selected_clients, self.momentum, self.lr, log_file)
                    #self.training_function(log_file, job.task.pruning_rate, job.num_epochs, server, selected_clients)
                except Exception as e:
                    log_file.write(f"Error running the script: {e}")
                    log_file.flush()
                    
                    
            elif job.task.name == "Testing":
                try:
                    loss, accuracy = self.testing_function(testloaders, log_file)
                    print(f"Loss: {loss}, Accuracy: {accuracy}")
                except Exception as e:
                    log_file.write(f"Error running the script: {e}")
                    log_file.flush()
                    
                    
            elif job.task.name== "Pruning":
                try:
                    random_unstructured_pruning(pruning_rate=job.task.pruning_rate, device=device)               
                except Exception as e:
                    log_file.write(f"Error running the script: {e}")
                    log_file.flush()
                    
                    
            elif job.task.name== "LSPruning":
                try:
                    local_random_structured_pruning(pruning_rate=job.task.pruning_rate, device=device)               
                except Exception as e:
                    log_file.write(f"Error running the script: {e}")
                    log_file.flush()
                    
                    
            else:
                log_file.write(f"Task {job.task.name} not recognized")
                log_file.flush()
            
        
