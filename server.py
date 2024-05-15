"""
The server class has to:
- send initial weights to the clients
- receive new weights from the clients
- merge the weights received
- sent back to the clients the new model
"""

import random
from collections import OrderedDict
import torch
from model import test
from pruning import global_unstructured_pruning, local_structured_pruning, local_unstructured_pruning
from utils import measure_latency_cpu_usage, measure_gpu_throughput_and_macs
from torch.optim.lr_scheduler import StepLR

PRUNED_TRAINED_FILE='lstrainedpruned'
PRUNED_FILE='lspruned'


class Server:
    def __init__(self, number_of_clients, testloader, model, number_of_channels, input_size_x, input_size_y, lr, momentum, step_size, gamma, device):
        self.number_of_clients = number_of_clients
        self.model=model
        self.optimizer=torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.scheduler=StepLR(optimizer=self.optimizer, step_size= step_size, gamma=gamma)
        self.testloader = testloader
        self.model_parameters=OrderedDict((key, 0) for key in self.model.state_dict().keys())
        self.model_saving_path="./models/"
        self.comp_model_saving_path="./models/"
        self.number_of_channels = number_of_channels
        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        self.device=device
        
        
        
    def save_model(self, log_file, pruning_rate):
        if pruning_rate==0.0:
            filename='trained_model'   
        else:
            pruning_rate_str= "{:02d}".format(int(pruning_rate * 10))
            filename=f"pruned_{pruning_rate_str}"    
        path=f"{self.model_saving_path}{filename}.pth"
        torch.save(self.model.state_dict(), f"{path}")
        log_file.write(f"Model saved at {path}\n")
        print(f"Model saved at {path}\n")
        latency, cpu_usage=measure_latency_cpu_usage(self.model, self.device, self.number_of_channels, self.input_size_x, self.input_size_y)
        log_file.write(f"Latency: {latency} ms\n")
        log_file.write(f"CPU Usage: {cpu_usage}%\n")
        log_file.flush()
        throughput, macs=measure_gpu_throughput_and_macs(self.model, 128, self.device, self.number_of_channels, self.input_size_x, self.input_size_y)
        log_file.write(f"Throughput: {throughput}\n")
        log_file.flush()
        log_file.write(f"MACs: {macs / (1024 ** 3)} G\n")
        log_file.flush()
        log_file.write(f"FLOPs: {(2*macs) / (1024 ** 3)} G\n")
        log_file.flush()
        
            

    
        
    def average_params(self, num_selected_clients):
        for key, values in self.model_parameters.items():
            self.model_parameters[key] = values/num_selected_clients
        
        
    def aggregate_params_dict(self, parameters_dict):        
        merged_dict = OrderedDict()

        for key, value in parameters_dict.items():
            if key in merged_dict:
                merged_dict[key] += value
            else:
                merged_dict[key] = value

        for key, value in self.model_parameters.items():
            if key in merged_dict:
                res=merged_dict[key]+value
                merged_dict[key]=res
            else:
                merged_dict[key] = value
                
        self.model_parameters=merged_dict
   
    def aggregate_params(self, parameters):        
        starting_params=list(self.model_parameters.values())
        received_params=list(parameters)
        # Sum the corresponding elements of the two lists
        updated_params = [x + y for x, y in zip(starting_params, received_params)]
        new_params=zip(self.model.state_dict().keys(), updated_params)
        self.model_parameters=OrderedDict({k: torch.Tensor(v) for k,v in new_params})
   
    def select_clients(self, number_of_clients, clients_to_select):
        random_clients = random.sample(range(number_of_clients), clients_to_select)
        return random_clients

    def send_model(self, clients):
        for client in clients:
            client.model = self.model
            
            

    def start_testing(self, clients, device, log_file, trained_model=None):
        clients_loss = []
        clients_acc=[]
        
        if trained_model is not None:
            self.model=trained_model
            
        # Testing on clients
        log_file.write("Starting testing on clients \n")
        log_file.flush()
        print("Starting testing on clients \n")
        for client in clients:
            client.set_parameters(log_file, self.model.state_dict())
            loss, acc=client.client_test(log_file, device)
            clients_loss.append(loss)
            clients_acc.append(acc)
        
        # Testing on server
        self.model.to(device)            
        log_file.write("Starting testing on server \n")
        log_file.flush()
        print("Starting testing on server \n")
        if self.model is not None:
            self.model.load_state_dict(self.model_parameters, strict=True)
            log_file.write(f"Model parameters updated on server \n")
            log_file.flush()
            print(f"Model parameters updated on server \n")
        else:
            log_file.write("Cannot update parameters on server because the model is None\n")
            log_file.flush()
            
        server_loss, server_acc=test(
            self.model,
            self.testloader,
            log_file=log_file,
            device=device,
        ) 
        
        return server_loss, server_acc, clients_loss, clients_acc


    def start_training(self, job_id,training_clients, log_file, global_epochs, local_epochs, gradient_clipping,trained_model=None):
        
        if trained_model is not None:
            self.model=trained_model
        
        for epoch in range(global_epochs):
            self.scheduler.step()  # Update the learning rate
            log_file.write("New learning rate: " + str(self.optimizer.param_groups[0]['lr'])+'\n')
            log_file.write(f"Starting global epoch {epoch}\n")
            print(f"Starting global epoch {epoch}")
            # Retrieving weights to send to clients
            #params_to_send_dict=self.model.state_dict()
            for client in training_clients:
                # Updating clients' weights after the first global epoch
                # if epoch>0:
                #     client.set_parameters(log_file, params_to_send_dict)
                
                # Training the client
                params_dict, _, _=client.fit(local_epochs, self.optimizer, gradient_clipping, log_file)
                
                # Collecting received weights
                self.aggregate_params_dict(params_dict)
            # averaging reveived weights
            self.average_params(len(training_clients))
            # aggiornamento dei pesi del server 
            if self.model is not None:
                self.model.load_state_dict(self.model_parameters, strict=True)
                log_file.write(f"Model parameters updated on server \n")
                log_file.flush()
                print(f"Model parameters updated on server \n")
            else:
                log_file.write("Cannot update parameters on server because the model is None\n")
                log_file.flush()
        if job_id==1:
            self.save_model(log_file=log_file, pruning_rate=0.0)
        
        
    
    def prune_fn(self, task, clients, device, log_file):
        if task.name== "GUPruning":
            try:
                global_unstructured_pruning(
                    model=self.model,
                    pruning_rate=task.pruning_rate,
                    isRandom=task.isRandom
                    ) 
                self.model_parameters=self.model.state_dict()  
                # for client in clients:
                #     client.set_parameters(log_file, self.model.state_dict())
                self.save_model(log_file, task.pruning_rate)       
            except Exception as e:
                log_file.write(f"Error running the script: {e}")
                log_file.flush()
        elif task.name== "LSPruning":
            try:
                local_structured_pruning(
                    model=self.model,
                    pruning_rate=task.pruning_rate,
                    isRandom=task.isRandom
                    ) 
                self.model_parameters=self.model.state_dict()  
                # for client in clients:
                #     client.set_parameters(log_file, self.model.state_dict())
                self.save_model(log_file, task.pruning_rate)          
            except Exception as e:
                log_file.write(f"Error running the script: {e}")
                log_file.flush()
        elif task.name== "LUPruning":
            try:
                local_unstructured_pruning(
                    model=self.model,
                    pruning_rate=task.pruning_rate,
                    isRandom=task.isRandom
                    ) 
                self.model_parameters=self.model.state_dict()  
                # for client in clients:
                #     client.set_parameters(log_file, self.model.state_dict())
                self.save_model(log_file, task.pruning_rate)          
            except Exception as e:
                log_file.write(f"Error running the script: {e}")
                log_file.flush()                 
        else:
            log_file.write(f"Task {task.name} not recognized")
            log_file.flush()