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
from pruning import random_unstructured_pruning, local_random_structured_pruning
import GPUtil




PRUNED_TRAINED_FILE='lstrainedpruned'
PRUNED_FILE='lspruned'


class Server:
    def __init__(self, number_of_clients, testloader, model, device):
        self.number_of_clients = number_of_clients
        self.model=model
        self.testloader = testloader
        self.model_parameters=OrderedDict((key, 0) for key in self.model.state_dict().keys())
        self.model_saving_path="./models/"
        self.device=device
        
        
        
    def save_model(self, log_file):
        
        path=f"{self.model_saving_path}trained_model.pth"
        torch.save(self.model.state_dict(), f"{path}")
        log_file.write(f"Model saved at {path}\n")
        print(f"Model saved at {path}\n")
        GPUtil.showUtilization(all=False, attrList=None, useOldCode=False)
            

    
        
    def average_params(self, num_selected_clients):
        params=list(self.model_parameters.values())
        averaged_params=[x/num_selected_clients for x in params]
        new_params=zip(self.model.state_dict().keys(), averaged_params)
        self.model_parameters=OrderedDict({k: torch.Tensor(v) for k,v in new_params})
        self.model.load_state_dict(self.model_parameters, strict=True)
   
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
            client.set_parameters(log_file, self.model.state_dict().values())
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


    def start_training(self, training_clients, momentum, lr, log_file, global_epochs, local_epochs, trained_model=None):
        if trained_model is not None:
            self.model=trained_model
        
        for epoch in range(global_epochs):
            print(f"Starting global epoch {epoch}")
            # Retrieving weights to send to clients
            params_to_send=self.model.state_dict().values()
            for client in training_clients:
                # Updating clients' weights after the first global epoch
                if epoch>1:
                    client.set_parameters(log_file, params_to_send)
                client.model.to(self.device)
                # Training the client
                params, _, _=client.fit(local_epochs, lr, momentum, log_file)
                if epoch==global_epochs-1:
                    client.GPU_usage(log_file)
                # Collecting received weights
                self.aggregate_params(params)
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
        if global_epochs==10 and local_epochs==10:
            self.save_model(log_file=log_file)
        
        
    
    def prune_fn(self, task, clients, device, log_file):
        if task.name== "Pruning":
            try:
                random_unstructured_pruning(pruning_rate=task.pruning_rate, device=device)               
            except Exception as e:
                log_file.write(f"Error running the script: {e}")
                log_file.flush()
        elif task.name== "LSPruning":
            try:
                local_random_structured_pruning(
                    model=self.model,
                    pruning_rate=task.pruning_rate, 
                    device=device) 
                self.model_parameters=self.model.state_dict()  
                for client in clients:
                    client.set_parameters(log_file, self.model.state_dict().values())          
            except Exception as e:
                log_file.write(f"Error running the script: {e}")
                log_file.flush()
                
        else:
            log_file.write(f"Task {task.name} not recognized")
            log_file.flush()