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



GLOBAL_EPOCHS = 1
LOCAL_EPOCHS=1
class Server:
    def __init__(self, number_of_clients, testloader, model, device):
        self.number_of_clients = number_of_clients
        self.model=model
        self.testloader = testloader
        self.model_parameters=[]
        self.device=device
        

    def select_clients(self, number_of_clients, clients_to_select):
        random_clients = random.sample(range(number_of_clients), clients_to_select)
        return random_clients

    def send_model(self, clients):
        for client in clients:
            client.model = self.model

    def test_fn(self, model, device, parameters, log_file):
        self.model=model()
        device = device
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        test(
            model,
            self.testloader,
            log_file=log_file,
            device=device,
        ) 


    def start_training(self, clients, momentum, lr, log_file):
      
        for epoch in range(GLOBAL_EPOCHS):
            print(f"Global epoch: {epoch}")
            for client in clients:
                #Training all clients
                client.model.to(self.device)
                params, _, _=client.fit(LOCAL_EPOCHS, lr, momentum, log_file)
                self.model_parameters.append(params)
        # merging received weights 
        print(len(self.model_parameters))
        # aggiornamento dei pesi dei clients
            
    def prune_fn(task, device, log_file):
        if task.name== "Pruning":
            try:
                random_unstructured_pruning(pruning_rate=task.pruning_rate, device=device)               
            except Exception as e:
                log_file.write(f"Error running the script: {e}")
                log_file.flush()
        elif task.name== "LSPruning":
            try:
                local_random_structured_pruning(pruning_rate=task.pruning_rate, device=device)               
            except Exception as e:
                log_file.write(f"Error running the script: {e}")
                log_file.flush()
        else:
            log_file.write(f"Task {task.name} not recognized")
            log_file.flush()