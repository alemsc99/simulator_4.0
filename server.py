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



GLOBAL_EPOCHS = 2
LOCAL_EPOCHS=1



class Server:
    def __init__(self, number_of_clients, testloader, model, device):
        self.number_of_clients = number_of_clients
        self.model=model
        self.testloader = testloader
        self.model_parameters=OrderedDict((key, 0) for key in self.model.state_dict().keys())
        self.device=device
        
    def average_params(self, num_selected_clients):
        params=list(self.model_parameters.values())
        averaged_params=[x/num_selected_clients for x in params]
        new_params=zip(self.model.state_dict().keys(), averaged_params)
        self.model_parameters=OrderedDict({k: torch.Tensor(v) for k,v in new_params})
   
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
            print(f"Starting global epoch {epoch}")
            # Retrieving weights to send to clients
            params_to_send=self.model.state_dict().values()
            for client in clients:
                # Updating clients' weights
                client.set_parameters(log_file, params_to_send)
                client.model.to(self.device)
                # Training the client
                params, _, _=client.fit(LOCAL_EPOCHS, lr, momentum, log_file)
                # Collecting received weights
                self.aggregate_params(params)
            # averaging reveived weights
            self.average_params(len(clients))
            # aggiornamento dei pesi del server 
            if self.model is not None:
                self.model.load_state_dict(self.model_parameters, strict=True)
                log_file.write(f"Model parameters updated on server \n")
                print(f"Model parameters updated on server \n")
            else:
                log_file.write("Cannot update parameters on server because the model is None\n")
        
    
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