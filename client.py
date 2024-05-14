'''
The Client class has to:
- receive weights from the server 
- train the model on the local training set 
- send weights to the server
'''
import torch
from model import train, test



class Client:
    def __init__(self, id:int, neighbors:list, trainloader, valloader, testloader, gpu_fraction, device):
        self.id=id
        self.neighbors = neighbors
        self.model=None
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.gpu_fraction = gpu_fraction
        self.GPU_usage_table=None
        self.device = device
        
        
        
    def GPU_usage(self, log_file):
        #Printing GPU utilization at the end of global epochs
        log_file.write(f"Finished training model of client {self.id}\n")
        log_file.write(str(self.GPU_usage_table))
        log_file.write("\n")
        log_file.flush()
    
    
    def set_parameters(self, log_file, parameters_dict):
        #it recives the parameters from the server
        if self.model is not None:
            self.model.load_state_dict(parameters_dict, strict=True)
            log_file.write(f"Model parameters updated on client {self.id}\n")
            print(f"Model parameters updated on client {self.id}\n")
        else:
            log_file.write(f"Cannot update parameters on client {self.id} because the model is None\n")
            
    def get_parameters(self):
        #it sends back parameters to the server
        parameters=self.model.state_dict()
        return parameters
    
    
    def fit(self,  epochs, optimizer, gradient_clipping, log_file):
        #parameters is a list of numpy arrays representing the current state of the global model
        #config is a python dictionary with additional information

        # When this client starts the computation, it receives the weights from the server so we want to 
        # overwrite generate_nodesthe initial random weights with the ones sent from the server
        log_file.write(f"Starting training model of client {self.id}\n")
        print(f"Starting training model of client {self.id}\n")

    
        
        
        # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        # session = tf.compat.v1.Session(config=config)
     
        
        train(net=self.model,
              trainloader=self.trainloader, 
              valloader=self.valloader, 
              optimizer=optimizer, 
              epochs=epochs, 
              gradient_clipping=gradient_clipping,
              log_file=log_file, 
              device=self.device)

        # Send back the updated model paraemters to the server with get_parameters()
        # it also returns the number of samples used by this client for training
        # and metrics about training

        return self.get_parameters(), len(self.trainloader), {}




    def client_test(self,log_file, device):
        log_file.write(f"Starting testing on client {self.id} \n")
        print(f"Starting testing on client {self.id} \n")
        loss, accuracy=test(
            net=self.model,
            testloader=self.testloader,
            log_file=log_file,
            device=device,
        )
        
        return loss, accuracy
        
   
            