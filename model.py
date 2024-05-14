import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pruning import calculate_sparsity_and_NNZ
import psutil
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
from gputil_decorator import gputil_decorator



writer=SummaryWriter()
SIMULATION_LOGS_FOLDER='./simulation_logs'
# Function to print GPU and CPU utilization
def print_utilization(device, log_file):
    gpu_usage = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GPU memory usage in GB
    gpu_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)  # Total GPU memory reserved by PyTorch in GB
    log_file.write(f"GPU Utilization: {gpu_usage:.2f} GB / {gpu_reserved:.2f} GB\n")

    cpu_usage = psutil.cpu_percent()
    log_file.write(f"CPU Utilization: {cpu_usage:.2f}%\n")
    
    
    

def train(net, trainloader, valloader,  optimizer, epochs, gradient_clipping,log_file, device: str):
   

    sparsity, nnz=calculate_sparsity_and_NNZ(net)
    log_file.write(f"Sparsity: {sparsity}, NNZ: {nnz}\n")
    print(f"Starting training for {epochs} epochs")
    """Train the network on the training set"""
    criterion = torch.nn.CrossEntropyLoss()
   
    net.to(device)
    
    
    print_interval = 1
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=100, warmup=100, active=100, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./simulation_logs/resource_usage'),
    #     with_flops=True,
    #     profile_memory=True,
    #     record_shapes=True,
    #     with_stack=True)
    # prof.start()
    for epoch in tqdm(range(epochs)):
        net.train()
        running_loss = 0.0
         # Validation
        if epoch % print_interval == 0:
           
            val_loss, val_accuracy = eval(net, valloader, log_file, device)
            writer.add_scalar(f"{SIMULATION_LOGS_FOLDER}/val_loss", val_loss, epoch)
            writer.add_scalar(f"{SIMULATION_LOGS_FOLDER}/training_loss", running_loss/1000, epoch)
            writer.flush()
            log_file.write(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}\n")
            print(f"\nValidation Loss: {val_loss}, Validation Accuracy: {val_accuracy}\n")
            log_file.flush()
            net.train()
        for i ,(images, labels) in enumerate(tqdm(trainloader)):
            
            # prof.step()
            # if i >= 100 + 100 + 100:
            #     prof.stop()
            images, labels = images.to(device), labels.to(device)
          
            optimizer.zero_grad()
            outputs = net(images)            
            loss = criterion(outputs, labels)
            loss.backward()
            if gradient_clipping is not None:
                nn.utils.clip_grad_norm_(net.parameters(), gradient_clipping)
            optimizer.step()
            running_loss += loss.item()
            if i % 10== 9:    # Print every 100 mini-batches
                log_file.write(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss/100:.3f}\n")
                log_file.flush()
                running_loss = 0.0
            # if i>5:
            #     break
            
      
            
    
    #    # Validation
    #     if epoch % print_interval == 0:
    #         val_loss, val_accuracy = eval(net, valloader, log_file, device)
    #         writer.add_scalar(f"{SIMULATION_LOGS_FOLDER}/val_loss", val_loss, epoch)
    #         writer.add_scalar(f"{SIMULATION_LOGS_FOLDER}/training_loss", running_loss/1000, epoch)
    #         writer.flush()
    #         log_file.write(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}\n")
    #         print(f"\nValidation Loss: {val_loss}, Validation Accuracy: {average_paramsval_accuracy}\n")
    #         log_file.flush()
        
            #print_utilization(device, log_file)

   
                
def eval(net, valloader, log_file, device: str):
    
    print(f"\nStarting validation")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():        
        for data in tqdm(valloader):           
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs.data, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct/total
    return loss, accuracy

def test(net, testloader, log_file, device: str):
    print(f"\nStarting testing")
    """Validate the network on the entire test set"""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0,0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():        
        for data in tqdm(testloader):           
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs.data, labels).item()
            writer.add_scalar(f"{SIMULATION_LOGS_FOLDER}/test_loss", loss)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct/total
    log_file.write(f"Loss: {loss}, Accuracy: {accuracy}\n")
    return loss, accuracy









