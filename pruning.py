'''
PyTorch goes beyond simply setting pruned parameters to zero. PyTorch copies the parameter <param> 
into a parameter called <param>_original and creates a buffer that stores the pruning mask <param>_mask.
It also creates a module-level forward_pre_hook (a callback that is invoked before a forward pass) 
that applies the pruning mask to the original weight.
Printing <param> will print the parameter with the applied mask, but listing it via <module>.parameters() 
or <module>.named_parameters() will show the original, unpruned parameter. Yet, it comes at cost of some memory overhead.
'''

import torch 
import torch.nn as nn
import torch.nn.utils.prune as prune
import os




model_saving_path="./models/"


def retrieve_model(folder):
    # List all the files in the directory
    files_list = os.listdir(folder)
    # Filter only the files 
    files_list = [file for file in files_list if file.endswith(".pth")]
    # Order the files by modification date
    files_list.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
    # Restituisci il percorso del primo file (il più recente) nella lista
    return os.path.join(folder, files_list[0]) if files_list else None

def retrieve_file(folder, file_name):
    # Get the list of all files in the folder
    files_in_folder = os.listdir(folder)
    # Search for the file with the specified name
    for file in files_in_folder:
        if file == file_name:
            return os.path.join(folder, file)
    # If the file is not found, return None
    return None


'''
This function applies UNSTRUCTURED GLOBAL pruning to the model.
'''
def global_unstructured_pruning(model, pruning_rate: float, isRandom: bool):    
    modules_list=filter(lambda x: isinstance(x[1], (nn.Conv2d, nn.Linear, nn.BatchNorm2d)), model.named_modules())
    modules_list = map(lambda x: (x[1], 'weight'), modules_list)
    modules_list=tuple(modules_list)
    
    if isRandom:
        prune.global_unstructured(modules_list, pruning_method=prune.RandomUnstructured, amount=pruning_rate)
    else:
        prune.global_unstructured(modules_list, pruning_method=prune.L1Unstructured, amount=pruning_rate)
        
    for module in modules_list:
        prune.remove(module[0], module[1])
        
    
    
'''
This function applies STRUCTURED LOCAL pruning to the model.
'''    
def local_structured_pruning(model, pruning_rate: float, isRandom: bool):
   
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):  # Pruning su Conv2d, Linear
            if isRandom:
                m=prune.random_structured(module, 'weight', pruning_rate, dim=1) #TO TEST
            else:
                m=prune.ln_structured(module, 'weight', pruning_rate,n =float("-inf"), dim=1 )
                
                
            m=prune.remove(m, name='weight')
            

'''
This function applies UNSTRUCTURED LOCAL pruning to the model.
'''
            
def local_unstructured_pruning(model, pruning_rate:float, isRandom:bool):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):  # Pruning su Conv2d, Linear
            if isRandom:
                m=prune.random_unstructured(module, 'weight', pruning_rate) #TO TEST
            else:
                m=prune.l1_unstructured(module, 'weight', pruning_rate) # TO TEST
                
                
            m=prune.remove(m, name='weight')
            
            
def calculate_sparsity_and_NNZ(model):
    total_params = 0
    pruned_params = 0
    non_zero_params =0

    # Iterate over all the modules in the model
    for name, module in model.named_modules():
        
        # Check if the module is prunable
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            total_params += module.weight.nelement()
            pruned_params += module.weight.masked_select(module.weight == 0).nelement()
            non_zero_params += torch.sum(module.weight != 0).item()

            # If BatchNorm2d module has bias parameter, count its non-zero elements too
            if isinstance(module, nn.BatchNorm2d) and module.bias is not None:
                non_zero_params += torch.sum(module.bias != 0).item()

    sparsity = pruned_params / total_params if total_params > 0 else 0.0
    
    return sparsity, non_zero_params
    
