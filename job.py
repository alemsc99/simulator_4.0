'''
This class defines a job and its attributes:
- id: different ids execute different python scripts
- generation_time: to evaluate the computation time of each job
- dataset: the dataset to be used (mnist, cifar10)
- model: the model to be trained
- network: the network to be used
- task: the task to be performed (Training, Testing, Evaluation)
- model_compression_technique: the model compression technique to be used
- name: the name of the job
'''
class Task:
    def __init__(self, name:str, isGlobal=None, isRandom=None, isStructured=None, pruning_rate=None, pre_trained=None) -> None:
        if name in {"Training", "Testing", "Pruning", "LSPruning"}:
                self.name=name
        else:
            raise ValueError("task must be either Training, Testing, Pruning or LSPruning")
        self.isGlobal=isGlobal
        self.isRandom=isRandom
        self.isStructured=isStructured
        if pruning_rate is not None and pruning_rate!='None':
            self.pruning_rate=float(pruning_rate)
        else:
            self.pruning_rate=0.0
        self.pre_trained=pre_trained

class Job:
    def __init__(self, id, generation_time, dataset=None, num_epochs=None,task=Task) -> None:
        self.id=id
        self.generation_time=generation_time
        if dataset in {"mnist", "cifar10"}:
            self.dataset=dataset
        else:
            raise ValueError("Dataset must be either mnist or cifar10")
        
        if num_epochs is not None and num_epochs!='None':
            self.num_epochs=num_epochs
        else:
            self.num_epochs=0
       
        self.task=task
        
            
            



    