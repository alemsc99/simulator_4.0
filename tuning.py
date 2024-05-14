from datetime import datetime
import os
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import torch
from tqdm import tqdm
from utils import define_model
from dataset import prepare_dataset
from model import train 
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

EPOCHS=5



def test_step(dataloader, model, device, test_acc_metric):
    """
    Perform a single test step.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader for the test data.
        model (torch.nn.Module): The model to test.
        device (torch.device): The device to test the model on.
        test_acc_metric (torchmetrics.Accuracy): The accuracy metric for the model.

    Returns:
        The accuracy of the model on the test data.
    """

    for (X, y) in tqdm(dataloader):
        # Move the data to the device.
        X = X.to(device)
        y = y.to(device)

        # Forward pass.
        y_preds = model(X)

        # Calculate the accuracy.
        test_acc_metric.update(y_preds, y)

    return test_acc_metric.compute()

def train_step(dataloader, model, optimizer, criterion, device, train_acc_metric):
    """
    Perform a single training step.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader for the training data.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        criterion (torch.nn.Module): The loss function for the model.
        device (torch.device): The device to train the model on.
        train_acc_metric (torchmetrics.Accuracy): The accuracy metric for the model.

    Returns:
        The accuracy of the model on the training data.
    """

    for (X, y) in tqdm(dataloader):
        # Move the data to the device.
        X = X.to(device)
        y = y.to(device)

        # Forward pass.
        y_preds = model(X)

        # Calculate the loss.
        loss = criterion(y_preds, y)

        # Calculate the accuracy.
        train_acc_metric.update(y_preds, y)

        # Backpropagate the loss.
        loss.backward()

        # Update the parameters.
        optimizer.step()

        # Zero the gradients.
        optimizer.zero_grad()

    return train_acc_metric.compute()

def create_writer(
    experiment_name: str, model_name: str, learning_rate, dropout) -> SummaryWriter:
    """
    Create a SummaryWriter object for logging the training and test results.

    Args:
        experiment_name (str): The name of the experiment.
        model_name (str): The name of the model.
        conv_layers (int): The number of convolutional layers in the model.
        dropout (float): The dropout rate used in the model.
        hidden_units (int): The number of hidden units in the model.

    Returns:
        SummaryWriter: The SummaryWriter object.
    """

    timestamp = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    log_dir = os.path.join(
        "runs",
        timestamp,
        experiment_name,
        model_name,
        
        f"{dropout}",
        f"{learning_rate}"
        
        
    ).replace("\\", "/")
    return SummaryWriter(log_dir=log_dir)


def main():
    trainloader, valloader, testloader, num_classes, input_channels, input_size_x, input_size_y=prepare_dataset('cifar10', 64, 0.1)
    device="cuda" if torch.cuda.is_available() else "cpu"  

    experiment_number = 0

    # hyperparameters to tune
    hparams_config = {
        "dropout": [0.0, 0.25, 0.5],
        "learning_rate": [0.01, 0.001, 0.0001, 0.00001],
    }

    for dropout in hparams_config["dropout"]:
        for lr in hparams_config["learning_rate"]:
            experiment_number += 1
            print(
                f"\nTuning Hyper Parameters || Dropout: {dropout} || Learning rate: {lr} \n"
            )
            model=define_model('ResNet18', num_classes, input_channels, input_size_x).to(device)
            
            # create the optimizer and loss function
            
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
           
            criterion = torch.nn.CrossEntropyLoss()
            
            # create the accuracy metrics
            train_acc_metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=10
            ).to(device)
            test_acc_metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=10
            ).to(device)
            # create the TensorBoard writer
            writer = create_writer(
                experiment_name=f"{experiment_number}",
                model_name="Resnet18",
                learning_rate=lr,
                dropout=dropout,
            )
            
             # train the model
        for epoch in range(EPOCHS):
            train_step(
                trainloader,
                model,
                optimizer,
                criterion,
                device,
                train_acc_metric,
            )
            test_step(testloader, model, device, test_acc_metric)
            writer.add_scalar(
                tag="Training Accuracy",
                scalar_value=train_acc_metric.compute(),
                global_step=epoch,
            )
            writer.add_scalar(
                tag="Test Accuracy",
                scalar_value=test_acc_metric.compute(),
                global_step=epoch,
            )
        # add the hyperparameters and metrics to TensorBoard
        writer.add_hparams(
            {
                "learning_rate": lr,
                "dropout": dropout,
               
            },
            {
                "train_acc": train_acc_metric.compute(),
                "test_acc": test_acc_metric.compute(),
            },
        )
        
        
        
if __name__=="__main__":   
    main()
