from resnet import ResNet18
import torch
from ptflops import get_model_complexity_info
import re
import torch.nn as nn
import torch.nn.utils.prune as prune
import time
import psutil
from utils import measure_latency_cpu_usage, measure_gpu_throughput_and_macs
import tensorflow as tf


device="cuda" if torch.cuda.is_available() else "cpu"      
model=ResNet18(num_classes=10, input_channels=3).to(device)#MODEL DEFINITION


# for name, module in model.named_modules():
#         if isinstance(module, (nn.Conv2d, nn.Linear)):  # Pruning su Conv2d, Linear
#             m=prune.ln_structured(module, 'weight', 0.8,n =float("-inf"), dim=1 )
#             m=prune.remove(m, name='weight')


latency, cpu_usage=measure_latency_cpu_usage(model, device)
print(f"Latency: {latency} ms\n")
print(f"CPU Usage: {cpu_usage}%\n")
throughput, macs=measure_gpu_throughput_and_macs(model, 128, device)
print(f"Throughput: {throughput}\n")
print(f"MACs: {macs / (1024 ** 3)} G\n")
print(f"FLOPs: {(2*macs) / (1024 ** 3)} G\n")
        