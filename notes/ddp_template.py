## How to convert any PyTorch project into a DistributedDataParallel project

# Distributed training
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Dummy variables to make Pylance happy :D
train_dataset = None
local_rank = -1
global_rank = -1
num_epochs = 100
step_number = 0
last_step = False

class MyModel:
    pass

def initialize_services():
    pass

def collect_statistics():
    pass

def train():
    if global_rank == 0:
        initialize_services() # W&B, etc.

    data_loader = DataLoader(train_dataset, shuffle=False, sampler=DistributedSampler(train_dataset, shuffle=True))
    model = MyModel()
    if os.path.exists('latest_checkpoint.pth'): # Load latest checkpoint
        # Also load optimizer state and other variables needed to restore the training state
        model.load_state_dict(torch.load('latest_checkpoint.pth'))

    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-4, eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for data, labels in data_loader:
            if (step_number + 1) % 100 != 0 and not last_step: # Accumulate gradients for 100 steps
                with model.no_sync(): # Disable gradient synchronization
                    loss = loss_fn(model(data), labels) # Forward step
                    loss.backward() # Backward step + gradient ACCUMULATION
            else:
                loss = loss_fn(model(data), labels) # Forward step
                loss.backward() # Backward step + gradient SYNCHRONIZATION
                optimizer.step() # Update weights
                optimizer.zero_grad() # Reset gradients to zero

            if global_rank == 0:
                collect_statistics() # W&B, etc.

        if global_rank == 0: # Only save on rank 0
            # Also save the optimizer state and other variables needed to restore the training state
            torch.save(model.state_dict(), 'latest_checkpoint.pth')
        

if __name__ == '__main__':
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])

    init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank) # Set the device to local rank

    train()
    
    destroy_process_group()

