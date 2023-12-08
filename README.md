# pytorch-transformer-distributed

Distributed training of an attention model. Forked from: [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer)

## Instructions for Paperspace

### Machines

Make sure to create everything in the same region. I used `East Coast (NY2)`.

1. Create 1x Private network. Assign both computers to the private network when creating the machines.
2. Create 2x nodes of `P4000x2` (multi-GPU) with `ML-in-a-Box` as operating system
3. Create 1 Network drive (250 GB)

### Setup

Login on each machine and perform the following operations:

1. `sudo apt-get update`
2. `sudo apt-get install net-tools`
3. If you get an error about `seahorse` while installing `net-tools`, do the following:
   1. sudo rm /var/lib/dpkg/info/seahorse.list
   2. sudo apt-get install seahorse --reinstall
4. Get each machine's private IP address using `ifconfig`
5. Add IP and hostname mapping of all the slave nodes on `/etc/hosts` file of the master node
6. Mount the network drive
   1. `sudo apt-get install smbclient`
   2. `sudo apt-get install cifs-utils`
   3. `sudo mkdir /mnt/training-data`
   4. Replace the following values on the command below:
      1. `NETWORD_DRIVE_IP` with the IP address of the network drive
      2. `NETWORK_SHARE_NAME` with the name of the network share
      3. `DRIVE_USERNAME` with the username of the network drive
   5. `sudo mount -t cifs //NETWORD_DRIVE_IP/NETWORK_SHARE_NAME /mnt/training-data -o uid=1000,gid=1000,rw,user,username=NETWORK_DRIVE_USERNAME`
      1. Type the drive's password when prompted
7. `git clone https://github.com/hkproj/pytorch-transformer-distributed`
8. `cd pytorch-transformer-distributed`
9. `pip install -r requirements.txt`
10. Login on Weights & Biases
    1. `wandb login`
    2. Copy the API key from the browser and paste it on the terminal
11. Run the training command from below

### Local training

Run the following command on any machine. Make sure to not run it on both, otherwise they will end up overwriting each other's checkpoints.

`torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:48123 train.py --batch_size 8 --model_folder "/mnt/training-data/weights"`

### Distributed training

Run the following command on each machine (replace `IP_ADDR_MASTER_NODE` with the IP address of the master node):

`torchrun --nproc_per_node=2 --nnodes=2 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=IP_ADDR_MASTER_NODE:48123 train.py --batch_size 8 --model_folder "/mnt/training-data/weights"`

### Monitoring

Login to Weights & Biases to monitor the training progress: https://app.wandb.ai/
