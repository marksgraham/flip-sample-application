# Hello PyTorch Example Application

Example of using [NVIDIA FLARE](https://nvidia.github.io/NVFlare) to train an image classifier using federated averaging ([FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))) and [PyTorch](https://pytorch.org/) as the deep learning training framework.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvidia.github.io/NVFlare/installation.html) instructions.
Install additional requirements:

```
pip3 install torch
```

### 2. Set up your FL workspace

Follow the [Quickstart](https://nvidia.github.io/NVFlare/quickstart.html) instructions to set up your POC ("proof of concept") workspace.

Ensure you also copy the `src/` directory into the NVFlare examples folder.

```bash
mkdir poc/admin/transfer/<APPLICATION_NAME>
cp -rf src/* poc/admin/transfer/<APPLICATION_NAME>
```

### 3. Run the experiment

Log into the Admin client by entering `admin` for both the username and password.
Then, use these Admin commands to run the experiment:

```bash
set_run_number 1
upload_app <APPLICATION_NAME>
deploy_app <APPLICATION_NAME> all
start_app all
```

### 4. Shut down the server/clients

To shut down the clients and server, run the following Admin commands:
```
shutdown client
shutdown server
```

> **_NOTE:_** For more information about the Admin client, see [here](https://nvidia.github.io/NVFlare/user_guide/admin_commands.html).
