# Sample FLIP Application

The purpose of this repository is to provide a sample application that can be developed and tested locally before being uploaded onto FLIP.
## **How to use this repository**

This repository contains a simplified sample application that replicates a running application on FLIP. This application runs on [NVIDIA Flare](https://github.com/NVIDIA/NVFlare).

Download or clone this repository and use the `./flip-app` directory as a sample application to run in NVIDIA Flare.

## **Development guide**

The `./flip-app` directory contains a replica of an application that can be run on FLIP. Some modules are stubbed with only a return type set. There are two main files that FLIP requires before running any training - `trainer.py` and `validator.py`. Both of these files you will find within `flip-app/custom` and contain a working example application that can be used as a starting point.

This example uses [NVIDIA FLARE](https://nvidia.github.io/NVFlare) to train an image classifier using federated averaging ([FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))) and [PyTorch](https://pytorch.org/) as the deep learning training framework.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

These two files are where you should add your own application code. The `./samples` directory contains empty templates of both `trainer.py` and `validator.py`.

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/2.0/installation.html) instructions.

> ⚠️ Please ensure you install version `2.0.16`

> Requires specific protobuf version [NVFlare GitHub Issue](https://github.com/NVIDIA/NVFlare/issues/608)

Install additional requirements:

```bash
pip install torch
pip install protobuf==3.20
pip install pandas
```

### 2. Set up your FL workspace

Follow the [Quickstart](https://nvflare.readthedocs.io/en/2.0/quickstart.html) instructions to set up your POC ("proof of concept") workspace.

Ensure you also copy this (`./flip-app`) directory into the NVFlare examples folder.

```bash
mkdir poc/admin/transfer/<APPLICATION_NAME>
cp -rf flip-app/* poc/admin/transfer/<APPLICATION_NAME>
```

### 3. Run the experiment

Log into the Admin client by entering `admin` for both the username and password.
Then, use these Admin commands to run the experiment:

```bash
upload_app <APPLICATION_NAME>
set_run_number 1
deploy_app <APPLICATION_NAME> all
start_app all
```

### 4. Shut down the server/clients

To shut down the clients and server, run the following Admin commands:
```
shutdown client
shutdown server
```

> **_NOTE:_** For more information about the Admin client, see [here](https://nvflare.readthedocs.io/en/2.0/user_guide/admin_commands.html).

## **FLIP methods**
The following methods are available to be used in training, located in `flip.py`:

- `get_dataframe(self, project_id: str, query: str) -> DataFrame`
   This retrieves data in the form of a Dataframe containing, at the minimum, accession IDs.
   The method takes in the project ID and the project query as parameters. These values are
   already passed in as parameters to the trainer to be used.

- `get_data(self, project_id: str, net_id: str) -> Path`
   This method is for internal use only, and is not be called by the trainer.

- `update_status(self, model_id: str, new_model_status: ModelStatus)`
   This method is for internal use only, and is not be called by the trainer.

### Import FLIP and call methods
- Import the module: `from flip import FLIP`
- Make an instance of the class: `flip = FLIP()`
- Use the instance to call one of the methods: `dataframe = flip.get_dataframe(project_id, query)`

This will allow successful calls to any the methods in `flip.py`.