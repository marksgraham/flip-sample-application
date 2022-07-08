# Sample FLIP Application

This repository provides an example application that will be run on FLIP.

Model developers uploading applications to the platform can develop and test their code using this sample application and be confident that it will work once uploaded.

## **How to use this repository**

This repository contains a working sample application, albeit simplified, that represents a running application on FLIP. This application is run using [NVIDIA Flare](https://github.com/NVIDIA/NVFlare) but controlled entirely by FLIP.

1. [Download, install and configure NVIDIA Flare](https://nvflare.readthedocs.io/en/2.1.1/quickstart.html) - Follow this guide to get a basic system running locally.
1. Clone / download this repository.
1. Add the `flip-app` folder from this repository to the `poc/admin/transfer` folder in your NVIDIA Flare install directory.
1. Develop, run and test your application.

## **Development guide**

The `flip-app` folder contains a replica of the application that is run on FLIP. Some modules are stubbed with test data returned. There are two main files that FLIP requires before running any training, `trainer.py` and `validator.py`. Both of these files you will find within `flip-app/custom` and contain a working example application.

These two files are where you should add your own application code. `flip-app/samples` contains `trainer.py` and `validator.py` and do not contain any functionality. These files can be used as a blank template and replace the existing files within `flip-app/custom` once developed and ready to run. 
