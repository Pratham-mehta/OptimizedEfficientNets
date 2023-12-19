# Profiling Efficientnet on Cifar100 and Food101

This project involves evaluation and profiling of the training of EfficientNet B0, B3 and B5 models on 1,2 and 4 GPUs using the Cifar100 and Food101 dataset. The aim is to understand the computational resource utilization, identify bottlenecks, and optimize model efficiency across different GPU configurations.

## Overview

EfficientNet models are known for their efficiency and scalability. By profiling these models on different GPU setups, we aim to provide insights into how they scale with increased computational resources and identify the key factors that influence training time and performance.

## Requirements

- Python 3.8.6
- PyTorch
- torchvision
- CUDA-compatible GPUs (if running on GPU)

## Installation

To set up your environment to run the code, follow these steps:

```bash
git clone https://github.com/Pratham-mehta/OptimizedEfficientNets.git
cd OptimizedEfficientNets
pip install -r requirements.txt
```

## Usage

To start profiling and training of the EfficientNet model, run:

1. Connect to NYU HPC using [Open on demand](https://ood-2.hpc.nyu.edu:5554/auth/ldap?req=wlclz6y2ppvz4r5h5iszxi7ht)  or through `ssh` method 
   
   > Make sure you are connected on NYU VPN through cisco Anyconnect

2. Upload the github project files through scp on NYU HPC

3. Modify as per the requirements for the `sbatch` files, Once done, this is how you would run the python project files on terminal
   
   > Make sure you run this in the `scratch` directory

```shell
module purge
module load python/intel/3.8.6
python download.py # To download food101 dataset
```

4. Execute sbatch files to send it to slurm scheduler for running it on HPC. 

```shell
sbatch job1.sh
```

To perform profiling of the models, run this another sbatch file:

```shell
sbatch job.sh
```

- Replace `'efficientnet_b0'` with `'efficientnet_b3'` to train & profile the B3 model. Likewise for b5 model as well.

- Adjust the `--arch` flag according to the model that you want to run (default is B0 model). 

- Adjust the `--epochs` flag for training & profiling the model. 

- Adjust the `--dataset` flag for the dataset to be used for the model like food101 or cifar100 for this project.
## DataLoading + Training Time Results
**![Image1](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_1_GPUs_Cifar-100.png)![Image2](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_1_GPUs_Food101.png)**
**![](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_2_GPUs_Cifar-100.png)**
**![](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_2_GPUs_Food101.png)**
**![](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_4_GPUs_Cifar-100.png)**
**![](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_4_GPUs_Food101.png)**
**![](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Cifar-100%20Top-1%20Accuracy.png)**
**![](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Food101-Top1%20Accuracy.png)**
## Profiling Results

The profiling results are shown below:

- **Profiling Results on B0 using 1 GPU:**

**![](https://lh7-us.googleusercontent.com/o2WLQkt_7IAHIUOQ4famPhVyOTKyq9M6B82Eh8Vs9Kg8QzY030hT32Aune8cgfPqx1G_Y-CXw1dxzRTkkwR2emc3mbrg-kX3g-M3fSvo0blDs_L9dLTaFx-Je8Wo1ga7p8sT2kYLSxEA3fGFJrRvscfIMQ=s2048)**

- **Profiling Results on B3 using 1 GPU:**

**![](https://lh7-us.googleusercontent.com/mUXNmnkDw34mJYKlKyFWcJ275eif8tFONKOtMt3jKO3hi7W70cwAB5qn56-BxvHJZ3-BpSKKiRSH_g98z6Qcs7b0FWI-SoTE5gA05H7jH8e_LZuSeDRI96A9I2uaqv5cvGdaZoCwIrpEz1O_KMPgHqHq9g=s2048)**

- **Profiling Results on B0 using 2 GPUs:**

**![](https://lh7-us.googleusercontent.com/wzKDkce4UWXm0zwdqU09D6rpNYtkhQaxVT8GZO7rYjWAMOBvYTJdytl593D4s05Po-kNhqggsds7UqLLZrCgbvGxpv72lUlHXENShkIYgQXLbpTVOoKxMZ4l366ZH0rK819Eekx8zEfPhzwQYx5wX3-FgQ=s2048)**

- **Profiling Results on B3 using 2 GPUs:**

**![](https://lh7-us.googleusercontent.com/0T2J_Co7DE7WLInTKDW_SDLUO-k9TY5h5Ybq5mye4xdT7F0BkIqF23k-uxxRcI-66Y1eXftvEFCCtCeylNC-tNt4Oi_2YTxu3cEfULA7H3aCM6fY8W1hXoi6xfVu5tEvgaVV1NsIwrVXnl6HOEgkQMszeg=s2048)**

- **Profiling Results on B0 using 4 GPUs:**

**![](https://lh7-us.googleusercontent.com/UVWUprGIVIYccPnrJ4r4sZJtmz3j4Im8pi9hqlDtqH14cYJbcuKXaoeslZp51Sip0A3jTea6W377Ia50H6QoigFOTQV3rz2qcPQeDBfZPNajY2ycxsRs0buEamvmbXL7ieiBYXhkXvCXiqo5OtANOZlvVw=s2048)**

- **Profiling Results on B3 using 4 GPUs:**

**![](https://lh7-us.googleusercontent.com/BxoQfOCZF4IiwdnEILicsdDsN0-gGO1vP5WygCW46DYeleGrbV6oRVucnP3rktdAvISQL4VFwqOEwLHXhsjfhz1yUWHk7gHOAwnbhT1Fox6KMNusxLm7jUEAXyk37hsYy_JLHDnDksjOMLp6slPR8iwpyg=s2048)**

## Observations

- EfficientNet B3 requires more computational resources than B0, reflected in higher CPU and CUDA times.
- Data loading time remains a significant factor in training time and does not decrease proportionally with more GPUs.
- Training time per epoch generally decreases with more GPUs, but the rate of decrease is less with more GPUs due to overheads.

## Conclusions

- EfficientNet B3 is more computationally demanding than B0.
- Scaling from 1 to 2 GPUs shows significant improvement in training time, but further increases in GPU count offer diminishing returns.
- Optimization efforts should consider data loading and distributed training strategies to further reduce training time.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

 Pratham Mehta - pm3483@nyu.edu

Somik Dhar - sd5023@nyu.edu

Project Link: https://github.com/Pratham-mehta/OptimizedEfficientNets
