# <b>Profiling EfficientNets on Multi-GPU Systems</b>
## Overview
EfficientNets are a family of image classification models that achieve state-of-the-art accuracy, yet are an order of magnitude smaller and faster than previous models. They are developed based on AutoML and Compound Scaling. <br/>
This project is a comprehensive evaluation and profiling of the training process of EfficientNet B0, B3, and B5 models across multiple GPU configurations. The models are trained on two distinct datasets: Cifar100 and Food101.

The primary objectives of this project are as follows:
* <b>Computational Resource Utilization</b>: Assess the usage of computational resources during the training process. This includes GPU memory consumption, CPU usage, and network I/O. The goal is to maximize resource utilization while maintaining model performance.

* <b>Bottleneck Identification</b>: Identify any potential bottlenecks in the training process. This could be related to data loading, model architecture, or hardware constraints. Once identified, these bottlenecks can be addressed to improve training efficiency.

* <b>Model Efficiency Optimization</b>: Optimize the efficiency of the EfficientNet models across different GPU configurations. This involves tuning hyperparameters, adjusting batch sizes, and potentially modifying the model architecture to better suit the hardware setup.

By achieving these objectives, the project aims to enhance the scalability and efficiency of training EfficientNet models on multi-GPU systems.


## Requirements
- Python: 3.8 or higher
- PyTorch: 1.10.0
- torchvision: 0.11.1(for CUDA 11.6)

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

2. Upload the GitHub project files through scp on NYU HPC

3. Modify as per the requirements for the `sbatch` files, Once done, this is how you would run the python project files on terminal
   
   > Make sure you run this in the `scratch` directory

```shell
module purge
module load python/intel/3.8.6
python download.py # To download food101 dataset
```

4. Execute sbatch files to send it to slurm scheduler for running it on HPC.
To train the model on different architectures use the following arguments:
- Assign arguments `--arch` to `efficientnet-b0`, `efficientnet-b3` or `efficientnet-b5` depending on the architecture to train
- Assign arguments `--epochs` flag to set up the number of epochs to run to train and profile
- Assign arguments `--profile` flag to `True` or `False` whether profiling is to be performed or not(default = False)
- Adjust the `--dataset` flag to either `cifar` for CIFAR100 dataset or `food` for FOOD101 dataset


A sample sbatch file has been provided to perform either training or profiling:

```shell
sbatch job.sh
```

## DataLoading + Training Time Results
<p float="left">
  <img src="https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_1_GPUs_Cifar-100.png" width="400" />
  <img src="https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_1_GPUs_Food101.png" width="400" />
   
</p>
<p float="left">
  <img src="https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_2_GPUs_Cifar-100.png" width="400" />
  <img src="https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_2_GPUs_Food101.png" width="400" />
   
</p>
<p float="left">
  <img src="https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_4_GPUs_Cifar-100.png" width="400" />
  <img src="https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_4_GPUs_Food101.png" width="400" />
   
</p>
<p float="left">
  <img src="https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Cifar-100%20Top-1%20Accuracy.png" width="400" />
  <img src="https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Food101-Top1%20Accuracy.png" width="400" />
   
</p>
<!-- **![Image1](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_1_GPUs_Cifar-100.png)**
**![Image2](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_1_GPUs_Food101.png)**
**![](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_2_GPUs_Cifar-100.png)**
**![](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_2_GPUs_Food101.png)**
**![](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_4_GPUs_Cifar-100.png)**
**![](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Batch_Time_4_GPUs_Food101.png)**
**![](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Cifar-100%20Top-1%20Accuracy.png)**
**![](https://github.com/Pratham-mehta/OptimizedEfficientNets/blob/main/Result%20Images/Food101-Top1%20Accuracy.png)** -->

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
