# RADNet
This repository contains the code implementation for the paper "RADet: Refine Feature Pyramid Network and Multi-Layer Attention Network for Arbitrary-Oriented Object Detection of Remote Sensing Images". 
[paper link](https://www.mdpi.com/2072-4292/12/3/389)

# Introduction

Object detection has made significant progress in many real-world scenes. Despite this remarkable progress, the common use case of detection in remote sensing images remains challenging even for leading object detectors, due to the complex background, objects with arbitrary orientation, and large difference in scale of objects. In this paper, we propose a novel rotation detector for remote sensing images, mainly inspired by Mask R-CNN, namely RADet. RADet can obtain the rotation bounding box of objects with shape mask predicted by the mask branch, which is a novel, simple and effective way to get the rotation bounding box of objects. Specifically, a refine feature pyramid network is devised with an improved building block constructing top-down feature maps, to solve the problem of large difference in scales. Meanwhile, the position attention network and the channel attention network are jointly explored by modeling the spatial position dependence between global pixels and highlighting the object feature, for detecting small object surrounded by complex background. Extensive experiments on two remote sensing public datasets, DOTA and NWPUVHR -10, show our method to outperform existing leading object detectors in remote sensing field.

# Usage

This guide offers steps on how to configure your environment, train, and test your model.

## 1. Configuring the Runtime Environment:
### a. Creating a Virtual Environment

To start, create a virtual environment named "mydet" using conda, and activate it:
```shell
conda create -n mydet python=3.6 -y
source activate mydet
```
 
### b. Installing Libraries

Next, install the PyTorch and torchvision libraries:
```shell
conda install pytorch torchvision -c pytorch
```
 
### c. Installing MMDetection

Install the mmdetection library. Other dependencies will install automatically:
```shell
pip install mmcv
```
 
Please ensure that the version installed is consistent with the version noted in the envs.txt file. For any other missing libraries, refer to this file for the correct versions to install.

### d. Compilation

Finally, compile the model:
```shell
python setup.py develop  # or "pip install -v -e ."
```
 
Make sure to delete the "build" directory and .so files in each subfolder under "myDet/mmdet/ops" before you start to compile. Once you see the word "finished", the compilation was successful. If not, delete the "build" folder and .so files and try again.

## 2. Training The Model
### a. Activating the Virtual Environment

Ensure that you activate your virtual environment before training:
```shell
source activate mydet
```
 
### b. Navigating to myDet/tools

Navigate to the "myDet/tools" directory. If you want to train the model for the second innovation point on the Small-DOTA dataset, use the corresponding configuration file, CANet.py:
```shell
CUDA_VISIBLE_DEVICES=0 python train.py --config='../sdota_configs/CANet.py' --gpus=1 
```
 
## 3. Testing The Model
### a. Activating the Virtual Environment

Before testing, make sure you're in your virtual environment:
```shell
source activate mydet
```
 
### b. Modifying HBB_test.py

Navigate to the "myDet" directory and open "HBB_test.py". Modify the following as needed:

The position of the config_file (configuration file)
The location of the checkpoint_file (trained model file)
The img_path (location of the image to be tested)
The txt_dir (if you wish to save the detection result)
The save_path and out_file (if you wish to save the detection image)
c. Running HBB_test.py

Finally, run "HBB_test.py" in the "myDet" directory:
```shell
python HBB_test.py
```

# Citation
If you find this code implementation useful in your research, please consider citing the following paper:


> Li Y, Huang Q, Pei X, Jiao L, Shang R. RADet: Refine Feature Pyramid Network and Multi-Layer Attention Network for Arbitrary-Oriented Object Detection of Remote Sensing Images. Remote Sensing. 2020; 12(3):389. https://doi.org/10.3390/rs12030389
