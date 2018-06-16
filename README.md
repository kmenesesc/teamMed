Description
===========
The following code was developed to analyze deep learning as it is applied to the detection of tumors in MRI scans. 
The code was developed by Team BioMed from UCSD. 
The repository is composed of implementations of VGG-net, AlexNet and UNet as well as image processing tools. Please note that various weights have been omitted due to data size. 



Requirements
============

Before running notebooks in the folder AlexNet, download the pretrained weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ titled "bvlc_alexnet.npy" and place them inside the AlexNet Folder. This is only required if you will be running the finetuning of AlexNet.

If running AlexNet notebooks, place the data set inside the AlexNet folder. If running VGGNet notebooks, place the data set inside the VGGNet folder. 

TensorFlow version 1.8.0 is required to run the AlexNet Notebooks. 

Pytorch is required to run the VGGNet Notebooks. 

Other package dependencies: 


-h5py

-numpy

-os

-cv2

-Pillow

-matplotlib

-scipy

-TorchVision

-opencv-python

-tqdm

Code organization
=================


The repository contains three main subfolders "AlexNet," "VGGNet,"  and "TernausNet." In each subfolder, you will find code related to the each corresponding architecture.  On this layer, you will 
also find a python notebook, 

The AlexNet Folder contains the following items: 

"Plot.ipynb"- Replicates AlexNet Plots

PreTrainedValidation and 
run_tumors_2018-06-15-06_11_23.263597-tag-Accuracy_Validation  - Data corresponding to the best validation given by AlexNet on the Data set.

AlexNet.m - An implementation of AlexNet in MatLab. 

alexnet.py -  An implementation of AlexNet in Python using TensorFlow.

blvc_alexnet - Initial Weighs obtained from a pre-trained version of AlexNet on a different data set of pictures. 

caffe_classes.py - The labels utilized for the data on the pre-trained version of AlexNet. 

datagenerator.py -  A helper class used to input images into TensorFlow. 

finetune.py - A finetuning script for the python version of AlexNet. 

OriginalAlexNet.ipynb - Notebook utilized to train AlexNet from scratch. 

PreTrainedAlexNet.ipynb - Notebook utilized to finetune AlexNet. 

TrainingData - The output data during the loss evaluation of an epoch of AlexNet. 



Next, the VGGNet Folder contains the following items:

VGGnet_brainTum-2.ipynb - Python Notebook that implements the training of VGG along with a demo of plots. 

CreateDataFolders2.ipynb - A tool to generate data folders to then feed into VGGNet.


Lastly, the folder TernausNet includes the following: 

Supporting Functions - Helping functions for image processing.

unet16_binary_20- includes the pretrained weights for the UNet implementation. -This has been omitted due to its large size-

demo_285.py- A demonstration for using UNet to compute masks on the MRI scans.


=========

