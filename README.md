Description
===========
The following code was developed to analyze deep learning as it is applied to the detection of tumors in MRI scans. 
The code was developed by Team BioMed from UCSD. 
The repository is composed of implementations of VGG-net and AlexNet as well as image processing tools.



Requirements
============
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


Code organization
=================
The repository contains two main subfolders "AlexNet" and "VGGNet." In each subfolder, you will find code related to the each corresponding architecture. 


The AlexNet Folder contains the following items: 

AlexNet.m - An implementation of AlexNet in MatLab. 

alexnet.py -  An implementation of AlexNet in Python using TensorFlow

blvc_alexnet - Initial Weighs obtained from a pre-trained version of AlexNet on a different data set of pictures. 

caffe_classes.py - The labels utilized for the data on the pre-trained version of AlexNet. 

datagenerator.py -  A helper class used to input images into TensorFlow. 

finetune.py - A finetuning script for the python version of AlexNet. 

OriginalAlexNet.ipynb - Notebook utilized to train AlexNet from scratch. 

PreTrainedAlexNet.ipynb - Notebook utilized to finetune AlexNet. 

TrainingData - The output data during the loss evaluation of an epoch of AlexNet. 



VGGNet Folder contains the following items:

VGGnet_brainTum-2.ipynb - Python Notebook that implements the training of VGG along with a demo of plots. 

CreateDataFolders2.ipynb - A tool to generate data folders to then feed into VGGNet.