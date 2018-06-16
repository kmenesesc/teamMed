Description
===========
The following code was developed to analyze deep learning as it is applied to the detection of tumors in MRI scans. The code was developed by Team BioMed from UCSD. 
The repository is composed of implementations of VGG-net and AlexNet as well as image processing tools.



Requirements
============
Install package 'imageio ' a s follow :
$ pip install -- user imageio


Code organization
=================
demo . ipynb -- Run a demo of our code ( reproduce Figure 3 of our report )
train . upyng -- Run the training of our model ( a s described in Section 2)
code / backprop . py -- Module implementing backprop
code / visu . py -- Module for visualizing our dataset
assets / model . dat -- Our model trained a s described in Section 4
