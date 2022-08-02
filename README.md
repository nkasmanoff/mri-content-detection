# MRI Content Detection


This repo houses the code used for [Deep Multi-Task Learning for Brain MR Series Classification by Sequence and Orientation](https://link.springer.com/article/10.1007/s00234-022-03023-7) a part of the 2020 MSDSE Summer Research Initiative Project, A practical DL-based tool for medical image specifications identification. 


## Motivation 

Increasingly complex MRI studies and variable series naming conventions reveal limitations of rule-based image routing, especially in health systems with multiple scanners and sites. Accurate methods to identify series based on image content would aid post-processing and PACS viewing. Recent deep/machine learning efforts classify 5–8 basic brain MR sequences. We present an ensemble model combining a convolutional neural network and a random forest classifier to differentiate 25 brain sequences and image orientation.


## Methodology

Series were grouped by descriptions into 25 sequences and 4 orientations. Dataset A, obtained from our institution, was divided into training (16,828 studies; 48,512 series; 112,028 images), validation (4746 studies; 16,612 series; 26,222 images) and test sets (6348 studies; 58,705 series; 3,314,018 images). Dataset B, obtained from a separate hospital, was used for out-of-domain external validation (1252 studies; 2150 series; 234,944 images). We developed an ensemble model combining a 2D convolutional neural network with a custom multi-task learning architecture and random forest classifier trained on DICOM metadata to classify sequence and orientation by series.

## Data 

The data for this project is not publicly available. Please reach out to me at nsk367@nyu.edu to learn more about the data used here, and what steps we can take to create a similar dataset.  

## Results

Confusion Matrix Dataset A           |  Confusion Matrix Dataset B
:-------------------------:|:-------------------------:
![](https://github.com/nkasmanoff/mri-content-detection/blob/main/bin/model_seq_indistribution_cm.png) |  ![](https://github.com/nkasmanoff/mri-content-detection/blob/main/bin/model_seq_oodistribution_cm.png)


Model Architecture           
:-------------------------:
![](https://github.com/nkasmanoff/mri-content-detection/blob/main/bin/autolabelarchitecture.png)



### Out-the-Box Classifier

For the latest / best trained models, please reach out to me at nsk367@nyu.edu

### Emphasis on Flexible Functionality

For the CNN especially, the design of the dataloader and model was built with the intention of eventually incorporating as many relevant MRI metadata attributes as possible into the prediction procedure, with minimal cost to inference time. With that in mind, one possible follow-up to this work is to extend the CNN's functionality to also predict slice thickness, whether or not this MRI uses a Fat Sat pulse, or extending to predict what body part is imaged. 