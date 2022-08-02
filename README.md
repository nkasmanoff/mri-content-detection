# MRI Content Detection


This repo houses the code used for [Deep Multi-Task Learning for Brain MR Series Classification by Sequence and Orientation](https://link.springer.com/article/10.1007/s00234-022-03023-7) a part of the 2020 MSDSE Summer Research Initiative Project, A practical DL-based tool for medical image specifications identification. 

In this work, we use deep learning and random forests to sort MRI studies by sequence and orientation. This is the first work to sort by over 20 unique sequence brain MR images, utilizing multi-task learning in the process. 

However, this framework is flexible in that it could be trained on an arbitrary data set or body part, depending on the base dataset. As such, we have created two procedure tables below, one for instantiating our pre-trained model for direct use, and another for starting deep learning based multi-task routing system from scratch. 

On a typical GPU, this model can predict an image in under 6 milliseconds. This allows it to scale up quite nicely when dealing with a large number of MRI studies which need to be sorted. 

## Methodology

Series were grouped by series descriptions into 24 sequences and 4 orientations. Dataset A, obtained from our institution, was divided into training (20,904 studies; 86,089 series; 235,674 images) and test sets (7,265 studies; 62,223 series; 3,658,450 images). Dataset B, obtained from a separate hospital, was used for out-of-domain external validation (1,252 studies; 2,150 series; 234,944 images). We developed and compared a 2D CNN with custom MTL architecture, random forest classifier (RFC), and ensemble model combining CNN and RFC results.



## Data 

Please reach out to me at nsk367@nyu.edu to learn more about the data used here, and what steps we can take to provide a similar one if interested.  

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