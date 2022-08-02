import os
import torch
import pandas as pd
import numpy as np
import sys
import pydicom as dicom
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import datasets, transforms
from ..helpers.utils import squarify, str2vec, clipped_zoom
from ..src.config import root
from scipy import ndimage


#  Define Dataset

class make_Dataset(Dataset):
    """
        Data loader class. Reads in images from the batch dir, and obtains files or items based on whatever mode it is.
        So a pre req is to run read all mris to recover a master csv of all mri files, and what mode they are.
    """
    def __init__(self,
        mode = 'train',meta_data_dir = root,augment=True,batch='merged',seed = 42):

        self.mode = mode
        self.augment = augment

        if batch == 1:
            self.brain_df = pd.read_csv(meta_data_dir + 'batch_1_brain_metadata.csv',dtype=str) 
        elif batch == 2:
            self.brain_df = pd.read_csv(meta_data_dir + 'batch_2_brain_metadata.csv',dtype=str) 
        elif batch == 3:
            self.brain_df = pd.read_csv(meta_data_dir + 'batch_3_brain_metadata.csv',dtype=str) 
        elif batch == 4:
            self.brain_df = pd.read_csv(meta_data_dir + 'batch_4_brain_metadata.csv',dtype=str) 
        elif batch == 'merged':
            self.brain_df = pd.read_csv(meta_data_dir + 'batch_merged_brain_metadata.csv',dtype=str) 
        elif batch == 'wuh':
            self.brain_df = pd.read_csv(meta_data_dir + 'batch_wuh_brain_metadata.csv',dtype=str) 
        else:
            raise ValueError("Invalid data batch selected.")

        if batch != 'wuh':
            # don't split into train / valid / test when using the out of distribution MRIs.
            self.brain_df = self.brain_df.loc[self.brain_df['Mode'] == mode]

        self.contrast_codes = pd.read_csv(meta_data_dir + 'contrasts_OHE_codes.csv',index_col=0)
        self.orientation_codes = pd.read_csv(meta_data_dir + 'orientation_OHE_codes.csv',index_col=0)
        self.brain_df = self.brain_df.sample(frac=1,random_state = seed)

    def __len__(self):
        return len(self.brain_df)

    def __getitem__(self,idx):
        """
        Return an item based on the given idx. 
        Also performs random augmentations when self.augment is set to True.
        """

        fixed_length = 256
        fixed_width = 256
        dim = (fixed_length, fixed_width)
        ds_path = self.brain_df['DCM_Path'].values[idx] + self.brain_df['Study_Num'].values[idx] + '/' + self.brain_df['Slice_Num'].values[idx]+ '.dcm'

        ds = dicom.dcmread(ds_path) #ds = dicom slice

        #normalize input. Only use this image
        img = ds.pixel_array.astype(np.float32)

        try:
            img = squarify(img)
        except:
            # There appear to be some cases where there are 3-D pixel arrays, i.e 192x256x256.
            # This is an entire series compressed into a single dcm file, so
            # the fix we implement here is to take the mean of the 3-D array.
            img = squarify(img.mean(axis=0)) #mean val of all together

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


        contrast = str2vec(self.brain_df['Contrast'].values[idx],self.contrast_codes)
        orientation = str2vec(self.brain_df['Orientation'].values[idx],self.orientation_codes)

        # when the slize is entirely black, the model should predict it's class as OTHER and not labeled for contrast & orientation.
        if img.std().item() < 1e-4:
            contrast = str2vec('OTHER_',self.contrast_codes)
            orientation = str2vec('UNLABELED',self.orientation_codes)

        if self.augment:
            rot = np.random.uniform(low=-10,high=10)
            zoom_factor = np.random.uniform(low = .95,high = 1.05)
            shift_x = np.random.uniform(low=-20,high=20)
            shift_y = np.random.uniform(low=-20,high=20)
            img = ndimage.rotate(img, rot, reshape=False)
            img = clipped_zoom(img=img, zoom_factor=zoom_factor)
            img = ndimage.shift(img,shift=(shift_x,shift_y))

        min_img = np.min(img)
        max_img = np.max(img)
        img = (img - min_img)/(max_img - min_img + 1e-4)


        img = torch.tensor(img, dtype=torch.float).resize(1,fixed_length,fixed_width)
        slice_thickness = torch.tensor(slice_thickness,dtype=torch.float)
        contrast = torch.from_numpy(contrast)
        orientation = torch.from_numpy(orientation)


        return img, contrast, orientation, ds.PatientID, ds.SeriesDescription  # return labels & patient metadata 
 


# This function creates the different datasets

def create_datasets(batch_size,weighted_trainer,batch,seed):
    train_Dataset = make_Dataset(mode='train', batch=batch, seed = seed)

    if weighted_trainer == True:
        if os.path.exists(root + "train_sample_weights.npy"):
            train_samples_weight = np.load(root + "train_sample_weights.npy")
        else:
            train_Dataset.brain_df['Class'] = train_Dataset.brain_df['Contrast'].apply(lambda z: str2vec(z, train_Dataset.contrast_codes).argmax())
            class_sample_counts = train_Dataset.brain_df['Class'].value_counts().sort_index().values
            num_classes = train_Dataset.contrast_codes.shape[1]
            class_weights = 1./torch.Tensor(class_sample_counts)

            train_targets = train_Dataset.brain_df['Class'].values
            train_samples_weight = [class_weights[class_id] for class_id in train_targets]
            train_samples_weight = np.array(train_samples_weight)
            np.save(root + 'train_sample_weights.npy',train_samples_weight)


    else:
        train_loader  = DataLoader(dataset=train_Dataset, batch_size=batch_size,
                               shuffle=True)


    valid_Dataset = make_Dataset(mode='valid', augment=False,batch=batch, seed = seed)
    valid_loader  = DataLoader(dataset=valid_Dataset, batch_size=batch_size,
                               shuffle=False)

    test_Dataset  = make_Dataset(mode='test', augment=False,batch=batch, seed = seed)
    test_loader   = DataLoader(dataset=test_Dataset,  batch_size=batch_size,
                               shuffle=False)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = create_datasets(batch_size=256)
    print("Works!")
