"""
Utility functions to be used by the MRI data pre-processing. 
"""
import numpy as np
import cv2
import re
import pandas as pd



        

def Intersection(a, b): 
    """Compute the intersection tween two lists. """
    return list(set(a) & set(b))


def assign_split(patient_id,train_patients,valid_patients,test_patients):
    """
    Assigns a patient ID to a split, based on the existing splits.
    """
    if patient_id in train_patients:
        return 'train'
    elif patient_id in valid_patients:
        return 'valid'
    elif patient_id in test_patients:
        return 'test'
    else:
        return None
    


def evenly_subsample(brain_df,max_slices=2000):
    """
    From the brain metadata csv, while there is already a proportionate split in target classes in train-valid-test, this is 
    still way too many slices to train with. 
    Instead, now I'm going to iterate over the training set, and reduce thr # of classes by a large amount. 

    For validation I want to retain the same distribution, but on a reduced # of samples. To do this, I am going to uniformly sample
    from that set. 
    """

    train = brain_df.loc[brain_df['Mode'] == 'train']
    valid_test = brain_df.loc[brain_df['Mode'] != 'train']
    train_subsampled = pd.DataFrame()
    for _,g in train.groupby(["Contrast","Orientation"]):
        if g.shape[0] > max_slices:
            g = g.sample(n=max_slices)

        #otherwise use the same size. 
        train_subsampled = train_subsampled.append(g)

    valid_test = valid_test.sample(n = int(round(train_subsampled.shape[0] /  2)))
    #basically reduce to half as large as train. This is subsampling, but retains original distribution which is 
    # what we want to capture.  

    brain_df_subsampled = pd.concat([train_subsampled,valid_test],axis=0)
    return brain_df_subsampled



def fatsat_flag(z):
    z = z.lower().replace('fse','') #removes fse as a false True for this flag. 
    if "fs" in z:
        return 1.

    elif "fsat" in z:
        return 1.

    elif "fat sat" in z:
        return 1.

    else:
        return 0. 


def contrast_flavor(z,contrasts):
    """
    Using the pre-determined types of contrasts, classify a BRAIN series description belonging to one of these flavors for 
    downstream analysis + classification. 
    
    """
    return contrasts.where(contrasts== z).dropna(axis=1,how='all').columns[0]

    #once these are the actual names, it is ready. 

def str2vec(z,OHE_df):
    ### Transform the string name of contrast / imaging orientation to a one hot vector for data loader.
    return OHE_df.iloc[OHE_df.index== z].values[0]


def vec2str(z,OHE_df):
    ### Transform this vector back to a string. 

    return OHE_df[str(z.argmax())].idxmax()

def pre_post(z):
    ### Say if this value is pre
        # now do pre/post processing
        
    if z['Flavor'] in ['T1_2D','T1_3D_MPRAGE','VENOVIBE'
    ,'T1_2D_IAC','VIBE']:
        if 'post' in z['Series_Description'].lower() or 'c+' in z['Series_Description'].lower() :
            return 'post'
        else: 
            return 'pre'
    else:
        return ''
    
    #Once I get clarification I do this for not just t1 series, or I do for just that, done. 
    
def get_orientation(z):
    """
    Given series description, return the imaging orientation. 
    """
    if 'ax' in z.lower():
        return 'AXIAL'
    elif 'sag' in z.lower():
        return 'SAGITTAL'    
    elif 'cor' in z.lower():
        return 'CORONAL'
    else: 
        return 'UNLABELED'

def collate_fn(batch):
    # For the data loader to print all the info 
    return tuple(zip(*batch))

def squarify(M):
    """
    Squarify a matrix.
    """
    
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(M,padding,mode='minimum')



def gaussian_noise(image,var):
    row,col = image.shape
    mean = 0
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy


def clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)

    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width

    return result



def stringlisttoarr(z):
   # [-+]?\d*\.\d+|\d+[e]
    try:
        # A sample regular expression to find digits.  
        #
        match = re.findall(r'[+-]?(?:\d+\.\d*|\.\d+|\d+(?=[eE]))(?:[eE][+-]?\d+)?', z)  
        return np.array([float(m) for m in match])
    except:
        return np.array([-1,-1,-1,-1,-1,-1])


## Following that initial save, now clean into a data-sciencey format. This takes a looooong time. 
def populate(z,attribute):
    ## Fill in the hot encoded space based on the list present in that attribute's column. 
    
    # Converting string to list 
    ini_list = z[attribute]
    ini_list = ini_list.replace("'","")
    res = ini_list.strip('][').split(', ')

    for i in res: 
        z[attribute+"_"+i] = 1     
    return z



def pixel_spacing(z):
    try:
        ini_list = z['PixelSpacing']
        res = ini_list.strip('][').split(', ')
        z['PixelSpacing_0'] = float(res[0])
        z['PixelSpacing_1'] = float(res[1])
        return z
    except:
        z['PixelSpacing_0'] = np.nan
        z['PixelSpacing_1'] = np.nan
        return z


