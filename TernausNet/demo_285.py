

from pylab import *
import cv2
from dataset import load_image #function to load images
import torch
from utils import variable
from generate_masks import get_model
from torchvision.transforms import ToTensor, Normalize, Compose

#%%
rcParams['figure.figsize'] = 10, 10

#%%

def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

#%%
    
model_path = 'unet16_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet16', problem_type='binary')
#%% 
img_file_name = '3064_img.jpg'
img= load_image(img_file_name)
imshow(img)
#%%
img_transform  = Compose([
    ToTensor(),
    Normalize(mean=[0.1509,0.1509,0.1509], std=[0.0612,0.0612,0.0612])
])
    
#%%

   temp = variable(img_transform(img))

   input_img = torch.unsqueeze(temp, dim=0)

#%%

mask = model(input_img)

    
#%%
 mask_array = mask.data[0].cpu().numpy()[0]
 
 #%%
 imshow(mask_array > 20)
