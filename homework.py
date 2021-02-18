#!/usr/bin/env python
# coding: utf-8

# In[21]:


import torch, torchvision
from PIL import Image
from urllib.request import urlopen
import matplotlib.pyplot as plt
import numpy as np


# # Part 1

# In[115]:


img = Image.open(urlopen("https://upload.wikimedia.org/wikipedia/en/5/5f/Original_Doge_meme.jpg"))
img


# In[116]:


# load model
resnet = torchvision.models.resnet101(pretrained=True)

# set network to evaluation mode
resnet.eval()

transform = torchvision.transforms.Compose([          
 torchvision.transforms.CenterCrop(224),               
 torchvision.transforms.ToTensor(),                     
 torchvision.transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                            
 std=[0.229, 0.224, 0.225]                             
 )])


# In[117]:


img_t = transform(img)
print(img_t.shape)
plt.imshow(torch.einsum("chw -> hwc", img_t))


# In[118]:


batch_t = torch.unsqueeze(img_t, 0)

# perform inference
out = resnet(batch_t)

# lookup the labels

# print top-5 classes predicted by model
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
for idx in indices[0][:5]:
    print('Label:', idx, '. Confidence Score: {:.3f} %'.format(percentage[idx].item()))


# #### Task
# 
# Modify the code above, to perform data augmentation for the testing sample (averaging the scores of 5 crops: center crop, upper left crop, lower left crop, lower right crop, upper right crop).
# 
# Please discuss the advantages and disadvantages of using testing data augmentation.

# In[119]:


# https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.FiveCrop


# In[120]:


transform_after_crop = torchvision.transforms.Compose([
 torchvision.transforms.ToTensor(),                     
 torchvision.transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                            
 std=[0.229, 0.224, 0.225]
 )])


# In[121]:


npimg = np.array(img)

crops = [npimg[:224,:224,:], npimg[:224,-224:,:], # top left, top right
         npimg[-224:,:224,:], npimg[-224:,-224:,:], # bottom left, bottom right
         npimg[npimg.shape[0]//2 - 112:npimg.shape[0]//2 + 112,
               npimg.shape[1]//2 - 112:npimg.shape[1]//2 + 112,:]]  # center plot

fig, ax = plt.subplots(1,5,figsize=(10,3))
for i,crop in enumerate(crops):
    ax[i].imshow(crop)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.show()


# In[128]:


batch_t = torch.stack([transform_after_crop(crop) for crop in crops])
plt.imshow(torch.einsum("chw -> hwc", batch_t[-1]))


# In[132]:


# perform inference
out_arr = resnet(batch_t)

# lookup the labels

# print top-5 classes predicted by model
for out in out_arr:
    out = torch.stack([out])
    _, indices = torch.sort(out, descending=True)
    print()
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    for idx in indices[0][:5]:
        print('Label:', idx, '. Confidence Score: {:.3f} %'.format(percentage[idx].item()))


# In[124]:


# indices


# In[125]:


get_ipython().system('jupyter nbconvert --to script homework.ipynb')


# In[ ]:




