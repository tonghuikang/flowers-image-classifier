#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch, torchvision
from PIL import Image
from urllib.request import urlopen
import matplotlib.pyplot as plt
import numpy as np


# # PART ONE

# In[2]:


img = Image.open(urlopen("https://upload.wikimedia.org/wikipedia/en/5/5f/Original_Doge_meme.jpg"))
img


# In[3]:


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


# In[4]:


img_t = transform(img)
print(img_t.shape)
plt.imshow(torch.einsum("chw -> hwc", img_t))


# In[5]:


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
# Modify the code above, to perform data augmentation for the testing sample averaging the scores of 5 crops
# - center crop
# - upper left crop
# - lower left crop
# - lower right crop
# - upper right crop

# In[6]:


transform_after_crop = torchvision.transforms.Compose([
 torchvision.transforms.ToTensor(),                     
 torchvision.transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                            
 std=[0.229, 0.224, 0.225]
 )])


# In[7]:


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


# In[8]:


batch_t = torch.stack([transform_after_crop(crop) for crop in crops])
plt.imshow(torch.einsum("chw -> hwc", batch_t[-1]))


# In[9]:


RESNET_LABEL_URL = "https://gist.githubusercontent.com/steverichey/3d87d581a8713e65ac6c2d0848151ff5/raw/272734c0bb45847d47f1ecb1f75b814379a5c3d0/imagenet_labels.txt"
text = urlopen(RESNET_LABEL_URL).read().decode('utf-8')
labels = {i:k for i,k in enumerate(text.split("\n"), start=1)}


# In[ ]:


# perform inference
out_arr = resnet(batch_t)

# lookup the labels
all_percentages = np.zeros(1000)

# print top-5 classes predicted by model
for out in out_arr:
    out = torch.stack([out])
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    all_percentages += percentage.detach().numpy()/5
    _, indices = torch.sort(out, descending=True)
    print()
    for idx in indices[0][:5]:
        print('Confidence Score: {:.3f} % - {}'.format(percentage[idx].item(), labels[idx.item()]))


# In[ ]:


for idx, _ in sorted(enumerate(all_percentages), key=lambda x:x[1])[-5:][::-1]:
    print('Confidence Score: {:.3f} % - {}'.format(all_percentages[idx], labels[idx]))


# Please discuss the advantages and disadvantages of using testing data augmentation.
# 
# Advantages
# - Data augmentation with corner and center crops allows for different views of the testing data.
# - It is important to carry out the same augmentation that is applied on the training data (except distortions), such as normalising and resizing. This allow the test sample to match how the train sample was prepared and is of the correct dimension.
# 
# Disadvantages
# - If the laballed object is small and is in the center of the picture, the center crops may not capture the object and the prediction will be irrelevant, weakening the confidence of the prediction.

# # PART TWO

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


get_ipython().system('jupyter nbconvert --to script homework.ipynb')


# In[ ]:




