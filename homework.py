#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch, torchvision
from PIL import Image

# load model
resnet = torchvision.models.resnet101(pretrained=True)

# set network to evaluation mode
resnet.eval()

transform = torchvision.transforms.Compose([          
 torchvision.transforms.Resize(256),                   
 torchvision.transforms.CenterCrop(224),               
 torchvision.transforms.ToTensor(),                     
 torchvision.transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                            
 std=[0.229, 0.224, 0.225]                             
 )])


img = Image.open("https://upload.wikimedia.org/wikipedia/en/5/5f/Original_Doge_meme.jpg") 
img_t = transform(img)
print(img_t.shape)

batch_t = torch.unsqueeze(img_t, 0)


# perform inference
out = resnet(batch_t)

# print top-5 classes predicted by model
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
for idx in indices[0][:5]:
  print('Label:', idx, '. Confidence Score:', percentage[idx].item(), '%')


# In[ ]:




