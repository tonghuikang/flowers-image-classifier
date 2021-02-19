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


# In[10]:


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


# In[11]:


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
# (Note: In this task, if you are adapting the code based on the open-source projects, pls acknowledge the original source in your code files, and also clearly mention it in your report. Also you need to clearly highlight which parts are done by yourself)

# ## TASK ONE
# 
# 
# (1) Replace the used base model (densenet169) to another model (refer to https://pytorch.org/vision/0.8/models.html for more types of models). Pls compare the performance of these two models on the validation set. 

# In[12]:


get_ipython().system('python train.py "./flowers" --arch densenet169 --gpu --epochs=1')


# In[13]:


get_ipython().system('python train.py "./flowers" --arch resnet18 --gpu --epochs=1')


# ## TASK TWO
# 
# (2) Please try different training methods that use densenet169 as the base model and compare their performance on the validation set.
# - finetuning the model but only updating the top layers
# - finetuning the whole model
# - training the whole model from scratch
# 
# Please also draw the curves of training/validation losses over training steps for these methods, and give your analysis based on the observed curves.

# ### Finetuning the model but only updating the top layers

# In[27]:


get_ipython().system('python train.py "./flowers" --arch densenet169 --gpu --epochs=10 > ./logs/finetune_only_top_layer.txt')


# ### Finetuning the whole model

# In[28]:


get_ipython().system('python train.py "./flowers" --arch densenet169 --gpu --train_all_layers --epochs=10 > ./logs/finetune_whole_model.txt')


# ### Training the whole model from scratch

# In[29]:


get_ipython().system('python train.py "./flowers" --arch densenet169 --gpu --not_use_pretrained --train_all_layers --epochs=10 > ./logs/train_from_scratch.txt')


# In[33]:


import matplotlib.pyplot as plt

files = ["finetune_only_top_layer", "finetune_whole_model", "train_from_scratch"]
colors = ["r", "g", "b"]
plt.figure(figsize=(14,6))

for file, color in zip(files, colors):
    with open("./logs/{}.txt".format(file), "r") as f:
        text = f.readlines()
        steps = []
        loss_train = []
        loss_valid = []
        for line in text:
            words = line.split()
            if words[0] != "Epoch:": continue
            steps.append(int(words[3]))
            loss_train.append(float(words[7]))
            loss_valid.append(float(words[11]))
        plt.plot(steps, loss_train, label="{} train loss".format(file), ls="--", color=color)
        plt.plot(steps, loss_valid, label="{} valid loss".format(file), ls="-", color=color)

plt.ylabel("loss")
plt.xlabel("steps")
plt.legend()
plt.show()


# ## TASK THREE
# 
# (3) For the model based on densenet169, please also report its performance (when you use the training method of finetuning the model but only updating the top layers) on the testing set.

# In[60]:


get_ipython().system('python train.py "./flowers" --arch densenet169 --gpu --test_model --epochs=1')


# ## TASK FOUR
# 
# (4) Please replace the base model to a new model which contains some convolutional layers. You need to write this new model by yourselves, and then report its performance on the validation set. Note, pls try different numbers of convolutional layers for your model, and compare their results, and give analysis for the results. You need to try at least 2 different numbers of conv layers.

# In[83]:


get_ipython().system('python train.py "./flowers" --arch homemade_CNN_small --not_use_pretrained --gpu --epochs=1')


# In[84]:


get_ipython().system('python train.py "./flowers" --arch homemade_CNN_large --not_use_pretrained --gpu --epochs=1')


# ## Extra tasks (not included in Homework 3)
# 
# (5) Please try using two different learning rate scheduling schemes for densenet169, and compare the performance on the validation set.
# 
# (6) Please try using two different optimizers for densenet169, and compare the performance on the validation set.

# In[42]:


get_ipython().system('python model_ic.py')


# In[34]:


get_ipython().system('jupyter nbconvert --to script homework.ipynb')


# In[ ]:




