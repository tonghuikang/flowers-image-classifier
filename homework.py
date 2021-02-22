#!/usr/bin/env python
# coding: utf-8

# # Deep Learning HW 3

# This code was run on Google Cloud Platform Notebooks, with PyTorch 1.7 and NVIDIA Tesla T4
# 
# This notebook
# - https://github.com/tonghuikang/flowers-image-classifier/blob/master/homework.ipynb
# 
# Code changes relative to the reference repository
# - https://github.com/MiguelAMartinez/flowers-image-classifier/compare/master...tonghuikang:master
# 
# Logs that are not printed in this notebook
# - https://github.com/tonghuikang/flowers-image-classifier/tree/master/logs

# In[25]:


from urllib.request import urlopen

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import torch, torchvision


# # PART ONE

# In[26]:


img = Image.open(urlopen("https://upload.wikimedia.org/wikipedia/en/5/5f/Original_Doge_meme.jpg"))
img


# In[20]:


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


# In[21]:


img_t = transform(img)
print(img_t.shape)
plt.imshow(torch.einsum("chw -> hwc", img_t))


# In[24]:


batch_t = torch.unsqueeze(img_t, 0)

# perform inference
out = resnet(batch_t)

# lookup the labels
RESNET_LABEL_URL = "https://gist.githubusercontent.com/steverichey/3d87d581a8713e65ac6c2d0848151ff5/raw/272734c0bb45847d47f1ecb1f75b814379a5c3d0/imagenet_labels.txt"
text = urlopen(RESNET_LABEL_URL).read().decode('utf-8')
labels = {i:k for i,k in enumerate(text.split("\n"), start=1)}

# print top-5 classes predicted by model
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
for idx in indices[0][:5]:
    print('Confidence Score: {:.3f} % - {}'.format(percentage[idx].item(), labels[idx.item()]))


# #### Task
# 
# > Modify the code above, to perform data augmentation for the testing sample averaging the scores of 5 crops
# > - center crop
# > - upper left crop
# > - lower left crop
# > - lower right crop
# > - upper right crop

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


# In[10]:


# perform inference
out_arr = resnet(batch_t)

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
# > (Note: In this task, if you are adapting the code based on the open-source projects, pls acknowledge the original source in your code files, and also clearly mention it in your report. Also you need to clearly highlight which parts are done by yourself)
# 
# Please refer to https://github.com/MiguelAMartinez/flowers-image-classifier/compare/master...tonghuikang:master to understand how the code is adapted for this homework

# ## TASK ONE
# 
# 
# > (1) Replace the used base model (densenet169) to another model (refer to https://pytorch.org/vision/0.8/models.html for more types of models). Pls compare the performance of these two models on the validation set. 

# In[12]:


get_ipython().system('python train.py "./flowers" --arch densenet169 --gpu --epochs=3')


# In[13]:


get_ipython().system('python train.py "./flowers" --arch resnet18 --gpu --epochs=3')


# ## TASK TWO
# 
# > (2) Please try different training methods that use densenet169 as the base model and compare their performance on the validation set.
# > - finetuning the model but only updating the top layers
# > - finetuning the whole model
# > - training the whole model from scratch
# 
# Please also draw the curves of training/validation losses over training steps for these methods, and give your analysis based on the observed curves.

# ### Finetuning the model but only updating the top layers

# In[14]:


# !python train.py "./flowers" --arch densenet169 --gpu --epochs=100 > ./logs/finetune_only_top_layer.txt


# ### Finetuning the whole model

# In[15]:


# !python train.py "./flowers" --arch densenet169 --gpu --train_all_layers --epochs=100 > ./logs/finetune_whole_model.txt


# ### Training the whole model from scratch

# In[46]:


# !python train.py "./flowers" --arch densenet169 --gpu --not_use_pretrained --train_all_layers --epochs=100 > ./logs/train_from_scratch.txt


# In[8]:


import matplotlib.pyplot as plt

def compare_loss(files, colors, ymin=0.1):
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

    plt.ylabel("loss (on logarithmic scale)")
    plt.xlabel("steps")
    plt.yscale("log")
    plt.ylim(ymin,10)
    plt.legend()
    plt.show()

compare_loss(["finetune_only_top_layer", "finetune_whole_model", "train_from_scratch"], ["r", "g", "b"], ymin=0.01)


# Training from scratch results in a very slow convergence because all the weights needed to be trained from scratch. The loss it converges to is higher because it did not take advantage from the pretrained weights learnt from a far bigger dataset. There are only 8189 flowers in the dataset, whereas the imagenet dataset contains millions of pictures.
# 
# Finetuning pretrained model, whether is it just the full model or the last layer, allow us to achieve low loss quickly.
# 
# Both the finetuning of the whole model and the finetuning of the top layer overfitted soon after achieving low validation loss. This is because the number of paramters trained is much larger than the number of flowers.
# 
# The overfitting did not result in huge increases in validation loss. I suspect that this is due to the dropout that we have implemented in the final layer that prevented the model from overfitting badly.
# 
# I would have expected the finetuning of the whole model to overfit more because the number of parameters is much larger than the number of samples. This does not appear to be the case as the training and validation loss closely match each other. Again, I suspect that this is due to the dropout implemented in the final layer.

# ## TASK THREE
# 
# > (3) For the model based on densenet169, please also report its performance (when you use the training method of finetuning the model but only updating the top layers) on the testing set.

# In[29]:


get_ipython().system('python train.py "./flowers" --arch densenet169 --gpu --test_model --epochs=3')


# For this task, I added an option to run the code to test the model.

# ## TASK FOUR
# 
# > (4) Please replace the base model to a new model which contains some convolutional layers. You need to write this new model by yourselves, and then report its performance on the validation set. Note, pls try different numbers of convolutional layers for your model, and compare their results, and give analysis for the results. You need to try at least 2 different numbers of conv layers.
# 
# The new model `homemade_CNN` defined in `model_ic.py`. 
# 
# `homemade_CNN_small` contains two convolutional layers while `homemade_CNN_large` contains four convolutional layers.

# In[42]:


# !python train.py "./flowers" --arch homemade_CNN_small --is_homemade --gpu --epochs=100 > ./logs/homemade_CNN_small.txt


# In[43]:


# !python train.py "./flowers" --arch homemade_CNN_large --is_homemade --gpu --epochs=100 > ./logs/homemade_CNN_large.txt


# In[9]:


compare_loss(["train_from_scratch", "homemade_CNN_small", "homemade_CNN_large"], ["b", "g", "r"], ymin=1)


# The small CNN network (2 convolutional layers) starts to overfit at the 10th epoch, whereas the larger CNN network (4 convolutional layers) overfits later at the 40th epoch.
# - Overfitting is observed when the training loss is significantly lower than the validation loss.
# - I suggest that this because the of the small amount of parameters the smaller CNN network has.
# 
# The larger CNN network did not achieve a lower validation loss.
# - I suspect that this is because of the lack of meaningful combination of the extra convolutional layers.
# 
# Training from scratch results in a lower validation loss.
# - I suspect that this is due better combination of the convolutional layers.
# - The output size decreases down the dense block (due to average pooling in the transition layer), results in better abstraction of the features.

# ## Extra tasks (not included in Homework 3)
# 
# (5) Please try using two different learning rate scheduling schemes for densenet169, and compare the performance on the validation set.
# 
# (6) Please try using two different optimizers for densenet169, and compare the performance on the validation set.

# In[34]:


get_ipython().system('jupyter nbconvert --to script homework.ipynb')


# In[ ]:




