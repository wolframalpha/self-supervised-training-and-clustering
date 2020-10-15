#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
import torch
import augdataset
import loss, train_utils, models
from utils import get_features, show_images
from models import Model


# In[2]:


# get the dataloader with transformations - no augmentations


# In[11]:


# function to the features from images 
model = Model().cuda()
model.load_state_dict(torch.load('models/modelv2.pt')['model'])

fps, features = get_features(model, ['/home/devi_prasad/UrineSedimentation/data_imgs/10k_annotatio_details_20200901/confusing/*.*'])


# In[12]:


# cluster based on features
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
kmeans = KMeans(n_clusters=100, random_state=0).fit(features)
labels = kmeans.labels_


# In[13]:


# number of points/images
len(labels)


# In[14]:


# the distribution of labels
from collections import Counter
Counter(labels), len(set(labels))


# In[15]:


# zip labels with filepaths of images
preds = list(zip(labels, fps))


# In[22]:


from skimage import io
images = [io.imread(fp) for cluster, fp in preds if cluster == 3]


# In[23]:


show_images(images[:10])


# In[10]:



import shutil, os
output_dir = f'../../data_imgs/pretextselfsupervisedv4'
# output_dir = f'../../data_imgs/selfsuperviseddbscanesp0.12'

clusters_to_write = preds

try:
    os.mkdir(output_dir)
except Exception as e:
    print(e)
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    
for cluster in list(set([cluster for cluster, _ in clusters_to_write])):
#     print(class_)
    os.mkdir(os.path.join(output_dir, str(cluster)))
    
for cluster, fp in clusters_to_write:
    cluster_dir = os.path.join(output_dir, str(cluster))
    filename = os.path.basename(fp)
    dest_fp = os.path.join(cluster_dir, f'{filename}')
    shutil.copy(fp, dest_fp)
    


# In[11]:


get_ipython().system('7z a {os.path.abspath(os.path.dirname(output_dir))}/{os.path.basename(output_dir)}.7z {output_dir} ')


# In[ ]:




