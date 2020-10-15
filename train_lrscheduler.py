#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import augdataset
import loss, train_utils, models
import utils


# In[2]:


import logging
logging.basicConfig(filename='train_status.log',
                            filemode='a',
                            format='%(asctime)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


# In[3]:


# get the trainig data
train_dataloader = augdataset.get_train_dl(dirs=['/home/devi_prasad/UrineSedimentation/data_imgs/10k_annotatio_details_20200901/confusing/*.*'])


# In[4]:


# import model 
# output feature size = 128
model = models.Model(features_dim=128)
model = model.cuda()
# resume training
# model = torch.load('models/modelv1-Copy1.0.pt')


# In[5]:


# for param in model.parameters():
#     print(param.requires_grad)# = True


# In[6]:


v = 2
# SimCLR loss https://arxiv.org/pdf/2002.05709.pdf
criterion = loss.SimCLRLoss(temperature=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# decay the learning rate by 0.1 after every 500 epochs
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.1, last_epoch=-1,)

# decay by .1 if no improvement for 100 epochs
scheduler = utils.ReduceLROnPlateauWithBacktrack(optimizer, model, filename=f'models/modelv{v}.pt', factor=0.1, verbose=False, patience=100, warmup_steps=0, eps=1e-8)
n_epochs = 5000


# In[ ]:


# train model
for epoch in range(n_epochs):
    
    total_loss = train_utils.train(train_dataloader, model, criterion, optimizer)
    
    print(f"Epoch {epoch} total_loss {total_loss} lr : {optimizer.param_groups[0]['lr']}", )
    
    logging.info(f"modelv{v} Epoch {epoch} total_loss {total_loss} lr {optimizer.param_groups[0]['lr']}")
    scheduler.step([-total_loss])
    
    torch.save([model, optimizer], f'models/modelv{v}_curr.pt')


# In[ ]:




