#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[2]:


class LastDay(nn.Module):
    def __init__(self):
        super(LastDay, self).__init__()
    
    def run(self, x_u_v, x_adj):
        u, v = int(x_u_v[0]), int(x_u_v[1])
        sum_x_adj = x_adj[:,-1,:,:]     #(batch, U+2V, U+2V)
        estimate_y = sum_x_adj[:, 0:u, u:u+v]   #(batch, U, V)
        expand_y = estimate_y.view((estimate_y.size()[0], 1, estimate_y.size()[1], estimate_y.size()[2]))
        return expand_y


# In[ ]:




