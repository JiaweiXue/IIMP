#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import torch
import torch.nn as nn


# In[2]:


random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
random.seed(42)
r = random.random
device = torch.device("cuda:0")


# In[3]:


class LightGCNLayer(nn.Module):
    def __init__(self):
        super(LightGCNLayer, self).__init__()
    
    def forward(self, feature, D_n_A_D_n):
        H = torch.mm(D_n_A_D_n.float(), feature)   #(U+2V, U+2V) * (U+2V, K) = (U+2V, K)
        return H


# In[4]:


class LightGCN(nn.Module):
    def __init__(self):
        super(LightGCN, self).__init__()
        self.layer1 = LightGCNLayer()
        self.layer2 = LightGCNLayer()
        
    def compute_D_n_A_D_n(self, adjs):  
        n = adjs.size()[0]
        tilde_A = adjs + torch.diag(torch.ones(n, n)).to(device)    #(U+2V, U+2V)
        tilde_D_n = torch.diag(torch.pow(tilde_A.sum(-1).float(), -0.5))
        D_n_A_D_n = torch.mm(tilde_D_n, torch.mm(tilde_A, tilde_D_n))
        return D_n_A_D_n 

    def forward(self, x, adjs):
        #x = (U+2V, K)   #adjs: (U+2V, U+2V)
        D_n_A_D_n = self.compute_D_n_A_D_n(adjs)  #(U+2V, U+2V)  
        x1 = self.layer1(x, D_n_A_D_n)            #(U+2V, K)
        x2 = self.layer2(x1, D_n_A_D_n)           #(U+2V, K)
        x3 = (x + x1 + x2)/3.0
        return x3   


# In[5]:


#functionality: get user and location embeddings by propagating information along x days.
#Input1: x_u_v              dim = 2
#Input2: x_poi         dim = (V, 200)
#Input3: x_user             dim = (U, 200)
#Input4: x_adj              dim = (batch, x_day, U+2V, U+2V)
#Input5: batch_size           
#Output1: predict_score     dim = (batch, y_day, U, V)
class LightGCNFull(nn.Module):
    def init_tensor(self, dim_1, dim_2):
        t = nn.Parameter(torch.empty(size=(dim_1, dim_2)))
        t.requires_grad = True
        nn.init.xavier_uniform_(t.data, gain=1.414)
        return t
    
    def __init__(self, hid_dim):
        super(LightGCNFull, self).__init__()
        self.light_gcn = LightGCN()      
        self.fc_poi = self.init_tensor(200, hid_dim).to(device)
        self.fc_user = self.init_tensor(200, hid_dim).to(device)
        
    def run(self, x_u_v, x_poi, x_user, x_adj, b_s):
        u, v = int(x_u_v[0]), int(x_u_v[1])
        x_poi = x_poi.repeat(b_s, 1, 1)      #dim = (batch, V, 200)
        x_poi = torch.bmm(x_poi, self.fc_poi.repeat(b_s, 1, 1))   #dim = (batch, V, hid_dim)
        x_user = torch.bmm(x_user, self.fc_user.repeat(b_s, 1, 1)) #dim = (batch, U, hid_dim)
        x_user_poi = torch.cat([x_user, x_poi], dim=1)  #dim = (batch, U+V, hid_dim)
        x_user_poi = torch.cat([x_user_poi, x_poi], dim=1)  #dim = (batch, U+2V, hid_dim)        
        
        y_output_embed = torch.zeros((1, 1, u, v))
        for i in range(b_s):
            x = x_user_poi[i]     #(U+2V, hid_dim)
            mean_x_adj = torch.mean(x_adj[i], dim=0)     #(U+2V, U+2V)
            x_result = self.light_gcn(x, mean_x_adj)     #(U+2V, 8)

            u_v_score = torch.mm(x_result[0:u,], torch.transpose(x_result[u:u+v,], 1, 0))
            output_embed = u_v_score.unsqueeze(dim=0).unsqueeze(dim=0) #(1,1,U,V)
            if i == 0:
                y_output_embed = output_embed
            else:
                y_output_embed = torch.cat([y_output_embed, output_embed], dim=0)
        return y_output_embed

