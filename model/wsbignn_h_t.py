#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# In[2]:


random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
random.seed(42)
r = random.random
device = torch.device("cuda:0")


# In[3]:


#input1: x_u_v              dim = 2
#input2: x_poi              dim = (nv, 200)
#input3: x_user             dim = (nu, 200)
#input4: x_adj              dim = (batch, x_day, nu+2nv, nu+2nv)
#input5: batch_size           
#output1: predict_score     dim = (batch, y_day, nu, nv)


# In[4]:


class WSBiGNNLayer(nn.Module):
    def init_tensor(self, dim_1, dim_2):
        t = nn.Parameter(torch.empty(size=(dim_1, dim_2)))
        t.requires_grad = True
        nn.init.xavier_uniform_(t.data, gain=1.414)
        t.data = t.data
        return t
    
    def __init__(self, hid_dim):
        super(WSBiGNNLayer, self).__init__()
        #1-4: vanilla model
        #1. (from, to) = (mob POIs, users)
        self.W_mp_u_1 = self.init_tensor(hid_dim, hid_dim)
        self.W_mp_u_2 = self.init_tensor(hid_dim, hid_dim)
        self.W_mp_u_3 = self.init_tensor(1, hid_dim)
        #2. (web POIs, users)
        self.W_wp_u_1 = self.init_tensor(hid_dim, hid_dim)
        self.W_wp_u_2 = self.init_tensor(hid_dim, hid_dim)
        self.W_wp_u_3 = self.init_tensor(1, hid_dim)
        #3. (users, mob POIs)
        self.W_u_mp_1 = self.init_tensor(hid_dim, hid_dim)
        self.W_u_mp_2 = self.init_tensor(hid_dim, hid_dim)
        self.W_u_mp_3 = self.init_tensor(1, hid_dim)
        #4. (users, web POIs)
        self.W_u_wp_1 = self.init_tensor(hid_dim, hid_dim)
        self.W_u_wp_2 = self.init_tensor(hid_dim, hid_dim)
        self.W_u_wp_3 = self.init_tensor(1, hid_dim)        
        
        #5-7: hyperedges
        #5. (users, users)
        self.W_u_u_hyper_1 = self.init_tensor(hid_dim, hid_dim)
        self.W_u_u_hyper_2 = self.init_tensor(hid_dim, hid_dim)
        self.W_u_u_hyper_3 = self.init_tensor(1, hid_dim) 
        #6. (mob POIs, mob POIs)
        self.W_mp_mp_hyper_1 = self.init_tensor(hid_dim, hid_dim)
        self.W_mp_mp_hyper_2 = self.init_tensor(hid_dim, hid_dim)
        self.W_mp_mp_hyper_3 = self.init_tensor(1, hid_dim)        
        #7. (web POIs, web POIs)
        self.W_wp_wp_hyper_1 = self.init_tensor(hid_dim, hid_dim)
        self.W_wp_wp_hyper_2 = self.init_tensor(hid_dim, hid_dim)
        self.W_wp_wp_hyper_3 = self.init_tensor(1, hid_dim) 
        
    def forward(self, u, mp, wp, adj):
        #input1: u. (nu, hid_dim)
        #input2: mp. (nv, hid_dim)
        #input3: wp. (nv, hid_dim)
        #input4: adj. (nu+2nv, nu+2nv)
        #output1: new_u.    (nu, hid_dim)
        #output2: new_mp.   (nv, hid_dim)
        #output3: new_wp.   (nv, hid_dim)
        nu, nv = u.shape[0], mp.shape[0]
        idx_1, idx_2 = nu + nv, nu + 2*nv
    
        ################################################
        ones_nu_nv = torch.ones((nu, nv), device=device)*0.001
        ones_nv_nu = torch.ones((nv, nu), device=device)*0.001
        
        #1. Update embeddings for users
        #1.1 from mob POIs
        mp_u_1 = torch.mm(u, self.W_mp_u_1)              #(nu, hid_dim)
        adj_mp_u = adj[0:nu, nu:idx_1] + ones_nu_nv
        adj_mp_u_normalize = adj_mp_u / adj_mp_u.sum(dim=1, keepdim=True)    #(nu, nv)
        mp_u_2 = torch.mm(torch.mm(adj_mp_u_normalize, mp), self.W_mp_u_2)   #(nu, hid_dim)
        mp_u_3 = self.W_mp_u_3.repeat(nu, 1)               #(nu, hid_dim)
        mp_u = torch.sigmoid(mp_u_1 + mp_u_2 + mp_u_3)     #(nu, hid_dim)
        
        #1.2 from web POIs
        wp_u_1 = torch.mm(u, self.W_wp_u_1)                #(nu, hid_dim)
        adj_wp_u = adj[0:nu, idx_1:idx_2] + ones_nu_nv
        adj_wp_u_normalize = adj_wp_u / adj_wp_u.sum(dim=1, keepdim=True)     #(nu, nv)
        wp_u_2 = torch.mm(torch.mm(adj_wp_u_normalize, wp), self.W_wp_u_2)    #(nu, hid_dim)
        wp_u_3 = self.W_wp_u_3.repeat(nu, 1)               #(nu, hid_dim)
        wp_u = torch.sigmoid(wp_u_1 + wp_u_2 + wp_u_3)     #(nu, hid_dim)       
        #summarize 1.1 and 1.2
        new_u = (mp_u + wp_u)/2.0                          #(nu, hid_dim)  
        
        #1.3: from user hyperedges
        u_u_1 = torch.mm(u, self.W_u_u_hyper_1)            #(nu, hid_dim)
        adj_u_u_hyper_m = torch.mm(adj[0:nu, nu:idx_1], adj[nu:idx_1, 0:nu])          #(nu, nu)
        adj_u_u_hyper_t = torch.mm(adj[0:nu, idx_1:idx_2], adj[idx_1:idx_2, 0:nu])  #(nu, nu)
        adj_u_u_hyper = 1.0-(1.0-adj_u_u_hyper_m)*(1.0-adj_u_u_hyper_t)       #(nu, nu) (0,0)->0; (1,0)->1; (1,1)->1.
        u_u_2 = torch.mm(torch.mm(adj_u_u_hyper, u), self.W_u_u_hyper_2)
        u_u_3 = self.W_u_u_hyper_3.repeat(nu, 1)          #(nu, hid_dim)
        u_u = torch.sigmoid(u_u_1 + u_u_2 + u_u_3)
        
        ################################################
        #2. Update embeddings for mob POIs
        #2.1: from users
        u_mp_1 = torch.mm(mp, self.W_u_mp_1)                                  #(nv, hid_dim)        
        adj_u_mp = adj[nu:idx_1, 0:nu] + ones_nv_nu
        adj_u_mp_normalize = adj_u_mp / adj_u_mp.sum(dim=1, keepdim=True)     #(nv, nu)
        u_mp_2 = torch.mm(torch.mm(adj_u_mp_normalize, u), self.W_u_mp_2)     #(nv, hid_dim)
        u_mp_3 = self.W_u_mp_3.repeat(nv, 1)
        u_mp = torch.sigmoid(u_mp_1 + u_mp_2 + u_mp_3)                        #(nv, hid_dim)       
        #2.2 from hyperedges
        mp_mp_1 = torch.mm(mp, self.W_mp_mp_hyper_1)                          #(nv, hid_dim)
        adj_mp_mp_hyper = torch.mm(adj[nu:idx_1, 0:nu], adj[0:nu, nu:idx_1])  #(nv, nv)
        mp_mp_2 = torch.mm(torch.mm(adj_mp_mp_hyper, mp), self.W_mp_mp_hyper_2)
        mp_mp_3 = self.W_mp_mp_hyper_3.repeat(nv, 1)                          #(nv, hid_dim)
        mp_mp = torch.sigmoid(mp_mp_1 + mp_mp_2 + mp_mp_3)     
        
        ################################################
        #3. Update embeddings for web POIs
        #3.1 from users
        u_wp_1 = torch.mm(wp, self.W_u_wp_1)                                  #(nv, hid_dim)        
        adj_u_wp = adj[idx_1:idx_2, 0:nu] + ones_nv_nu
        adj_u_wp_normalize = adj_u_wp / adj_u_wp.sum(dim=1, keepdim=True)     #(nv, nu)
        u_wp_2 = torch.mm(torch.mm(adj_u_wp_normalize, u), self.W_u_wp_2)     #(nv, hid_dim)
        u_wp_3 = self.W_u_wp_3.repeat(nv, 1)
        u_wp = torch.sigmoid(u_wp_1 + u_wp_2 + u_wp_3)                        #(nv, hid_dim)
        #3.2 from hyperedges
        wp_wp_1 = torch.mm(wp, self.W_wp_wp_hyper_1)                          #(nv, hid_dim)
        adj_wp_wp_hyper = torch.mm(adj[idx_1:idx_2, 0:nu], adj[0:nu, idx_1:idx_2])   #(nv, nv)
        wp_wp_2 = torch.mm(torch.mm(adj_wp_wp_hyper, wp), self.W_wp_wp_hyper_2)
        wp_wp_3 = self.W_wp_wp_hyper_3.repeat(nv, 1)                          #(nv, hid_dim)
        wp_wp = torch.sigmoid(wp_wp_1 + wp_wp_2 + wp_wp_3)
        
        return new_u, u_mp, u_wp, u_u, mp_mp, wp_wp


# In[5]:


#functionality: get user and location embeddings by propagating information along x days.
#Input1: x_u_v              dim = 2
#Input2: x_location         dim = (nv, 200)
#Input3: x_user             dim = (nu, 200)
#Input4: x_adj              dim = (batch, x_day, nu+2nv, nu+2nv)
#Input5: batch_size           
#Output1: predict_score     dim = (batch, y_day, nu, nv)
class WSBiGNN(nn.Module):
    def init_tensor(self, dim_1, dim_2):
        t = nn.Parameter(torch.empty(size=(dim_1, dim_2)))
        t.requires_grad = True
        nn.init.xavier_uniform_(t.data, gain=1.414)
        return t
    
    def __init__(self, hid_dim, hid_dim_cons, x_day, member):
        super(WSBiGNN, self).__init__()    
        self.WSBiGNNLayer = WSBiGNNLayer(hid_dim)
        self.hid_dim = hid_dim
        self.member = member
        self.nu = member.size()[1]
        self.nv = member.size()[2]
        self.W_u = self.init_tensor(200, hid_dim)
        self.W_mp = self.init_tensor(200, hid_dim)
        self.W_wp = self.init_tensor(200, hid_dim)
        
        #M2: temporal weighting
        self.weight_u = torch.nn.Parameter(torch.empty(x_day))
        self.weight_u.requires_grad = True
        torch.nn.init.normal_(self.weight_u.data, mean=1.0/x_day, std=0.000)
        self.weight_mp = torch.nn.Parameter(torch.empty(x_day))
        self.weight_mp.requires_grad = True
        torch.nn.init.normal_(self.weight_mp.data, mean=1.0/x_day, std=0.000)
        
        #M1: hyperedge
        self.weight_hyper = torch.nn.Parameter(torch.empty(3))
        self.weight_hyper.requires_grad = True
        torch.nn.init.normal_(self.weight_hyper, mean=0.01, std=0.000) 
        
        #M3: search mobility memory
        self.s_g_u = self.init_tensor(self.nu, hid_dim_cons)
        self.s_g_v = self.init_tensor(self.nv, hid_dim_cons)
        self.s_ng_u = self.init_tensor(self.nu, hid_dim_cons)
        self.s_ng_v = self.init_tensor(self.nv, hid_dim_cons)
        self.ns_g_u = self.init_tensor(self.nu, hid_dim_cons)
        self.ns_g_v = self.init_tensor(self.nv, hid_dim_cons)
        self.ns_ng_u = self.init_tensor(self.nu, hid_dim_cons)
        self.ns_ng_v = self.init_tensor(self.nv, hid_dim_cons)
        
        self.s_g_w = self.init_tensor(hid_dim, self.nv)
        self.s_ng_w = self.init_tensor(hid_dim, self.nv)
        self.ns_g_w = self.init_tensor(hid_dim, self.nv)
        self.ns_ng_w = self.init_tensor(hid_dim, self.nv)
        
        self.const = torch.nn.Parameter(torch.empty(4))
        self.const.requires_grad = True
        torch.nn.init.uniform_(self.const,0,0.5)
 
    def run(self, x_u_v, x_poi, x_user, x_adj, bs):
        nu, nv = int(x_u_v[0]), int(x_u_v[1])
        idx1, idx2 = nu+nv, nu+2*nv
        x_day = x_adj.size()[1]
        #poi
        x_poi = x_poi.repeat(bs, 1, 1)   #dim = (batch, nv, 200)
        x_mp = torch.bmm(x_poi, self.W_mp.repeat(bs, 1, 1))   #dim = (batch, nv, hid_dim)
        x_wp = torch.bmm(x_poi, self.W_wp.repeat(bs, 1, 1))
        #user
        x_user = torch.bmm(x_user, self.W_u.repeat(bs, 1, 1))  #dim = (batch, nu, hid_dim)
        #loc_user
        x_user_mp = torch.cat([x_user, x_mp], dim=1)  #dim = (batch, nu+nv, hid_dim)
        x_user_p = torch.cat([x_user_mp, x_wp], dim=1)  #dim = (batch, nu+2nv, hid_dim)        
        #output
        y_output_embed = torch.zeros((1, 1, nu, nv), device=device)
        
        for i in range(bs):              
            x_on_day = x_user_p[i]                            #(nu+2nv, hid_dim)
            adj_on_day = x_adj[i]                             #(x_day, nu+2nv, nu+2nv)
            u_seq = torch.zeros((x_day, nu, self.hid_dim), device=device)      #for M2: temporal weighting
            mp_seq = torch.zeros((x_day, nv, self.hid_dim), device=device)     #for M2: temporal weighting
            
            for j in range(x_day):
                u = x_on_day[0:nu, :]           #(nu, hid_dim)
                mp = x_on_day[nu:idx1, :]       #(nv, hid_dim)
                wp = x_on_day[idx1:idx2, :]     #(nv, hid_dim)
                adj = adj_on_day[j]             #(nu+2nv, nu+2nv)
                #GNN
                new_u, u_mp, u_wp, u_u, mp_mp, wp_wp = self.WSBiGNNLayer(u, mp, wp, adj)
                
                #M1: hyperedge 
                new_u= new_u + self.weight_hyper[0]*u_u          #(nu, hid_dim)
                new_mp = u_mp + self.weight_hyper[1]*mp_mp       #(nv, hid_dim)
                new_wp = u_wp + self.weight_hyper[2]*wp_wp       #(nv, hid_dim)
                x_on_day = torch.cat([new_u, new_mp, new_wp], dim=0)   
                #dim = (nu+2*nv, hid_dim) 
                
                u_seq[j] = new_u
                mp_seq[j] = new_mp
            x_result_u =  torch.zeros((nu, self.hid_dim), device=device)
            x_result_v =  torch.zeros((nv, self.hid_dim), device=device)
            
            #M2: temporal weighting
            for j in range(x_day):
                x_result_u = x_result_u + self.weight_u[j] * u_seq[j]
                x_result_v = x_result_v + self.weight_mp[j] * mp_seq[j]                 
            
            x_result = torch.cat([x_result_u, x_result_v], dim=0)                         #(nu,nv)
            u_v_score = torch.mm(x_result[0:nu,], torch.transpose(x_result[nu:nu+nv,], 1, 0))
            
            output_embed = u_v_score.unsqueeze(dim=0).unsqueeze(dim=0)                    #(1,1,nu,nv)
            if i == 0:
                y_output_embed = output_embed
            else:
                y_output_embed = torch.cat([y_output_embed, output_embed], dim=0)
        return y_output_embed


# In[ ]:




