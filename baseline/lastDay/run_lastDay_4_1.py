#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
from lastDay import LastDay


# # 1. set parameters

# In[2]:


print (torch.cuda.is_available())
device = torch.device("cuda:0")


# In[3]:


random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
r = random.random


# In[4]:


x_day, y_day = 4, 1  
train_ratio, validate_ratio = 0.70, 0.10
top_k = 3
hyper_param = {}


# In[5]:


root_path = "/home/umni2/a/umnilab/users/xue120/umni4/2023_web_mobility_summer"+\
                "/1_data_check/data_feature_generation/"
file_name = root_path + "feature_" + str(x_day) + "_" + str(y_day)
train_path = file_name + "/train.json"
vali_path = file_name + "/validate.json"
test_path = file_name + "/test.json"
sampled_user_location_path = file_name + "/sampled_user_location.json"


# # 2. compute the loss

# In[6]:


#2.1: compute Recall@k, NDCG@k
#input1:  gnn_output   dim = (batch, y_day, U, V)
#input2:  real_link    dim = (batch, y_day, n_edge, 2)
#inputs3,4:  n_user, n_loc
#input5: top_k
#output: Recall@K, NDCG@K
def compute_recall_ndcg(gnn_output, real_link, n_user, n_loc, top_k):
    batch, y_day = gnn_output.size()[0], gnn_output.size()[1]
    recall_all = [[0.0 for j in range(y_day)] for i in range(batch)] 
    ndcg_all = [[0.0 for j in range(y_day)] for i in range(batch)] 
    
    for i in range(batch):
        for j in range(y_day):
            recall_user, ndcg_user = {}, {}
            predicted, real = gnn_output[i][j].tolist(), real_link[i][j] 
            
            #1. construct the real mobility
            real_list, real_dict = {user: [] for user in range(n_user)}, {user: {} for user in range(n_user)}
            for k in range(len(real)):
                edge = real[k]
                user, poi = int(edge[0]), int(edge[1])
                if user > -1:
                    real_list[user].append(poi)
                else:
                    break
            for user in real_list:
                real_dict[user] = set(real_list[user])
            
            #2. compute Recall@k, NDCG@k
            for user in real_dict:
                real_poi = real_dict[user]
                len_real_poi = len(real_poi)
                if len_real_poi > 0:
                    predict_row = predicted[user]                 #[0,0,12,1,5]
                    largest_k_idx = np.argsort(predict_row)[::-1] #[2,4,3,1,0]
                    top_k_idx = largest_k_idx[0: top_k]            #[2,4,3]
                    
                    #compute Recall
                    predict_top_k = set(top_k_idx)
                    recall_user[user] = len(predict_top_k.intersection(real_poi))*1.0/len_real_poi
                    
                    #compute NDCG
                    weight = [1.0/(math.log(k+2)/math.log(2.0)) for k in range(top_k)]
                    #denominator
                    if len_real_poi < top_k:
                         best_rank = [1.0]* len_real_poi + [0.0]*(top_k-len_real_poi)
                    else:
                         best_rank = [1.0]* top_k        
                    #numerator  
                    predict_rank = [0.0]* top_k
                    for idx in range(len(top_k_idx)):
                        if top_k_idx[idx] in real_poi:
                            predict_rank[idx] = 1.0
                    #NDCG
                    ndcg_user[user] = float(np.dot(weight, predict_rank)/np.dot(weight, best_rank))
            
            #3. compute the average Recall@k, average NDCG@k.
            recall_all[i][j] = float(np.mean(list(recall_user.values())))
            ndcg_all[i][j] = float(np.mean(list(ndcg_user.values())))
    ave_recall, ave_ndcg = np.mean(recall_all), np.mean(ndcg_all)
    print ("ave Recall", ave_recall)
    print ("ave NDCG", ave_ndcg)
    return recall_all, ndcg_all, ave_recall, ave_ndcg

#2.2: evaluate the trained model on validation or test                     
def validate_test(trained_model, criterion, vali_test, hyper_param, top_k):    
    y_real = vali_test["y_mob"]  # dim = (batch, y_day, n_m, 2)
    
    x_adj = convert_to_adj(vali_test["x_mob"], vali_test["x_text"], hyper_param["n_user"], hyper_param["n_loc"])
    y_hat = trained_model.run(vali_test["u_v"].to(device), x_adj.to(device))
    
    all_recall, all_ndcg, ave_recall, ave_ndcg = compute_recall_ndcg(y_hat.cpu(), y_real, hyper_param["n_user"],\
                                              hyper_param["n_loc"], top_k)
    return all_recall, all_ndcg, ave_recall, ave_ndcg


# # 3. compute adjacency matrix based on mobility and web search

# In[7]:


#3.1:
#input1: x_mob_batch (tensor)        dim = (batch, x_day, n_m, 2)
#input2: x_text_batch (tensor)       dim = (batch, x_day, n_t, 2)
#input3,4: u_user, n_loc
#output: adj  (tensor)            dim = (batch, x_day, n_user+2*n_loc, n_user+2*n_loc)
def convert_to_adj(x_mob_batch, x_text_batch, n_user, n_loc):
    batch, x_day = x_mob_batch.size()[0], x_mob_batch.size()[1]
    adj_dim = n_user + 2*n_loc
    adj = torch.zeros((batch, x_day, adj_dim, adj_dim)).to(device)
    for i in range(batch):
        x_mob_record, x_text_record = x_mob_batch[i], x_text_batch[i]
        for j in range(x_day):
            x_mob_one_day, x_text_one_day = x_mob_record[j], x_text_record[j]
            #extract mob edges
            for link in x_mob_one_day:
                if link[0] != -1:
                    user, loc = link[0], link[1] 
                    adj[i][j][user][n_user + loc] = adj[i][j][user][n_user + loc] + 1 
                    adj[i][j][n_user + loc][user] = adj[i][j][n_user + loc][user] + 1
                else:
                    break
            #extract text edges
            for link in x_text_one_day:
                if link[0] != -1:
                    user, loc = link[0], link[1] 
                    adj[i][j][user][n_user + n_loc + loc] = adj[i][j][user][n_user + n_loc + loc] +1
                    adj[i][j][n_user + n_loc + loc][user] = adj[i][j][n_user + n_loc + loc][user] +1 
                else:
                    break
    return adj


# # 4. train the model

# In[8]:


#4.1: tensorize
def tensorize(train_vali_test):
    result = dict()
    result["u_v"] = torch.tensor(train_vali_test["u_v"]) 
    result["x_poi"] = torch.tensor(train_vali_test["x_poi"])     
    result["x_mob"] = torch.tensor(train_vali_test["x_mob"]) 
    result["x_text"] = torch.tensor(train_vali_test["x_text"]) 
    result["y_mob"] = torch.tensor(train_vali_test["y_mob"]) 
    return result


# In[9]:


#4.2: load the data
#train
train = tensorize(json.load(open(train_path)))
#validate
vali = tensorize(json.load(open(vali_path)))
#test
test = tensorize(json.load(open(test_path)))
sampled_user_location = json.load(open(sampled_user_location_path))
sampled_user_location["n_user"] = len(sampled_user_location["u"])
sampled_user_location["n_loc"] = len(sampled_user_location["p"])

u_list = sampled_user_location["u"]
hyper_param["n_user"] = len(u_list)
p_list = sampled_user_location["p"]
hyper_param["n_loc"] = len(p_list)


# In[10]:


#4.3: model 
trained_model = LastDay()
criterion = nn.CrossEntropyLoss()


# In[11]:


#4.4: validate and test the model
print ("---------------validation-------------------")
all_recall, all_ndcg, ave_recall, ave_ndcg =\
    validate_test(trained_model, criterion, vali, hyper_param, top_k)
print ("-----------finish model validation---------------")


# In[12]:


print ("----------------test-------------------")
all_recall, all_ndcg, ave_recall, ave_ndcg =\
    validate_test(trained_model, criterion, test, hyper_param, top_k)
print ("-----------finish model testing------------")
#print (len(test_real), len(test_hat))


# In[ ]:




