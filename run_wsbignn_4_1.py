#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import math
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from wsbignn import WSBiGNN
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
torch.autograd.set_detect_anomaly(True)


# In[2]:


#1. set parameters
#2. visualization
#3. compute the loss
#4. initialize user embeddings
#5. compute adjacency matrix based on mobility and web search
#6. train
#7. compute the Recall and NDCG
#8. main
#9. save


# # 1: set parameters

# In[3]:


print (torch.cuda.is_available())
device = torch.device("cuda:0")
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
r = random.random


# In[4]:


x_day, y_day = 4, 1  
case = str(x_day) + "_" + str(y_day)
train_ratio, validate_ratio = 0.70, 0.10
top_k, npr = 3, 5
num_epochs, batch_size, learning_rate = 200, 2, 0.001
hid_dim = 32
hid_dim_cons = 32
hyper_param = {"n_e": num_epochs, "b_s": batch_size, "l_r": learning_rate, "top_k": top_k}


# In[5]:


root_path = "/home/umni2/a/umnilab/users/xue120/umni4/2023_web_mobility_summer"+\
                "/1_data_check/data_feature_generation/"
file_name = root_path + "feature_" + str(x_day) + "_" + str(y_day)
train_path = file_name + "/train.json"
vali_path = file_name + "/validate.json"
test_path = file_name + "/test.json"
sampled_user_location_path = file_name + "/sampled_user_location.json"
member_path = root_path + "member/"


# # 2: visualization

# In[6]:


def visual_train_loss(e_losses):
    plt.figure(figsize=(4,3), dpi=300)
    x = range(len(e_losses))
    y1 = copy.copy(e_losses)
    plt.plot(x,y1, linewidth=1,  label="train")
    plt.legend()
    plt.title('Loss decline on training data')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(case + '/' + 'train_loss.png',bbox_inches = 'tight')
    plt.show()
    
def visual_vali_test_loss(recall_vali, recall_test, ndcg_vali, ndcg_test):
    plt.figure(figsize=(4,3), dpi=300)
    x = range(len(recall_vali))    
    plt.plot(x, recall_vali, linewidth=1, label="Recall_validate")
    plt.plot(x, ndcg_vali, linewidth=1, label="NDCG_validate")
    plt.plot(x, recall_test, linewidth=1, label="Recall_test")
    plt.plot(x, ndcg_test, linewidth=1, label="NDCG_test")
    plt.legend()
    plt.title('Recall/NDCG on validate/test sets')
    plt.xlabel('Epoch')
    plt.ylabel('Recall@3, NDCG@3')
    plt.savefig(case + '/' + 'vali_test_recall_ndcg.png',bbox_inches = 'tight')
    plt.show()


# # 3: compute the loss

# In[7]:


#compute the cross entropy loss
#input1:  gnn_output   dim = (batch, y_day, U, V).
#input2:  real_link    dim = (batch, y_day, n_edge, 2).
#input3:  criterion
#inputs 4,5,6:  n_user, n_loc, npr
#output: average loss for batch*y_day terms
def compute_loss(gnn_output, real_link, criterion, n_user, n_loc, npr):
    batch, y_day = gnn_output.size()[0], gnn_output.size()[1]
    loss = torch.tensor([0.0])
    all_edge = [str(u)+"_"+str(v) for u in range(n_user) for v in range(n_loc)]
    
    for i in range(batch):
        for j in range(y_day):
            predicted, real = gnn_output[i][j], real_link[i][j] 
            #positive edges
            str_real = [str(int(real[k][0])) + "_" + str(int(real[k][1])) for k in range(len(real))]
            num_real = len(real) - str_real.count(str(-1)+"_"+str(-1))
            set_pos = set(str_real[0: num_real])
            all_pos = list(set_pos)
            n_pos = len(all_pos)
            
            #sample negative edges
            all_neg = list(set(all_edge) - set_pos)
            n_sampled_neg = int(n_pos * npr)
            sampled_neg = random.sample(all_neg, n_sampled_neg)
            
            #prepare for loss computing
            pos = [[int(item.split("_")[0]), int(item.split("_")[1])] for item in all_pos] 
            neg = [[int(item.split("_")[0]), int(item.split("_")[1])] for item in sampled_neg] 
            pos_idx = [pos[k][0]*n_loc + pos[k][1] for k in range(n_pos)]
            neg_idx = [neg[k][0]*n_loc + neg[k][1] for k in range(n_sampled_neg)]
            hat_1_pos = torch.take(predicted, torch.tensor(pos_idx)) 
            hat_1_neg = torch.take(predicted, torch.tensor(neg_idx))
            hat_1 = torch.sigmoid(torch.cat((hat_1_pos, hat_1_neg)).unsqueeze(dim=0))  
            hat = torch.log(torch.transpose(torch.cat((1.0-hat_1, hat_1), dim=0),1,0))  #NLLLOSS
            
            real = torch.tensor([1]*n_pos + [0]*n_sampled_neg)
            loss += criterion(hat, real)
    loss = loss*1.0/(batch*y_day)
    return loss


# # 4: initialize user embeddings

# In[8]:


#define user embeddings based on POI embeddings.
#input1: x_loc             dim = (V, 200)
#input2: x_mob_batch      dim = (batch, x_day, n_m, 2)
#input3: x_text_batch         dim = (batch, x_day, n_t, 2)
#input4: n_user         
#output: x_user                dim = (batch, U, 200)
def compute_user_embedding(x_loc, x_mob_batch, x_text_batch, n_user):
    x_user = torch.zeros((0, n_user, 200), device=device)
    x_m_t_batch = torch.cat([x_mob_batch, x_text_batch], dim=2)  #dim = (batch, x_day, n_m+n_t, 2)
    batch = x_m_t_batch.size()[0]
    
    for i in range(batch):
        #initialize 
        user_sum_embed = torch.zeros((n_user, 200), device=device)
        user_ave_embed = torch.zeros((n_user, 200), device=device)
        user_count_embed, user_with_edge = [0]*n_user, list()

        #update user embeddings
        link_record = x_m_t_batch[i][0]  #extract the first day
        for link in link_record:
            if link[0] != -1:
                user, loc = link[0], link[1] 
                user_with_edge.append(user)              
                user_count_embed[user] = user_count_embed[user] + 1     
                user_sum_embed[user] = user_sum_embed[user] + x_loc[loc]
            else:
                break
                
        set_user_with_edge = set(user_with_edge)
        for user in set_user_with_edge:
            user_ave_embed[user] = user_sum_embed[user]/user_count_embed[user]
        
        #update the user embedding for other users with mobility records on the first day
        #compute the average embedding  
        n_user_with_edge = len(set_user_with_edge)
        ave_embedding = torch.sum(user_ave_embed, dim=0)/(1.0*n_user_with_edge)
        
        #define the embeddings for remaining users as the average embedding 
        set_remain = set(range(n_user))-set_user_with_edge
        dict_remain = {user:0 for user in set_remain}
        for user in dict_remain:
            user_ave_embed[user] = ave_embedding
        
        #concatenate different batches
        x_user = torch.cat([x_user, user_ave_embed.unsqueeze(0)],dim=0)
    return x_user                                        


# # 5: compute adjacency matrix based on mobility and web search

# In[9]:


#input1: x_mob_batch       dim = (batch, x_day, n_m, 2)
#input2: x_text_batch      dim = (batch, x_day, n_t, 2)
#inputs3,4: u_user, n_loc
#output1: x_adj            dim = (batch, x_day, n_user+2*n_loc, n_user+2*n_loc)
def convert_to_adj(x_mob_batch, x_text_batch, n_user, n_loc):
    time_1 = time.time()
    batch, x_day = x_mob_batch.size()[0], x_mob_batch.size()[1]
    adj_dim = n_user + 2*n_loc
    adj = torch.zeros((batch, x_day, adj_dim, adj_dim), device=device)
    
    for i in range(batch):
        x_mob_record, x_text_record = x_mob_batch[i], x_text_batch[i]
        for j in range(x_day):
            x_mob_one_day, x_text_one_day = x_mob_record[j], x_text_record[j]
            #extract mob edges
            for link in x_mob_one_day:
                if link[0] != -1:
                    user, loc = link[0], link[1]
                    n_idx = n_user + loc
                    adj[i][j][user][n_idx] = adj[i][j][user][n_idx] + 1 
                    adj[i][j][n_idx][user] = adj[i][j][user][n_idx]
                else:
                    break
            
            #extract text edges
            for link in x_text_one_day:
                if link[0] != -1:
                    user, loc = link[0], link[1] 
                    n_idx = n_user + n_loc + loc
                    adj[i][j][user][n_idx] = adj[i][j][user][n_idx] + 1
                    adj[i][j][n_idx][user] = adj[i][j][user][n_idx]
                else:
                    break
    return adj


# # 6: train

# In[10]:


#6.1: one training epoch   
#output: the average loss, model         
def train_epoch(model, opt, criterion, train, hyper_param_dict, y_day, npr, loss_batch_all):
    time_1 = time.time()
    model.train()
    losses = list()
    
    n_user, n_loc, b_s = hyper_param["n_user"], hyper_param["n_loc"], hyper_param["b_s"]
    x_u_v, x_poi, train_x_mob, train_x_text, train_y_mob =\
        train["u_v"].to(device), train["x_poi"].to(device), train["x_mob"],\
            train["x_text"], train["y_mob"]
    n = train_x_mob.size()[0] 
    print ("# batch: ", int(n/b_s))
    
    for i in range(0, n-b_s, b_s):
        time_1 = time.time()          
        x_mob_batch, x_text_batch, y_mob_batch = train_x_mob[i:i + b_s], train_x_text[i:i + b_s], train_y_mob[i:i + b_s]    
    
        opt.zero_grad()
        
        loss = torch.zeros(1, dtype=torch.float)
        x_user = compute_user_embedding(x_poi, x_mob_batch, x_text_batch, n_user)  #4.
        
        x_adj = convert_to_adj(x_mob_batch, x_text_batch, n_user, n_loc)                  #5. 
        
        model_output = model.run(x_u_v, x_poi, x_user.to(device), x_adj.to(device), b_s) 
        
        loss = compute_loss(model_output.cpu(), y_mob_batch, criterion, n_user, n_loc, npr)  #3.
        loss_batch_all.append(loss.data.numpy()[0])
        loss.backward()
        opt.step()
        losses.append(loss.data.numpy())  # sum over batches
        
        time2 = time.time()
        if i%20 == 0:
            print ("i_batch: ", i/b_s)
            print ("the loss is: ", loss.data.numpy()[0])
            print ("time for this batch: ", round(time2 - time_1,3))
            print ("-----------------a batch ends---------------")
    return sum(losses)/float(len(losses)+0.000001), model, loss_batch_all


# In[11]:


#6.2
def train_process(train, vali, test, net, criterion, hyper_param, y_day, loss_batch_all):   
    e_losses_train = list()
    recall_vali, recall_test, ndcg_vali, ndcg_test = list(), list(), list(), list()
    l_r, n_e, b_s  = hyper_param["l_r"], hyper_param["n_e"], hyper_param["b_s"]

    opt = optim.Adam(net.parameters(), l_r, betas = (0.9,0.999), weight_decay = 0.0001)
    opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[150]) 
    print ("# epochs: ", n_e)
    print ("------------------------------------------------------------")        
    time_start = time.time()
    no_improve_in_n = 0
    
    #prepare for vali and test
    print ("start preparing for vali and test")
    vali_u_v, vali_x_poi, vali_x_user, vali_x_adj, vali_y_real = prepare_validate_test(vali, hyper_param)
    print ("finish vali")
    test_u_v, test_x_poi, test_x_user, test_x_adj, test_y_real = prepare_validate_test(test, hyper_param)
    print ("finish test")
           
    for i in range(n_e):
        print ("i_epoch: ", i)
        print ("----------------an epoch starts-------------------")
        time1 = time.time()
        n_train = len(train["x_mob"])
        number_list = copy.copy(list(range(n_train)))
        random.shuffle(number_list, random = r)
        shuffle_idx = torch.tensor(number_list)
        
        #train one epoch
        train_shuffle = dict()
        train_shuffle["u_v"] = train["u_v"]
        train_shuffle["x_poi"], train_shuffle["x_mob"] = train["x_poi"], train["x_mob"][shuffle_idx]
        train_shuffle["x_text"], train_shuffle["y_mob"] = train["x_text"][shuffle_idx], train["y_mob"][shuffle_idx]
        
        loss, net, loss_batch_all =  train_epoch(net, opt, criterion, train_shuffle, hyper_param, y_day, npr, loss_batch_all)
        
        opt_scheduler.step() 
        
        loss = float(loss)
        print ("train loss for this epoch: ", round(loss, 6))
        e_losses_train.append(loss)
        visual_train_loss(e_losses_train)
        
        print ("----------------validate-------------------")
        val_all_recall, val_all_ndcg, val_ave_recall, val_ave_ndcg =\
                validate_test(net, hyper_param, \
                              vali_u_v, vali_x_poi, vali_x_user, vali_x_adj, vali_y_real, False)
        
        print ("----------------test-------------------")
        test_all_recall, test_all_ndcg, test_ave_recall, test_ave_ndcg =\
               validate_test(net, hyper_param,\
                             test_u_v, test_x_poi, test_x_user, test_x_adj, test_y_real, False)
        
        if len(recall_vali) > 0:
            past_max = np.max(recall_vali)
        else:
            past_max = 0.0
        recall_vali.append(val_ave_recall)
        recall_test.append(test_ave_recall)
        ndcg_vali.append(val_ave_ndcg)
        ndcg_test.append(test_ave_ndcg)
        visual_vali_test_loss(recall_vali, recall_test, ndcg_vali, ndcg_test)
        
        #store
        performance = {"recall_val": recall_vali, "recall_test": recall_test, \
                       "ndcg_val": ndcg_vali,"ndcg_test": ndcg_test,\
                        "e_losses_train": e_losses_train}
        subfile =  open(case + '/' + 'performance'+'.json','w')
        json.dump(performance, subfile)
        subfile.close()
                    
        #early stop
        if val_ave_recall < past_max:
            no_improve_in_n = no_improve_in_n + 1
        else:
            no_improve_in_n = 0
        if no_improve_in_n == 30:
            print ("Early stop at the " + str(i+1) + "-th epoch")
            return e_losses_train, net, loss_batch_all
        time2 = time.time()
        print ("running time for this epoch: ", time2 - time1)
        time_now = time.time()
        print ("running time until now: ", time_now - time_start)
        print ("-------------------------an epoch ends ---------------------------")
    return e_losses_train, net, loss_batch_all


# In[12]:


#6.3
def model_train(train, vali, test, hyper_param, x_day, y_day, member):
    with torch.autograd.set_detect_anomaly(True):       
        loss_batch_all = list()
        model = WSBiGNN(hid_dim, hid_dim_cons, x_day, member).to(device)          
        criterion = nn.NLLLoss() 
        print ("start train_process")
        e_losses, trained_model, loss_batch_all = train_process(train, vali, test, model,\
                                                criterion, hyper_param, y_day, loss_batch_all)  
        return e_losses, trained_model, loss_batch_all


# # 7: compute the Recall and NDCG

# In[13]:


#7.1: compute Recall@K, NDCG@K
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
            
            #2. compute Recall@K, NDCG@K
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
                         best_rank = [1.0]*len_real_poi + [0.0]*(top_k-len_real_poi)
                    else:
                         best_rank = [1.0]*top_k        
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

#7.2: evaluate the trained model on validation or test 
def prepare_validate_test(vali_test, hyper_param):
    n_user, n_loc = hyper_param["n_user"], hyper_param["n_loc"]
    u_v, x_poi, x_mob, x_text, y_real =\
        vali_test["u_v"].to(device), vali_test["x_poi"].to(device), vali_test["x_mob"].to(device), \
            vali_test["x_text"].to(device), vali_test["y_mob"]
    x_user = compute_user_embedding(x_poi, x_mob, x_text, n_user)
    x_adj = convert_to_adj(x_mob, x_text, n_user, n_loc)
    return u_v, x_poi, x_user, x_adj, y_real

def validate_test(trained_model, hyper_param, u_v, x_poi, x_user, x_adj, y_real, output=False):  
    n_user, n_loc = hyper_param["n_user"], hyper_param["n_loc"]
    top_k, b_s = hyper_param["top_k"], y_real.size()[0]
    
    y_hat = trained_model.run(u_v, x_poi, x_user, x_adj, b_s)
    all_recall, all_ndcg, ave_recall, ave_ndcg =\
        compute_recall_ndcg(y_hat.cpu(), y_real, n_user, n_loc, top_k)
    
    if output == True:
        return all_recall, all_ndcg, ave_recall, ave_ndcg, y_hat.cpu(), y_real
    else:
        return all_recall, all_ndcg, ave_recall, ave_ndcg


# # 8: train

# In[14]:


#8.1: tensorize
def tensorize(train_vali_test):
    result = dict()
    result["u_v"] = torch.tensor(train_vali_test["u_v"]) 
    result["x_poi"] = torch.tensor(train_vali_test["x_poi"])     
    result["x_mob"] = torch.tensor(train_vali_test["x_mob"]) 
    result["x_text"] = torch.tensor(train_vali_test["x_text"]) 
    result["y_mob"] = torch.tensor(train_vali_test["y_mob"]) 
    return result


# In[15]:


#8.2: load the data
train = tensorize(json.load(open(train_path)))
vali = tensorize(json.load(open(vali_path)))
test = tensorize(json.load(open(test_path)))
sampled_user_location = json.load(open(sampled_user_location_path))
sampled_user_location["n_user"] = len(sampled_user_location["u"])
sampled_user_location["n_loc"] = len(sampled_user_location["p"])

u_list, p_list = sampled_user_location["u"], sampled_user_location["p"]
hyper_param["n_user"], hyper_param["n_loc"] = len(u_list), len(p_list)

#supernode
member_dict = json.load(open(member_path + "member_" + case + ".json"))
#sg, s_ng, ns_g, ns_ng
member = torch.tensor([member_dict["s_g"], member_dict["s_ng"],\
                       member_dict["ns_g"], member_dict["ns_ng"]], device=device)


# In[16]:


#8.3: model 
e_losses, trained_model, loss_batch_all = model_train(train, vali, test, hyper_param,\
                                                      x_day, y_day, member)


# In[17]:


print ("start preparing for vali and test")
vali_u_v, vali_x_poi, vali_x_user, vali_x_adj, vali_y_real = prepare_validate_test(vali, hyper_param)
test_u_v, test_x_poi, test_x_user, test_x_adj, test_y_real = prepare_validate_test(test, hyper_param)


# In[18]:


#8.4: validate and test the model
print ("---------------validation-------------------")
all_recall, all_ndcg, ave_recall, ave_ndcg, vali_output, vali_real =\
    validate_test(trained_model, hyper_param, vali_u_v, vali_x_poi, vali_x_user, vali_x_adj, vali_y_real, True)
print ("-----------finish model validation---------------")


# In[19]:


print ("---------------test-------------------")
all_recall, all_ndcg, ave_recall, ave_ndcg, test_output, test_real =\
    validate_test(trained_model, hyper_param, test_u_v, test_x_poi, test_x_user, test_x_adj, test_y_real, True)
print ("-----------finish model validation---------------")


# # 9: save

# In[20]:


list_vali_hat = vali_output.cpu().detach().numpy().tolist()
list_vali_real = vali_real.cpu().detach().numpy().tolist()
print(len(list_vali_hat))
print(len(list_vali_hat[0]))
print(len(list_vali_hat[0][0]))
print(len(list_vali_hat[0][0][0]))

list_test_hat = test_output.cpu().detach().numpy().tolist()
list_test_real = test_real.cpu().detach().numpy().tolist()
print(len(list_test_hat))
print(len(list_test_hat[0]))
print(len(list_test_hat[0][0]))
print(len(list_test_hat[0][0][0]))


# In[21]:


result = {"vali_hat": list_vali_hat, "vali_real": list_vali_real, \
          "test_hat": list_test_hat, "test_real": list_test_real}

subfile = case+'/vali_predict.json'
savefile = open(subfile,'w')
json.dump(result, savefile)
savefile.close()

df = json.load(open(subfile))
df.keys()
print(len(df["test_real"]))


# In[ ]:




