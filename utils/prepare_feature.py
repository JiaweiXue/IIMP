#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import json
import time
import numpy as np
import pandas as pd
import sys
from sys import getsizeof
from heapq import nlargest


# In[2]:


# This code prepares features and labels for the IIMP.
# step 1. set parameters.
# step 2. read data.
# step 3. prepare features and labels.
# step 4. main.
# step 5. save.

# Output:
#1. train.json
#2. validate.json
#3. test.json
#4. sampled_user_location.json

# Each file:
# u_v: 1827, 80
# x_location: (80, 200)
# x_mobility: (126, 1/2/3/4/5/7/14 * edge_m * 2)
# x_text: (126, 1/2/3/4/5/7/14 * edge_t * 2)
# y_mobility: (126, 1, edge_m, 2)


# # 1: set parameters

# In[3]:


root_path = "/home/umni2/a/umnilab/users/xue120/umni4/2023_web_mobility/"
mobility_path = root_path + "1_data/mobility/"
text_path =  root_path + "1_data/text/"
comment_path = root_path + "1_data/comment/comment20220420.json"
user_poi_path = "1827_user_80_poi.json"

x_day, y_day = 5, 1   #we use x_day's features to predict y_day's labels
train_ratio, validate_ratio = 0.70, 0.10


# # 2: read data

# In[4]:


#read mobility and text data as tensors.
#all_u_v.         dim: 2.   U=1827, V=80.
#all_location.    dim: (V, 200).
#all_mobility.    dim: (n_day, n_m, 2).
#all_text.        dim: (n_day, n_t, 2).
#all_n_mobility.  dim: 1.   # mobility edges.
#all_n_text.      dim: 1.   # text edges.


# In[5]:


#function 2.1: read user/poi data
#input: path   #output: user_poi
##user_poi["u"] = ["12345", ...]  1827 users
##user_poi["p"] = ["12", ...]     80 pois
#sorted_user = {123: 0, 1234: 1, ...}
#sorted_poi = {12: 0, ...}
def read_user_poi(user_poi_path):
    with open(user_poi_path, 'r') as f:
        user_poi = json.load(f)
    print ("# user:", len(user_poi["u"]), "# poi:", len(user_poi["p"]))
    
    u_list = [int(user) for user in user_poi["u"]]
    u_list.sort()
    sorted_user = {u_list[i]: i for i in range(len(u_list))}  #{123: 0, 456: 1, ...}
    
    p_list = [int(poi) for poi in user_poi["p"]]
    p_list.sort() 
    sorted_poi = {p_list[j]:j for j in range(len(p_list))}   #{12: 0, 45: 1, ...}
    return user_poi, sorted_user, sorted_poi

#function 2.2: read mobility/text data
#Output
#output1: mt_result = {"20191001": [12345, 2],...  } 
#output2: max_num. the maximal # edges across all days.
def read_mob_or_text(mt_path, sampled_user_poi,\
                          sorted_user, sorted_poi, user_poi_path, is_mobility):
    daily_num = list()
    mt_result = dict() #mt: mobility, or text
    mt_name_list = os.listdir(mt_path)
    mt_name_list.sort()
    
    with open(user_poi_path, 'r') as f:
        user_poi = json.load(f)
    frequent_user_loc = user_poi["frequent_u_p"]  
        #{'625734': ['1', '33'], '134256': ['1', '30', '5'], ...}
    
    print ("# row: ", len(mt_name_list))
    for i in range(len(mt_name_list)):
        start1 = time.time()
        mt_day_result = list()       #output of mobility, text on one day
        file_name = mt_name_list[i]  #20191001_network_mobility_hour_location.json
        
        if "20" in file_name: 
            day = (file_name.split("_")[0])  #20191001
            if i % 10 == 0:
                print ("day: ", day)
            file_path = mt_path + '/' + file_name    
            f = open(file_path,)
            df_file = json.load(f)           
            f.close() 
            
            location_user_day = dict()                   #{user:[loc1, loc2,...],...}

            for key1 in df_file:                         #hour:  0,1,2,...,23
                for key2 in df_file[key1]:               #min/5: 0,1,2,...,11 
                    for key3 in df_file[key1][key2]:     #location: 1,2,3,...,100
                        if key3 != str(0):
                            trip_data = df_file[key1][key2][key3]  #{user_id_1: 5, user_id_2: 3, …}
                            if len(trip_data) > 0:
                                user_list = list(trip_data.keys())  #[user_id_1, user_id_2]

                                if key3 not in location_user_day:
                                    location_user_day[key3] = user_list
                                else:
                                    location_user_day[key3] = list(set(location_user_day[key3])\
                                                                   .union(set(user_list)))
            mt_result_day = list()
            for loc in location_user_day:
                for user in location_user_day[loc]:
                    if user in sampled_user_poi["u"] and loc in sampled_user_poi["p"]:   #sample      
                        if is_mobility == True:
                            if user in frequent_user_loc:         #frequent
                                if loc not in frequent_user_loc[user]:
                                    mt_result_day.append([sorted_user[int(user)], sorted_poi[int(loc)]])
                            else:
                                mt_result_day.append([sorted_user[int(user)], sorted_poi[int(loc)]])
                        else:
                            mt_result_day.append([sorted_user[int(user)], sorted_poi[int(loc)]])

            mt_result[day] = mt_result_day
            daily_num.append(len(mt_result_day))
        
    max_num = np.max(daily_num)
    print ("max_num", max_num)
    return mt_result, max_num  

#function 2.3: read comment data
#output: comment_result = {"23": [1.1, 1.2, ...,], ...}
def read_comment(comment_path, sampled_user_location):
    comment_result = dict()
    f = open(comment_path,)
    df_file = json.load(f)   
    f.close()
    for loc in sampled_user_location["p"]:
        comment_result[loc] = df_file[str(loc)][2]
    print ("# commented poi == 80?", len(comment_result))
    return comment_result


# # 3: prepare features and labels

# In[6]:


#function 3.1: ensemble the data
#input1: all_mobility = {"20191001": [1826, 79],...  }
#input2: all_text = {"20191001": [1826, 79],...  }
#input3: all_comment = {"100": [1.1, 1.2, ...,], ...}
#input4: sampled_user_location = {"u": ["123","456",...], "p":["1","3",...,"100"]}
#input5: max_n_mob: # mobility edges
#input6: max_n_text: # text edges

#output1: col_u_v: 2. [u, v] = [1827, 80]
#output2: col_location: (v, 200)                    
#output3: col_mobility: (180, n_m, 2)       
#output4: col_text: (180, n_t, 2)
#output5: col_n_mobility. 180.
#output6: col_n_text. 180.
def ensemble(all_mobility, all_text, all_comment,\
             sampled_user_location, max_n_mob, max_n_text):
    
    #extract user and location
    user_str, location_str = sampled_user_location["u"], sampled_user_location["p"]
    user_list = [int(user) for user in user_str]
    location_list = [int(loc) for loc in location_str]
    location_list.sort()
    
    #extract dates
    date_list = list(all_mobility.keys())  #["20191001":123, ...]
    date_list.sort()
    #define outputs
    col_u_v, col_location, col_mob, \
        col_text, col_n_mob, col_n_text =\
            list(), list(), list(), list(), list(), list()

    col_u_v = [len(user_list), len(location_list)]                     #term 1
    col_location = [all_comment[str(loc)] for loc in location_list]   #term 2 (V, 200)
    for i in range(len(date_list)):                   
        date = date_list[i]
        if date in all_mobility:
            #term 3: all_mobility: (180, n_m, 2)
            col_mob_oneday = [item for item in all_mobility[date]]
            n_remain = max_n_mob - len(col_mob_oneday)
            if n_remain > 0:
                col_mob_oneday += [[-1,-1] for j in range(n_remain)]
            col_mob.append(col_mob_oneday)  #add one day
            col_n_mob.append(max_n_mob - n_remain)

            #term 4: all_text (180, n_t, 2)
            col_text_oneday = [item for item in all_text[date]]
            n_remain = max_n_text - len(col_text_oneday)
            if n_remain > 0:
                col_text_oneday += [[-1,-1] for j in range(n_remain)]
            col_text.append(col_text_oneday)
            col_n_text.append(max_n_text - n_remain)
    return col_u_v, col_location, col_mob, col_text, col_n_mob, col_n_text

#function 3.2: split the data into train, validate, and test
def split_data(final_x, idx_1, idx_2, idx_3):
    final_x_1 = [final_x[idx] for idx in idx_1]
    final_x_2 = [final_x[idx] for idx in idx_2]
    final_x_3 = [final_x[idx] for idx in idx_3]
    return final_x_1, final_x_2, final_x_3

#output: train, validate, test
#output1: train["u_v"]               2
#output2: train["x_location"]        (V, 200)
#output3: train["x_mobility"]        (150, x_day, n_m, 2)
#output4: train["x_text"]            (150, x_day, n_t, 2)
#output5: train["y_mobility"]        (150, y_day, n_m, 2)
def split(col_u_v, col_poi, col_mob, col_text, col_n_mob, col_n_text,\
               train_ratio, validate_ratio, x_day, y_day):
    final_x_u_v, final_x_poi, final_x_mob, final_x_text, final_y_mob = list(), list(), list(), list(), list()
    print ("len_final_col_mob", len(col_mob)) 
    #2, (V, 200), (180, x_days, n_m, 2), (180, x_days, n_t, 2), (180, y_days, n_m, 2)
    
    train_x_u_v, val_x_u_v, test_x_u_v = list(), list(), list()
    train_x_poi, val_x_poi, test_x_poi = list(), list(), list()
    train_x_mob, val_x_mob, test_x_mob = list(), list(), list()
    train_x_text, val_x_text, test_x_text = list(), list(), list()
    train_y_mob, val_y_mob, test_y_mob = list(), list(), list()
    
    #step 1
    final_x_u_v, final_x_poi = col_u_v, col_poi   #[1827, 80] #[[1.1, 1.2, ...,], ...]
    
    num_day = len(col_mob) - x_day - y_day + 1
    for i in range(num_day):                         
        final_x_mob.append([col_mob[i+j] for j in range(x_day)]) 
        final_x_text.append([col_text[i+j] for j in range(x_day)])         
        final_y_mob.append([col_mob[i+x_day+j] for j in range(y_day)])
    print ("len_final_x_mob", len(final_x_mob)) 
    
    #step 2
    num_train, num_val =\
        round(num_day*train_ratio), round(num_day*validate_ratio)
    num_test = num_day - num_train - num_val 
    idx_train, idx_val, idx_test = [j for j in range(num_train)], [j+num_train for j in range(num_val)],\
        [j+num_train+num_val for j in range(num_test)]
    
    train_x_u_v, val_x_u_v, test_x_u_v = final_x_u_v, final_x_u_v, final_x_u_v
    train_x_poi, val_x_poi, test_x_poi = final_x_poi, final_x_poi, final_x_poi
    
    train_x_mob, val_x_mob, test_x_mob = split_data(final_x_mob, idx_train, idx_val, idx_test)     
    train_x_text, val_x_text, test_x_text = split_data(final_x_text, idx_train, idx_val, idx_test)     
    train_y_mob, val_y_mob, test_y_mob = split_data(final_y_mob, idx_train, idx_val, idx_test) 
    
    train = {"u_v":train_x_u_v, "x_poi": train_x_poi, "x_mob": train_x_mob, "x_text": train_x_text, "y_mob": train_y_mob} 
    validate = {"u_v": val_x_u_v, "x_poi": val_x_poi, "x_mob": val_x_mob, "x_text": val_x_text, "y_mob": val_y_mob} 
    test = {"u_v": test_x_u_v, "x_poi": test_x_poi, "x_mob": test_x_mob, "x_text": test_x_text, "y_mob": test_y_mob} 
    return train, validate, test

#function 3.3: read the data into train, validate, and test
#output: train, validate, test
#output1: train["u_v"]          2
#output2: train["x_poi"]        (V, 200)
#output3: train["x_mob"]        (150, x_day, n_m, 2)
#output4: train["x_text"]       (150, x_day, n_t, 2)
#output5: train["y_mob"]        (150, y_day, n_m, 2)
def read_data(user_poi_path, mob_file_path, text_file_path, comment_file_path, train_ratio, validate_ratio):
    #2.1
    sampled_user_poi, sorted_user, sorted_poi = read_user_poi(user_poi_path)                   
    #2.2  
    all_mob, max_n_mob = read_mob_or_text(mob_file_path, sampled_user_poi, sorted_user, sorted_poi, user_poi_path, True)  
    print ("max n mobility", max_n_mob) 
    #2.2  
    all_text, max_n_text = read_mob_or_text(text_file_path, sampled_user_poi, sorted_user, sorted_poi, user_poi_path, False) 
    print ("max n text", max_n_text)
    #2.3 
    all_comment = read_comment(comment_file_path, sampled_user_poi) 
    
    #3.1
    col_u_v, col_poi, col_mob, col_text, col_n_mob, col_n_text =\
        ensemble(all_mob, all_text, all_comment, sampled_user_poi, max_n_mob, max_n_text)           
     
    #3.2
    train, validate, test = split(col_u_v, col_poi, col_mob, col_text, col_n_mob, col_n_text,\
                   train_ratio, validate_ratio, x_day, y_day)              
    return train, validate, test, sampled_user_poi


# # 4: main

# In[7]:


time_1 = time.time()
train, validate, test, sampled_user_location = read_data(user_poi_path, mobility_path, text_path,\
              comment_path, train_ratio, validate_ratio)
time_2 = time.time()
print ("running time", time_2 - time_1)


# # 5: save

# In[8]:


#save train
train_file = open("feature_" + str(x_day) + "_" + str(y_day) +"/" + "train" +".json",'w')
json.dump(train, train_file)
train_file.close()
#save validate
validate_file = open("feature_" + str(x_day) + "_" + str(y_day) +"/" + "validate" +".json",'w')
json.dump(validate, validate_file)
validate_file.close()
#save test
test_file = open("feature_" + str(x_day) + "_" + str(y_day) +"/" + "test" +".json",'w')
json.dump(test, test_file)
test_file.close()
#save user_location
sampled_user_location_file = open("feature_" + str(x_day) + "_" + str(y_day) +"/" + "sampled_user_location" +".json",'w')
json.dump(sampled_user_location, sampled_user_location_file)
sampled_user_location_file.close()

