#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import json
import numpy as np
import pandas as pd


# In[2]:


#This code samples users and POIs.
#step 1. read data; aggregate daily data.
#step 2. sample users and POIs with sufficient number of mobility and web search records.
#step 3. delete frequent edges implying the trips made to the workplace.
#step 4. save files.

#Input:
#1. mobility data. 20191001_network_mobility.json
#2. text data. 20191001_network_text.json
#Output:
#1. 1827_user_80_poi.json


# # step 1: read data; aggregate daily data

# In[3]:


root_path = "/home/umni2/a/umnilab/users/xue120/umni4/2023_web_mobility/"
mobility_path, text_path = root_path + "1_data/mobility/", root_path + "1_data/text/"

#mobility data: {hour: {min/5: {poi_id: {user_id: 5, user_id: 3, …}}}}
#1. hour: 0, 1, 2, …, 23
#2. min/5: 0, 1, 2, …, 11
#3. poi_id: 1, 2, … (right node)
#4. user_id: 1, 2, … (left node)
#text data:  {hour: {min/5: {poi_id: {user_id: 5, user_id: 3, …}}}}
#1. hour: 0, 1, 2, …, 23
#2. min/5: 0, 1, 2, …, 11
#3. poi_id: 1, 2, …
#4. user_id: 1, 2, …


# # 1.1: read mobility data

# In[4]:


mobility_day = dict()  #{"20191001": {loc: [user1, user2,...],...}}

mobilityNameList = os.listdir(mobility_path)
mobilityNameList.sort()
print (len(mobilityNameList))

for i in range(len(mobilityNameList)):
    file_name = mobilityNameList[i]
    if "20" in file_name:
        day = (file_name.split("_")[0])   #20191001
        file_path = mobility_path + file_name   #/20191001_network_mobility.json
        f = open(file_path,)
        df_file = json.load(f)
        f.close()
        
        location_user_day = dict()                   #{loc: [user1, user2,...],...}
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
        location_edge_count_day = [len(location_user_day[location])\
                             for location in location_user_day]
        if i%30 == 0:       
            print ("today is: ", day)
            print ("#valid location on one day: ", len(location_user_day))
            print ("#unrepeated edges on one day: ", np.sum(location_edge_count_day))
        mobility_day[day] = location_user_day             


# In[5]:


ave_mobility_edge = np.mean([np.sum([len(mobility_day[day][loc]) for loc in mobility_day[day]])\
                                     for day in mobility_day])
print (ave_mobility_edge)


# # 1.2: read text data

# In[6]:


text_day = dict()  #{"20191001": {loc: [user1, user2,...],...}}

textNameList = os.listdir(text_path)
textNameList.sort()
print (len(textNameList))

for i in range(len(textNameList)):
    file_name = textNameList[i]
    if "20" in file_name:
        day = (file_name.split("_")[0])   #20191001
        file_path = text_path + file_name   #/20191001_network_text.json
        f = open(file_path,)
        df_file = json.load(f)
        f.close()
        
        location_user_day = dict()                   #{loc: [user1, user2,...],...}
        
        for key1 in df_file:                         #hour:  0,1,2,...,23
            for key2 in df_file[key1]:               #min: 0,1,2,...,11 
                for key3 in df_file[key1][key2]:     #location: 1,2,3,...,100
                    if key3 != str(0):
                        text_data = df_file[key1][key2][key3]  #{user_id_1: 5, user_id_2: 3, …}
                        if len(text_data) > 0:
                            user_list = list(text_data.keys())  #[user_id_1, user_id_2]
                            
                            if key3 not in location_user_day:
                                location_user_day[key3] = user_list
                            else:
                                location_user_day[key3] = list(set(location_user_day[key3])\
                                                               .union(set(user_list)))
        location_edge_count_day = [len(location_user_day[location]) for location in location_user_day]
        if  i%30==0:       
            print ("today is: ", day)
            print ("#valid location on one day: ", len(location_user_day))
            print ("#unrepeated edges on one day: ", np.sum(location_edge_count_day))
        text_day[day] = location_user_day             


# In[7]:


ave_text_edge = np.mean([np.sum([len(text_day[day][loc]) for loc in text_day[day]]) for day in text_day])
print (ave_text_edge)


# # step 2: sample users with #mob, #web >= 50, 5, and POIs with #mob, #web >= 500, 50

# In[8]:


def sample_entity(ul_mob_text, mob_thres, text_thres):
    sampled_entity = list()
    for key in ul_mob_text:
        mob_text_count = ul_mob_text[key]
        mob_count, text_count = mob_text_count[0], mob_text_count[1]
        if mob_count >= mob_thres and text_count >= text_thres:
            sampled_entity.append(key)
    return sampled_entity


# # 2.1: compute #mob, #web for users and POIs 

# In[9]:


user_mob_text = dict()        #{user: [#mob, #web],...}  
loc_mob_text = dict()        #{loc: [#mob, #web],...}  

for i in range(len(mobilityNameList)):
    file_name = mobilityNameList[i]
    if "20" in file_name:
        day = (file_name.split("_")[0])        #20191001
        mobility_record = mobility_day[day]    #{loc1: [ user1, user2,...],...}
        text_record = text_day[day]            #{loc1: [ user1, user2,...],...}
        for loc in mobility_record:
            for user in mobility_record[loc]:
                
                #record mobility data on users
                if user not in user_mob_text:
                    user_mob_text[user] = [1, 0]  #[#mob, #web]
                else:
                    user_mob_text[user][0] = user_mob_text[user][0] + 1
                    
                #record mobility data on locs
                if loc not in loc_mob_text:
                    loc_mob_text[loc] = [1, 0]
                else:
                    loc_mob_text[loc][0] = loc_mob_text[loc][0] + 1
                    
        for loc in text_record:
            for user in text_record[loc]:
                
                #record text data on users
                if user not in user_mob_text:
                    user_mob_text[user] = [0, 1]
                else:
                    user_mob_text[user][1] = user_mob_text[user][1] + 1  
                
                #record text data on locations
                if loc not in loc_mob_text:
                    loc_mob_text[loc] = [0, 1]
                else:
                    loc_mob_text[loc][1] = loc_mob_text[loc][1] + 1


# # 2.2: sample users

# In[10]:


print ("#users", len(user_mob_text))
print ("mean mobility count", np.mean([user_mob_text[user][0] for user in user_mob_text]))
print ("mean web count", np.mean([user_mob_text[user][1] for user in user_mob_text]))
print ("total mobility count", np.sum([user_mob_text[user][0] for user in user_mob_text]))
print ("total web count", np.sum([user_mob_text[user][1] for user in user_mob_text]))


# In[11]:


print ("# sampled user", len(sample_entity(user_mob_text, 10, 1)))
print ("# sampled user", len(sample_entity(user_mob_text, 20, 2)))
print ("# sampled user", len(sample_entity(user_mob_text, 30, 3)))
print ("# sampled user", len(sample_entity(user_mob_text, 40, 4)))
print ("# sampled user", len(sample_entity(user_mob_text, 50, 5)))
print ("# sampled user", len(sample_entity(user_mob_text, 80, 8)))
sampled_user = sample_entity(user_mob_text, 50, 5)       #Threshold 1


# # 2.2: sample POIs

# In[12]:


print ("#users", np.mean([loc_mob_text[user][0] for user in loc_mob_text]))
print ("mean mobility count", np.mean([loc_mob_text[user][0] for user in loc_mob_text]))
print ("mean text count", np.mean([loc_mob_text[user][1] for user in loc_mob_text]))
print ("total mobility count", np.sum([loc_mob_text[key][0] for key in loc_mob_text]))
print ("total text count", np.sum([loc_mob_text[key][1] for key in loc_mob_text]))


# In[13]:


print ("# sampled loc", len(sample_entity(loc_mob_text, 10, 1)))
print ("# sampled loc", len(sample_entity(loc_mob_text, 20, 2)))
print ("# sampled loc", len(sample_entity(loc_mob_text, 30, 3)))
print ("# sampled loc", len(sample_entity(loc_mob_text, 500, 50)))
print ("# sampled loc", len(sample_entity(loc_mob_text, 5000, 500)))
sampled_loc = sample_entity(loc_mob_text, 500, 50)             #Threshold 2
print (len(sampled_loc))


# In[14]:


print ("#sampled_user", len(sampled_user))
print ("#sampled_loc", len(sampled_loc))


# # step 3: delete frequent edges implying the trips made to the workplace.

# In[15]:


user_loc_mob_count = dict()  #{user: {loc1: 15, ...},...}

for i in range(len(mobilityNameList)):
    file_name = mobilityNameList[i]
    if "20" in file_name:
        day = (file_name.split("_")[0])        #20191001
        mob_record = mobility_day[day]    #{"13":["123", "456",...],...}
        
        for loc in mob_record:
            for user in mob_record[loc]:
                
                if user not in user_loc_mob_count :
                    user_loc_mob_count[user] = {loc: 1}
                elif loc not in user_loc_mob_count [user]:
                    user_loc_mob_count[user][loc] = 1
                else:
                    user_loc_mob_count[user][loc] = user_loc_mob_count[user][loc] + 1


# In[16]:


user_loc_frequent = dict()    #{user: [loc1, loc2, ...],...}       
for user in sampled_user:
    for loc in user_loc_mob_count[user]:
        if user_loc_mob_count[user][loc] > 183.0/7:      #Threshold 3
            if user not in user_loc_frequent:
                user_loc_frequent[user] = [loc]
            else:
                user_loc_frequent[user].append(loc)
print ("#user with a frequent loc", len(user_loc_frequent))


# # step 4: save files

# In[17]:


n_user, n_loc, n_fre = len(sampled_user), len(sampled_loc), len(user_loc_frequent)
print (n_user, n_loc, n_fre)

#file
user_loc = {"u": sampled_user, "p": sampled_loc, "frequent_u_p": user_loc_frequent}

#location
user_loc_name =  str(n_user) + "_user_" + str(n_loc) + "_poi" + ".json"
user_loc_file = open(user_loc_name, 'w')

#save
json.dump(user_loc, user_loc_file)
user_loc_file.close()


# In[ ]:




