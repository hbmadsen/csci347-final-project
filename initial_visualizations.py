#!/usr/bin/env python
# coding: utf-8

# Both prophet and plotly need to be installed on machine.

# In[1]:


import pandas as pd
from prophet import Prophet
#from fbprophet import Prophet


# In[2]:


df = pd.read_csv("Twitter_May_Aug_2014_TerrorSecurity_resolved.txt", sep=" ", header=None)
df.columns = ['dateTime', 'mentionedEntity1', 'mentionedEntity2']
df


# In[3]:


from sklearn.preprocessing import LabelEncoder


# Just going to observe anomalies in May, since there are so many data points.

# In[4]:


mask = (df['dateTime'] > '05:02:2014:20:15:22') & (df['dateTime'] <= '05:31:2014:23:59:00')
df_may_only = df.loc[mask]
df_may_only


# In[5]:


#label encoding first entity
pd.options.mode.chained_assignment = None
le = LabelEncoder()
df_may_only.iloc[:,1] = le.fit_transform(df_may_only.iloc[:,1])
df_may_only.head()


# In[6]:


#label encoding first entity
le = LabelEncoder()
df_may_only.iloc[:,2] = le.fit_transform(df_may_only.iloc[:,2])
df_may_only.head()


# In[7]:


import numpy as np
entity1_array = df_may_only[['mentionedEntity1']].to_numpy()
entity1_array = entity1_array.flatten()
entity1_array


# In[8]:


random_entity1_array = [np.random.choice(len(entity1_array), size=20, replace=False)]
random_entity1_array = np.asarray(random_entity1_array)
random_entity1_array = random_entity1_array.flatten()


# In[10]:


#for entity in df.iterrows():
    #print(entity[1])


# In[11]:


df_may_only['dateTime'] = pd.to_datetime(df_may_only['dateTime'], format="%m:%d:%Y:%H:%M:%S")

# change dateTime column to just dates for visualizations
df_may_only['dateTime'] = df_may_only['dateTime'].dt.date

entity1_ct_df = (df_may_only.reset_index()
          .groupby(['dateTime','mentionedEntity1'], as_index=False)
          .count()
          # rename isn't strictly necessary here, it's just for readability
          .rename(columns={'index':'ct'})
       )

entity1_ct_df.pop("mentionedEntity2")
entity1_ct_df


# In[13]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))

count = 1
# key gives the group name (i.e. category), data gives the actual values
for key, data in entity1_ct_df.groupby('mentionedEntity1'):
    data.plot(x='dateTime', y='ct', ax=ax)
    #print(data.index)
    if count == 200:
        break
    count = count + 1
    
#adding median dailing count for all entities
entity1_ct_df.groupby(entity1_ct_df["dateTime"])["ct"].median().plot(kind='line',
                                                                     rot=0, ax=ax, linewidth=6, color='red')

ax.get_legend().remove()


# #### Where to go from here
# 1. Make plots of current data
# 2. Run prophet and detect anomalies
# 3. Verify/Evaluate results
