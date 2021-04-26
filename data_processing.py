
import pandas as pd
from prophet import Prophet
#from fbprophet import Prophet

df = pd.read_csv("Twitter_May_Aug_2014_TerrorSecurity_resolved.txt", sep=" ", header=None)
df.columns = ['dateTime', 'mentionedEntity1', 'mentionedEntity2']
df

from sklearn.preprocessing import LabelEncoder


mask = (df['dateTime'] > '05:02:2014:20:15:22') & (df['dateTime'] <= '05:31:2014:23:59:00')
df_may_only = df.loc[mask]
df_may_only


#label encoding first entity
pd.options.mode.chained_assignment = None
le = LabelEncoder()
df_may_only.iloc[:,1] = le.fit_transform(df_may_only.iloc[:,1])
df_may_only.head()

#label encoding first entity
le = LabelEncoder()
df_may_only.iloc[:,2] = le.fit_transform(df_may_only.iloc[:,2])
df_may_only.head()


import numpy as np
entity1_array = df_may_only[['mentionedEntity1']].to_numpy()
entity1_array = entity1_array.flatten()
entity1_array

#random entities that were generated 4/25 are saved in random_entities.txt
random_entity1_array = np.loadtxt("random_entities.txt").reshape(5000)

temp_df1 = pd.DataFrame(columns=['dateTime','mentionedEntity1'])
data = []

for entity in df_may_only.iterrows():
    if entity[1][1] in random_entity1_array:
        data.append((entity[1][0], entity[1][1]))

temp_df2 = pd.DataFrame(data , columns=['dateTime','mentionedEntity1'])
may_random = pd.concat([temp_df1, temp_df2], ignore_index=True)

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


may_random['dateTime'] = pd.to_datetime(may_random['dateTime'], format="%m:%d:%Y:%H:%M:%S")

# change dateTime column to just dates for visualizations
may_random['dateTime'] = may_random['dateTime'].dt.date

random_entity1_ct_df = (may_random.reset_index()
          .groupby(['dateTime','mentionedEntity1'], as_index=False)
          .count()
          # rename isn't strictly necessary here, it's just for readability
          .rename(columns={'index':'ct'})
       )



temp_df1 = pd.DataFrame(columns=['dateTime','ct'])
data = []

for entity in random_entity1_ct_df.iterrows():
    if entity[1][1] == 383:
        data.append((entity[1][0], entity[1][2]))

temp_df2 = pd.DataFrame(data , columns=['dateTime','ct'])
entity383_df = pd.concat([temp_df1, temp_df2], ignore_index=True)


# In[17]:


import datetime
missing_dates = [(datetime.date(2014, 5, 15), 0), (datetime.date(2014, 5, 16), 0), 
                 (datetime.date(2014, 5, 17), 0), (datetime.date(2014, 5, 18), 0),
                 (datetime.date(2014, 5, 19), 0), (datetime.date(2014, 5, 20), 0),
                 (datetime.date(2014, 5, 21), 0), (datetime.date(2014, 5, 22), 0),
                 (datetime.date(2014, 5, 23), 0), (datetime.date(2014, 5, 24), 0),
                 (datetime.date(2014, 5, 25), 0), (datetime.date(2014, 5, 26), 0),
                 (datetime.date(2014, 5, 27), 0), (datetime.date(2014, 5, 28), 0),
                 (datetime.date(2014, 5, 29), 0), (datetime.date(2014, 5, 30), 0),
                 (datetime.date(2014, 5, 31), 0)]

temp_df3 = pd.DataFrame(missing_dates , columns=['dateTime','ct'])
entity383_df = pd.concat([entity383_df, temp_df3], ignore_index=True)

temp_df1 = pd.DataFrame(columns=['dateTime','ct'])
data = []

for entity in random_entity1_ct_df.iterrows():
    if entity[1][1] == 8101:
        data.append((entity[1][0], entity[1][2]))

temp_df2 = pd.DataFrame(data , columns=['dateTime','ct'])
entity8101_df = pd.concat([temp_df1, temp_df2], ignore_index=True)

missing_dates = [(datetime.date(2014, 5, 12), 0), (datetime.date(2014, 5, 16), 0), 
                 (datetime.date(2014, 5, 17), 0), (datetime.date(2014, 5, 18), 0),
                 (datetime.date(2014, 5, 19), 0), (datetime.date(2014, 5, 20), 0),
                 (datetime.date(2014, 5, 21), 0), (datetime.date(2014, 5, 22), 0)]

temp_df3 = pd.DataFrame(missing_dates , columns=['dateTime','ct'])
entity8101_df = pd.concat([entity8101_df, temp_df3], ignore_index=True)

entity8101_df = entity8101_df.sort_values(by='dateTime')

temp_df1 = pd.DataFrame(columns=['dateTime','ct'])
data = []

for entity in random_entity1_ct_df.iterrows():
    if entity[1][1] == 24742:
        data.append((entity[1][0], entity[1][2]))

temp_df2 = pd.DataFrame(data , columns=['dateTime','ct'])
entity24742_df = pd.concat([temp_df1, temp_df2], ignore_index=True)

missing_dates = [(datetime.date(2014, 5, 12), 0), (datetime.date(2014, 5, 15), 0), 
                 (datetime.date(2014, 5, 16), 0), (datetime.date(2014, 5, 17), 0),
                 (datetime.date(2014, 5, 18), 0), (datetime.date(2014, 5, 19), 0),
                 (datetime.date(2014, 5, 23), 0), (datetime.date(2014, 5, 24), 0),
                 (datetime.date(2014, 5, 26), 0)]

temp_df3 = pd.DataFrame(missing_dates , columns=['dateTime','ct'])
entity24742_df = pd.concat([entity24742_df, temp_df3], ignore_index=True)

entity24742_df = entity24742_df.sort_values(by='dateTime')
