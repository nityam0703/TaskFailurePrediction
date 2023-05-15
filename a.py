import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#d1=pd.read_csv('data.csv')
#checking cols
#print(d1.columns)
#required cols  imputs [cpu usage memory usage,unmapped page cachhe,mean disk i/o,disk usage,task
#priority, task resubmissions, and scheduling delay] output=failed
d1=pd.read_csv('data.csv',usecols = ['cpu usage and memory usage','pagecache','diskusage','input', 'output','priority','scheduling delay','event','failed'])
#print(d1.shape)

#sumbract output time - input time
d1['meanio']=d1['output']-d1['input']

#drop input and output columns
d1=d1.drop(['input'], axis=1)
d1=d1.drop(['output'], axis=1)

#renaming
d1.rename(columns = {'cpu usage and memory usage':'cpu'}, inplace = True)

#checking missing values
#a=d1.isnull().sum()
#print(a)
#only diskusage and scheduling delay have missing values which are in 10k compared to 450000 entries so we drop
d1=d1.dropna()

#cnverting cpu from string to int type and seperating it


d1['event'] = d1['event'].apply(preprocessing.LabelEncoder().fit_transform)
print(d1['events'].head(3))
d2=d1.cpu.str.split(",", expand = True)
d1=d1.drop(['cpu'], axis=1)


d2[0]=d2[0].str.replace("{'cpus': ","",)
d2[1]=d2[1].str.replace(" 'memory': ","")
d2[1]=d2[1].str.replace("}","")
d2[0] = d2[0].astype(float)
d2[1] = d2[1].astype(float)


d1 = d1.assign(cu = d2[0],mu=d2[1])
columns_titles = ["cu","mu","pagecache","diskusage",'priority','scheduling delay',"meanio","event","failed"]
d1=d1.reindex(columns=columns_titles)

scaler = MinMaxScaler()

d1s = scaler.fit_transform(d1.to_numpy())
d1s = pd.DataFrame(d1s , columns=["cu","mu","pagecache","diskusage",'priority','scheduling delay',"meanio","event","failed"])


#d1s.to_csv('data2.csv')