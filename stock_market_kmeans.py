#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing modules

import sys
from pandas_datareader import data
from matplotlib import pyplot as plt
import pandas as pd
import datetime
import numpy as np


# In[2]:


companies_dict = {
    'Amazon' : 'AMZN',
    'Apple' : 'AAPL',
    'Walgreen' : 'WBA',
    'Northrop Grumman' : 'NOC',
    'Boeing' : 'BA',
    'Lockheed Martin' : 'LMT',
    'McDonalds' : 'MCD' ,
    'Intel': 'INTC' ,
    'Navistar' : 'NAV',
    'IBM' : 'IBM' ,
    'Texas Instruments' : 'TXN',
    'MasterCard': 'MA',
    'Microsoft' : 'MSFT',
    'General Electric' : 'GE',
    'Symantec' : 'SYMC' , 
    'American Express' : 'AXP' ,
    'Pepsi' : 'PEP' , 
    'Coca Cola' : 'KO' ,
    'Johnson & Johnson' : 'JNJ' ,
    'Toyota' : 'TM', 
    'Honda' : 'HMC' ,
    'Mistubishi' : 'MSBHY' ,
    'Sony' : 'SNE' ,
    'Exxon' : 'XOM' ,
    'Chevron' : 'CVX' ,
    'Valero Energy' : 'VLO', 
    'Ford' : 'F' , 
    'bank of America' : 'BAC'
}
companies = sorted(companies_dict.items() , key = lambda x : x[1])


# In[3]:


#getting stock market value data

data_source = 'yahoo'
start_date = '2015-01-01'
end_date = '2018-12-31'
panel_data = data.DataReader(list(companies_dict.values()),data_source,start_date,end_date)


# In[28]:


#setting up the dataset

stock_close = panel_data['Close']
stock_open = panel_data['Open']
stock_close = np.array(stock_close).T
stock_open = np.array(stock_open).T
row,col = stock_close.shape


# In[33]:


#storing stock price movements

movements = np.zeros([row,col])

for i in range(0,row):
    movements[i][:] = np.subtract(stock_close[i][:],stock_open[i][:])

for i in range(0,len(companies)):
    print('Company {}, Change: {}'. format(companies[i][0], sum(movements[i][:])))


# In[50]:


#visulaising the dataset

plt.clf
plt.figure(figsize=(18,16))

ax1 = plt.subplot(221)
plt.plot(movements[0][:])
plt.title(companies[0])

ex1 = plt.subplot(222, sharey=ax1)
plt.plot(movements[1][:])
plt.title(companies[1])
plt.show()


# In[53]:


#normalizing because some companies are worth much more than others

from sklearn.preprocessing import Normalizer

normalizer = Normalizer()
new = normalizer.fit_transform(movements)

print(new.max())
print(new.min())
print(new.mean())


# In[54]:


#visualizing dataset after normalization

plt.clf
plt.figure(figsize=(18,16))

ax1 = plt.subplot(221)
plt.plot(new[0][:])

plt.title(companies[0])

ex1 = plt.subplot(222, sharey=ax1)
plt.plot(new[1][:])
plt.title(companies[1])
plt.show()


# In[56]:


#Pipeline 

from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

#Normalizer

normalizer = Normalizer()

#K-Means model - 10 clusters (tradeoff between number of clusters and inertial)

kmeans = KMeans(n_clusters=10,max_iter = 1000)

#Make Pipeline 

pipeline = make_pipeline(normalizer, kmeans)


# In[72]:


pipeline.fit(new)


# In[73]:


#lesser the inertia better it is

print(kmeans.inertia_)


# In[78]:


labels = pipeline.predict(new)
df = pd.DataFrame({'labels':labels, 'companies':companies})

print(df.sort_values('labels'))


# In[81]:


# PCA Analysis using Singular value decomposition
from sklearn.decomposition import PCA

reduced_data = PCA(n_components = 2).fit_transform(new)

#running K-Means on reduced data

kmeans = KMeans(n_clusters = 10, max_iter=1000)
kmeans.fit(reduced_data)
labels = kmeans.predict(reduced_data)

df = pd.DataFrame({'labels':labels, 'companies':companies})

print(kmeans.inertia_)
print(df.sort_values('labels'))


# In[97]:


h = 0.01

#printing the decision boundary

x_min,x_max = reduced_data[:,0].min()-1,reduced_data[:,0].max()+1
y_min,y_max = reduced_data[:,1].min()-1,reduced_data[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))


#labels for each point in the mesh using our trained model
Z = kmeans.predict(np.c_[xx.ravel(),yy.ravel()])

#results in color plot
Z = Z.reshape(xx.shape)

#colorplot
cmap = plt.cm.Paired

#Plotting figure
plt.clf()
plt.figure(figsize=(10,10))

plt.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          cmap = cmap,
          aspect = 'auto', origin = 'lower')

plt.plot(reduced_data[:,0],reduced_data[:,1],'k.', markersize=5)

#Plot the centroid of each cluster as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1],
           marker = 'x', s=169, linewidth=3,
           color='w', zorder=10)

plt.title('K-Means Clustering on Stock Market Movements (PCA-Reduced Data)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

