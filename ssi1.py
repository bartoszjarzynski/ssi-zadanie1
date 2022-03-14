#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


iris=pd.read_csv('C:\\Users\\bartj\\Downloads\\iris.csv')


# In[10]:


iris


# In[11]:


len(iris)


# In[12]:


iris.head(10)


# In[13]:


iris.tail(10)


# In[14]:


# iris to nasze DataFrame


# In[16]:


iris.loc[10]


# In[17]:


iris.loc[10].tolist()


# In[20]:


iris.at[10, 'sepal.length']


# In[21]:


iris.columns


# In[23]:


col=iris.columns.tolist()


# In[24]:


col


# In[26]:


col[:-1]


# In[84]:


iris.at[1, col[0]]


# In[83]:


iris.iat[0,0]


# In[34]:


iris.describe() #Opis statystyczny


# In[36]:


iris.info() #Opis co jest w DataFrame


# In[38]:


import seaborn as sns


# In[42]:


sns.set_palette('husl') #Pozniej użyć: 'husl'


# In[43]:


sns.pairplot(iris,hue='variety',markers='+') #(DataFrame, kolorowanie, kształt znaczków)


# In[44]:


# JEST TO WIZUALNA ANALIZA - WAŻNA W KONTEKŚCIE PRZYSZŁEGO PROJEKTU!


# In[47]:


sns.violinplot(y='variety',x='sepal.length',data=iris,inner='quartile') #quartile = kwartyle


# In[76]:


#x' = (x-min)/(max-min)


# In[4]:


#SplitSet(data, d), d ~ (0,1)
import random


# In[11]:


class ProcessingData:
    @staticmethod
    def tasowanie(arg):
        for i in range(len(arg)-1, 0, -1):
            rand = random.randint(0,i)
            arg.iloc[i], arg.iloc[rand] = arg.iloc[rand],arg.iloc[i]
            
        return arg
    
    @staticmethod
    def trening(arg):
        podzial=int(len(arg)*0.7)        
        x1 = arg.iloc[:podzial,:]
        y1 = arg.iloc[podzial,:]
        return x1,y1
        
    @staticmethod
    def normalizacja(Xprim):
        X = Xprim.copy()
        trzyKolumny = X.select_dtypes(exclude="object")
        nazwy = trzyKolumny.columns.tolist()
        for col in nazwy:
            data = X.loc[:,col]
            maximum = max(data)
            minimum = min(data)
            substractionMaxMin = maximum - minimum
            for r in range(0,len(X),1):
                q = (X.at[r,col] - minimum) / substractionMaxMin
                X.at[r,col] = q
        return X
    

test=ProcessingData.tasowanie(iris)
print("Tasowanie:\n",test)
test1=ProcessingData.trening(iris)
print("\nPodział:\n",test1)
test2=ProcessingData.normalizacja(iris)
print("\nNormalizacja:\n",test2)


# 

# In[ ]:





# In[ ]:





# In[ ]:




