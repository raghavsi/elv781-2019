
# coding: utf-8

# In[1]:


cd E:\Newfolder\EE_last_sem\ELV781\Project\AngEv98\AngEv98


# In[2]:


import pandas as pd


# In[10]:


df = pd.read_csv('data.csv')


# In[11]:


df


# In[12]:


newdf2 = pd.read_csv('newdata2.csv')


# In[13]:


n=(len(df['SEXK']))
l=[]
for i in range(n): 
    if df['FINGRADD'][i]==1 or df['FINGRADD'][i]==2:
        t = df['GRADED'][i]-2
    else:
        t = df['GRADED'][i]-3
    t = max(0,t)
    l.append(t)
newdf2['educd']=l


# In[14]:


newdf2['hsgradd'] = (newdf2['educd']==12)
newdf2['hsormored'] = (newdf2['educd']>=12)
newdf2['morethsd'] = (newdf2['educd']>12)


# In[15]:


newdf2


# In[28]:


n_list = list(df['RACED'].isnull())


# In[31]:


n = len(n_list)


# In[58]:


df_unmar = pd.DataFrame(columns = newdf2.columns)
df_mar = pd.DataFrame(columns = newdf2.columns)


# In[59]:


df_unmar


# In[60]:


mar = []
unmar = []
for i in range(n):
    if n_list[i] == True:
        unmar.append(i)
    else:
        mar.append(i)
df_mar = newdf2.copy()


# In[61]:


for i in range(len(unmar)):
    if i%10000==0:
        print(i)
    df_mar.drop(unmar[i])


# In[22]:


b = b.append(a.loc[0])
b = b.append(a.loc[2])
b = b.append(a.loc[4])

