
# coding: utf-8

# In[1]:


cd E:\Newfolder\EE_last_sem\ELV781\Project\AngEv98\AngEv98


# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('data.csv')


# In[4]:


df


# In[5]:


newdf2 = pd.read_csv('newdata2.csv')


# In[6]:


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


# In[7]:


newdf2['hsgradd'] = (newdf2['educd']==12)
newdf2['hsormored'] = (newdf2['educd']>=12)
newdf2['morethsd'] = (newdf2['educd']>12)


# In[8]:


newdf2


# In[9]:


n_list = list(df['RACED'].isnull())


# In[10]:


n = len(n_list)


# In[11]:


df_unmar = pd.DataFrame(columns = newdf2.columns)
df_mar = pd.DataFrame(columns = newdf2.columns)


# In[12]:


df_unmar


# In[13]:


mar = []
unmar = []
for i in range(n):
    if n_list[i] == True:
        unmar.append(i)
    else:
        mar.append(i)
df_mar = newdf2.copy()


# In[17]:


df_mar = df_mar.iloc[mar,:].copy()


# In[16]:


newdf2.to_csv('newdata2.csv')


# In[19]:


df_mar.to_csv('mar_data.csv')


# In[ ]:


for i in range(len(unmar)):
    if i%10000==0:
        print(i)
    df_mar.drop(unmar[i])


# In[22]:


b = b.append(a.loc[0])
b = b.append(a.loc[2])
b = b.append(a.loc[4])

