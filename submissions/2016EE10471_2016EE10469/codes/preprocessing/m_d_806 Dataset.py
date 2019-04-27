
# coding: utf-8

# In[1]:


cd E:\Newfolder\EE_last_sem\ELV781\Project\AngEv98\AngEv98


# In[76]:


import pandas as pd
from math import log


# In[77]:


df = pd.read_csv('data.csv')


# In[78]:


df


# In[79]:


df.keys()


# In[80]:


l = ['girls2','boys2', 'morekids', 'samesex', 'boy1st', 'boy2nd','blackm', 'othracem', 'whitem', 'hispm', 'blackd', 'othraced', 'whited', 'hispd', 'educm', 'hsgrad', 'hsormore', 'moreths', 'agem1', 'aged1', 'ageqm', 'ageqd', 'agefstd', 'agefstm', 'workedm', 'workedd', 'weeksm1', 'weeksd1', 'hourswm', 'hourswd', 'incomem', 'incomed', 'faminc1', 'nonmomi', 'famincl', 'nonmomil']
# 'yobd'


# In[81]:


newdf = pd.DataFrame(columns = l) 


# In[82]:


newdf


# In[90]:


newdf2 = pd.DataFrame(columns = l) 


# In[94]:


newdf2['boy1st']=(df['SEXK']==0)


# In[95]:


newdf2['boy2nd']=(df['SEX2ND']==0)


# In[102]:


n=(len(df['SEXK']))
l=[]
for i in range(n):
    l.append((df['SEX2ND'][i]==1) and (df['SEXK'][i]==1))


# In[137]:


n=(len(df['SEXK']))
l=[]
for i in range(n):
    l.append(newdf2['boys2'][i] or newdf2['girls2'][i])


# In[138]:


newdf2['samesex']=l


# In[109]:


newdf2['morekids']=(df['KIDCOUNT']>2)


# In[110]:


newdf2['blackm']=df['RACEM']==2
newdf2['hispm']=df['RACEM']==12
newdf2['whitem']=df['RACEM']==1


# In[111]:


n=(len(df['SEXK']))
l=[]
for i in range(n):
    l.append(not(newdf2['blackm'][i] or newdf2['hispm'][i] or (newdf2['whitem'][i])))


# In[112]:


newdf2['othracem']=l


# In[113]:


newdf2['blackd']=df['RACED']==2
newdf2['hispd']=df['RACED']==12
newdf2['whited']=df['RACED']==1


# In[114]:


n=(len(df['SEXK']))
l=[]
for i in range(n):
    l.append(not(newdf2['blackd'][i] or newdf2['hispd'][i] or (newdf2['whited'][i])))
newdf2['othraced']=l


# In[118]:


n=(len(df['SEXK']))
l=[]
for i in range(n): 
    if df['FINGRADM'][i]==1 or df['FINGRADM'][i]==2:
        t = df['GRADEM'][i]-2
    else:
        t = df['GRADEM'][i]-3
    t = max(0,t)
    l.append(t)
newdf2['educm']=l


# In[121]:


newdf2['educm']


# In[122]:


newdf2['hsgrad'] = (newdf2['educm']==12)
newdf2['hsormore'] = (newdf2['educm']>=12)
newdf2['moreths'] = (newdf2['educm']>12)


# In[150]:


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


# In[152]:


newdf2['hsgradd'] = (newdf2['educd']==12)
newdf2['hsormored'] = (newdf2['educd']>=12)
newdf2['morethsd'] = (newdf2['educd']>12)


# In[210]:


newdf2


# In[211]:


newdf2.columns


# In[124]:


newdf2['agem1'] = df['AGEM']
newdf2['aged1'] = df['AGED']


# In[144]:


n=(len(df['SEXK']))
lm=[]
ld=[]
for i in range(n): 
    if df['QTRBTHD'][i]==0:
        yobd = 80 - df['AGED'][i]
    else:
        yobd = 79 - df['AGED'][i]
    
    if df['QTRBTHM'][i]==0:
        yobm = 80 - df['AGEM'][i]
    else:
        yobm = 79 - df['AGEM'][i]
    ageqm = 4*(80-yobm) - df['QTRBTHM'][i] - 1
    ageqd = 4*(80-yobd) - df['QTRBTHD'][i]
    tm = int((ageqm-df['AGEQK'][i])/4)
    try:
        td = int((ageqd-df['AGEQK'][i])/4)
    except ValueError:
        td = ageqd
    lm.append(tm)
    ld.append(td)
    
    


# In[145]:


newdf2['agefstm'] = lm
newdf2['agefstd'] = ld

    


# In[126]:


newdf2['weeksm1'] = df['WEEKSM']
newdf2['weeksd1'] = df['WEEKSD']

newdf2['workedm'] = (df['WEEKSM']>0)
newdf2['workedd'] = (df['WEEKSD']>0)

newdf2['hourswd'] = df['HOURSD']
newdf2['hourswm'] = df['HOURSM']


# In[127]:


n=(len(df['SEXK']))
lm=[]
ld=[]
for i in range(n): 
    td = df['INCOME1D'][i] + max(0,df['INCOME2D'][i])
    tm = df['INCOME1M'][i] + max(0,df['INCOME2M'][i])
    ld.append(td*2.099173554)
    lm.append(tm*2.099173554)


# In[128]:


newdf2['incomed'] = lm
newdf2['incomem'] = ld


# In[130]:


n=(len(df['SEXK']))
l=[]
for i in range(n): 
    t = df['FAMINC'][i] * 2.099173554
    l.append(log(max(1,t)))
newdf2['famincl'] = l


# In[131]:


n=(len(df['SEXK']))
l=[]
for i in range(n): 
    t = df['FAMINC'][i] * 2.099173554
    l.append(t)
newdf2['faminc1']=l


# In[134]:


l= []
for i in range(n):
    l.append(newdf2['faminc1'][i] - df['INCOME1M'][i] * 2.099173554)
    
newdf2['nonmomi'] = l


# In[135]:


l = []
for i in range(n):
    l.append(log(max(1.0, newdf2['nonmomi'][i])))

newdf2['nonmomil'] = l


# In[149]:


pd.set_option('display.max_columns', 36)
newdf2.head()


# In[146]:


newdf2.to_csv('newdata2.csv')


# In[205]:


max(0,df['INCOME2D'][3])


# In[207]:


for i in range(len(df['SEXK'])):
    if i%100000==0:
        print(i)
    if df['RACED'][i]==2:
        newdf2['blackd'][i]=1
        newdf2['hispd'][i]=0
        newdf2['whited'][i]=0
        newdf2['othraced'][i]=0
    elif df['RACED'][i]==12:
        newdf2['blackd'][i]=0
        newdf2['hispd'][i]=1
        newdf2['whited'][i]=0
        newdf2['othraced'][i]=0
    elif df['RACED'][i]==1:
        newdf2['blackd'][i]=0
        newdf2['hispd'][i]=0
        newdf2['whited'][i]=1
        newdf2['othraced'][i]=0
    elif not(df['RACED'][i]>0):
        newdf2['blackd'][i]=df['RACED'][i]
        newdf2['hispd'][i]=df['RACED'][i]
        newdf2['whited'][i]=df['RACED'][i]
        newdf2['othraced'][i]=df['RACED'][i]
    else:
        newdf2['blackd'][i]=0
        newdf2['hispd'][i]=0
        newdf2['whited'][i]=0
        newdf2['othraced'][i]=1


# In[209]:


df['RACED'].isnull()


# In[83]:


for i in range(len(df['SEXK'])):      
    newdf = newdf.append({}, ignore_index = True)
    if df['SEXK'][i]==0:
        newdf['boy1st'][i]=1
    else:
        newdf['boy1st'][i]=0
    if df['SEX2ND'][i]==0:
        newdf['boy2nd'][i]=1
    else:
        newdf['boy2nd'][i]=0
        
    if df['SEXK'][i]==0 and df['SEX2ND'][i]==0:
        newdf['boys2'][i]=1
    else:
        newdf['boys2'][i]=0
    
    if df['SEXK'][i]==1 and df['SEX2ND'][i]==1:
        newdf['girls2'][i]=1
    else:
        newdf['girls2'][i]=0
    
    if newdf['boys2'][i]==1 or newdf['girls2'][i]==1:
        newdf['samesex'][i]=1
    else:
        newdf['samesex'][i]=0
        
    if df['KIDCOUNT'][i]>2:
        newdf['morekids'][i]=1
    else:
        newdf['morekids'][i]=0
    
    
    if df['RACEM'][i]==2:
        newdf['blackm'][i]=1
        newdf['hispm'][i]=0
        newdf['whitem'][i]=0
        newdf['othracem'][i]=0
    elif df['RACEM'][i]==12:
        newdf['blackm'][i]=0
        newdf['hispm'][i]=1
        newdf['whitem'][i]=0
        newdf['othracem'][i]=0
    elif df['RACEM'][i]==1:
        newdf['blackm'][i]=0
        newdf['hispm'][i]=0
        newdf['whitem'][i]=1
        newdf['othracem'][i]=0
    else:
        newdf['blackm'][i]=0
        newdf['hispm'][i]=0
        newdf['whitem'][i]=0
        newdf['othracem'][i]=1
    
    
    if df['RACED'][i]==2:
        newdf['blackd'][i]=1
        newdf['hispd'][i]=0
        newdf['whited'][i]=0
        newdf['othraced'][i]=0
    elif df['RACED'][i]==12:
        newdf['blackd'][i]=0
        newdf['hispd'][i]=1
        newdf['whited'][i]=0
        newdf['othraced'][i]=0
    elif df['RACED'][i]==1:
        newdf['blackd'][i]=0
        newdf['hispd'][i]=0
        newdf['whited'][i]=1
        newdf['othraced'][i]=0
    elif not(df['RACED'][i]>0):
        newdf['blackd'][i]=df['RACED'][i]
        newdf['hispd'][i]=df['RACED'][i]
        newdf['whited'][i]=df['RACED'][i]
        newdf['othraced'][i]=df['RACED'][i]
    else:
        newdf['blackd'][i]=0
        newdf['hispd'][i]=0
        newdf['whited'][i]=0
        newdf['othraced'][i]=1
    
    
    if df['FINGRADM'][i]==1 or df['FINGRADM'][i]==2:
        newdf['educm'][i] = df['GRADEM'][i]-2
    else:
        newdf['educm'][i] = df['GRADEM'][i]-3
    
    newdf['educm'][i] = max(0,newdf['educm'][i])
    
    
    newdf['hsgrad'][i] = int(newdf['educm'][i]==12)
    newdf['hsormore'][i] = int(newdf['educm'][i]>=12)
    newdf['moreths'][i] = int(newdf['educm'][i]>12)
    
    
    newdf['agem1'][i] = df['AGEM'][i]
    newdf['aged1'][i] = df['AGED'][i]
    
    
    if df['QTRBTHD'][i]==0:
        yobd = 80 - df['AGED'][i]
    else:
        yobd = 79 - df['AGED'][i]
    
    if df['QTRBTHM'][i]==0:
        yobm = 80 - df['AGEM'][i]
    else:
        yobm = 79 - df['AGEM'][i]
    
    
    ageqm = 4*(80-yobm) - df['QTRBTHM'][i] - 1
    ageqd = 4*(80-yobd) - df['QTRBTHD'][i]
    
    
    newdf['agefstm'][i] = int((ageqm-df['AGEQK'][i])/4)
    try:
        newdf['agefstd'][i] = int((ageqd-df['AGEQK'][i])/4)
    except ValueError:
        newdf['agefstd'][i] = ageqd
    newdf['weeksm1'][i] = df['WEEKSM'][i]
    newdf['weeksd1'][i] = df['WEEKSD'][i]
    
    newdf['workedm'][i] = int(df['WEEKSM'][i]>0)
    newdf['workedd'][i] = int(df['WEEKSD'][i]>0)
    
    newdf['hourswd'][i] = df['HOURSD'][i]
    newdf['hourswm'][i] = df['HOURSM'][i]
    
    
    newdf['incomed'][i] = df['INCOME1D'][i] + max(0,df['INCOME2D'][i])
    newdf['incomem'][i] = df['INCOME1M'][i] + max(0,df['INCOME2M'][i])
    
    newdf['incomed'][i] = newdf['incomed'][i]*2.099173554
    newdf['incomem'][i] = newdf['incomem'][i]*2.099173554
    
    newdf['faminc1'][i] = df['FAMINC'][i] * 2.099173554
    
    newdf['famincl'][i] = log(max(1,newdf['faminc1'][i]))
    
    newdf['nonmomi'][i] = newdf['faminc1'][i] - df['INCOME1M'][i] * 2.099173554
    newdf['nonmomil'][i] = log(max(1.0, newdf['nonmomi'][i]))
    
    
    


# In[ ]:


newdf.to_csv('newdata.csv')


# In[40]:


# newdf = newdf.append({}, ignore_index = True)


# In[84]:


newdf


# In[65]:


df['AGEQK'].isnull().sum()


# In[ ]:


df['INCOME1D']


# In[160]:


n


# In[159]:


n = len(df['RACED'])


# In[161]:


cl = list(newdf2.keys())


# In[162]:


print(cl)
print(len(cl))


# In[193]:


df_mar = pd.DataFrame(columns = cl)
df_unmar = pd.DataFrame(columns = cl)


# In[194]:


df_mar


# In[195]:


df3 = newdf2.copy()


# In[201]:


df['RACED'][3].isnan()


# In[196]:


for i in range(n):
    try:
        int(df['RACED'][i])
        rows = df3.loc[0]
        df_mar = df_mar.append(rows)
    except ValueError:
        rows = df3.loc[i]
        df_unmar = df_mar.append(rows)
        
        


# In[197]:


df_mar


# In[198]:


df_unmar


# In[168]:


df3


# In[187]:


rows = df3.loc[0]
df_mar = df_mar.append(rows)
# df3.drop(rows.index, inplace=True)


# In[188]:


df3


# In[189]:


df_mar

