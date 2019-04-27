import numpy as np
from sklearn import linear_model
from sklearn import tree
import graphviz
from sklearn import cross_validation
import os


data_80 = np.load('data_80.npy')

header = data_80[0]

new_data = data_80[100:,:]

data_len = new_data.shape[0]

key_dict = {header[i]:i for i in range(0,len(header))}

boy1st = np.zeros(data_len)
boy2nd = np.zeros(data_len)
boys2 = np.zeros(data_len)
workedm = np.zeros(data_len)
workedd = np.zeros(data_len)
girls2 = np.zeros(data_len)
samesex = np.zeros(data_len)
incomed = np.zeros(data_len)
incomem = np.zeros(data_len)
famincl = np.zeros(data_len)
nonmomi = np.zeros(data_len)
nonmomil = np.zeros(data_len)
yobd = np.zeros(data_len)
blackm = np.zeros(data_len)
hispm = np.zeros(data_len)
whitem = np.zeros(data_len)
othracem = np.zeros(data_len)
ageqm = np.zeros(data_len)
ageqd = np.zeros(data_len)
agefstm = np.zeros(data_len)
agefstd = np.zeros(data_len)
blackd = np.zeros(data_len)
hispd = np.zeros(data_len)
whited = np.zeros(data_len)
othraced = np.zeros(data_len)
workedm_tot=np.zeros((data_len))
weeksm_tot=np.zeros((data_len))
hoursm_tot=np.zeros((data_len))
incomem_tot=np.zeros((data_len))
famincl_tot=np.zeros((data_len))
ageqm = np.zeros(data_len)
ageqd = np.zeros(data_len)
agefstm = np.zeros(data_len)
agefstd = np.zeros(data_len)
msample=np.zeros((data_len))
morekids=np.zeros((data_len))
samesex=np.zeros((data_len))

def str2int(data, index_str, convert):
    if (convert == 0):
        index = key_dict[index_str]
        return data[:,index]
    else:
        index = key_dict[index_str]
        y = []
        for i in range(len(data[:, index])):
            x = data[i,index]
            if (x != ''):
                y.append(int(x))
            else:
                y.append(-1)
        return np.asarray(y)

income1d = str2int(new_data, 'INCOME1D',1)
weeksm = str2int(new_data, 'WEEKSM',1)
weeksd = str2int(new_data, 'WEEKSD',1)
hoursm = str2int(new_data, 'HOURSM',1)
hoursd = str2int(new_data, 'HOURSD',1)
income1d = str2int(new_data, 'INCOME1D',1)
income2d = str2int(new_data, 'INCOME2D',1)
faminc = str2int(new_data, 'FAMINC',1)
income1m = str2int(new_data, 'INCOME1M',1)
income2m = str2int(new_data, 'INCOME2M',1)

timesmar = str2int(new_data, 'TIMESMAR',1)
sexk = str2int(new_data,'SEXK',1)
sex2nd = str2int(new_data,'SEX2ND',1)
qtrbthm = str2int(new_data,'QTRBTHM',1)
qtrmar = str2int(new_data,'QTRMAR',1)
agemar = str2int(new_data,'AGEMAR',1)
qtrbkid = str2int(new_data,'QTRBKID',1)
aged = str2int(new_data,'AGED',1)
agem = str2int(new_data,'AGEM',1)
qtrbthd = str2int(new_data,'QTRBTHD',1)

ageq3rd = str2int(new_data,'QTRBTHD',1)
asex = str2int(new_data,'ASEX',1)
aqtrbrth = str2int(new_data,'AQTRBRTH',1)
asex2nd = str2int(new_data,'ASEX2ND',1)
aage = str2int(new_data,'AAGE',1)
aage2nd = str2int(new_data,'AAGE2ND',1)
agem = str2int(new_data,'AGEM',1)



agem1 = str2int(new_data,'AGEM',0)
aged1 = str2int(new_data,'AGED',0)
raced = str2int(new_data,'RACED',0)
racem = str2int(new_data,'RACEM',0)
ageq2nd = str2int(new_data,'AGEQ2ND',0)
yobk = str2int(new_data,'YOBK',0)
ageqk = str2int(new_data,'AGEQK',0)
kidcount = str2int(new_data,'KIDCOUNT',0)
yobm = str2int(new_data,'YOBM',0)

illegit=np.zeros(data_len)
yom = np.zeros(data_len)
for ind_i in range(data_len):
    if qtrmar[ind_i]>0 :
        qtrmar[ind_i]=qtrmar[ind_i]-1
        if qtrbthm[ind_i] <= qtrmar[ind_i]:
            yom[ind_i]=yobm[ind_i]+agemar[ind_i]
        elif qtrbthm[ind_i] > qtrmar[ind_i] :
            yom[ind_i] = yobm[ind_i] + agemar[ind_i] + 1
            dom_q = yom[ind_i] + (qtrmar[ind_i]/4)
            do1b_q = yobk[ind_i] + ((qtrbkid[ind_i])/4)
            if (dom_q - do1b_q)>0 :
                 illegit[ind_i]=1
            



for ind_i in range(data_len):
    if(sexk[ind_i] == 0):
            boy1st[ind_i]=1
    if(sex2nd[ind_i] == 0):
            boy2nd[ind_i]=1
    if(sexk[ind_i] == 0 and sex2nd[ind_i] ==0):
            boys2[ind_i]=1
    if(sexk[ind_i] == 1 and sex2nd[ind_i] ==1):
            girls2[ind_i]=1
        
    incomed[ind_i] = income1d[ind_i] + max(0,income2d[ind_i])
    incomem[ind_i] = income1m[ind_i] + max(0,income2m[ind_i])
    incomem[ind_i] = incomem[ind_i]*2.099173554
    incomed[ind_i] = incomed[ind_i]*2.099173554
    faminc[ind_i] = faminc[ind_i]*2.099173554
    famincl[ind_i] = np.log(max(faminc[ind_i],1))
    nonmomi[ind_i] = faminc[ind_i] - income1m[ind_i]*2.099173554
    nonmomil[ind_i] = np.log(max(1,nonmomi[ind_i]))
    
    if(racem[ind_i]==2):
        blackm[ind_i]=1
    if(racem[ind_i]==12):
        hispm[ind_i]=1
    if(racem[ind_i]==1):
        whitem[ind_i]=1
        
    othracem[ind_i] = 1 - (blackm[ind_i]+hispm[ind_i]+whitem[ind_i])

    if(raced[ind_i]==2):
        blackd[ind_i]=1
    if(raced[ind_i]==12):
        hispd[ind_i]=1
    if(raced[ind_i]==1):
        whited[ind_i]=1
        
    othraced[ind_i] = 1 - (blackd[ind_i]+hispd[ind_i]+whited[ind_i])
    
    if(weeksm[ind_i]>0):
        workedm[ind_i]=1
        
    if(weeksd[ind_i]>0):
        workedd[ind_i]=1



            
            

for ind_i in range(data_len): 
    if qtrbthd[ind_i] == 0 :
        yobd[ind_i] = 80-aged[ind_i];
    else:
        yobd[ind_i] = 79-aged[ind_i];
        



for ind_i in range(data_len):
    ageqm[ind_i] = 4*(80-yobm[ind_i])-qtrbthm[ind_i]-1;
    ageqd[ind_i] = 4*(80-yobd[ind_i])-qtrbthd[ind_i]
    agefstm[ind_i] = int((ageqm[ind_i]- ageqk[ind_i])/4)
    agefstd[ind_i] = int((ageqd[ind_i]-ageqk[ind_i])/4)
   


for ind_i in range(data_len):
    if(kidcount[ind_i]>2):
        morekids[ind_i]=1
    if(sexk[ind_i] == 1 and sex2nd[ind_i] ==1)or (sexk[ind_i] == 0 and sex2nd[ind_i]==0):
        samesex[ind_i]=1



for ind_i in range(data_len):
    
    if(aged[ind_i]!=0 and timesmar[ind_i]==1 and illegit[ind_i]==0 and agefstd[ind_i]>=15 and agefstm[ind_i]>=15):
        msample[ind_i]=1



for ind_i in range(0,data_len):
    if(morekids[ind_i]==1):
        workedm_tot[ind_i]=workedm[ind_i]*2
        weeksm_tot[ind_i]=weeksm[ind_i]*2
        hoursm_tot[ind_i]=hoursm[ind_i]*2
        incomem_tot[ind_i]=incomem[ind_i]*2
        famincl_tot[ind_i]=famincl_tot[ind_i]*2
    else:
        workedm_tot[ind_i]=workedm[ind_i]*-2
        weeksm_tot[ind_i]=weeksm[ind_i]*-2
        hoursm_tot[ind_i]=hoursm[ind_i]*-2
        incomem_tot[ind_i]=incomem[ind_i]*-2
        famincl_tot[ind_i]=famincl_tot[ind_i]*-2
        
        


#%%

index_all_women = []
for ind_i in range(data_len):
    if agem[ind_i] >= 21 and agem[ind_i] <=35 and kidcount[ind_i] >= 2 and ageq2nd[ind_i] > 4 and agefstm[ind_i] >=15 \
       and aage[ind_i] == 0 and asex[ind_i] == 0 and aqtrbrth[ind_i] == 0  and asex2nd[ind_i] == 0 and aage2nd[ind_i] == 0:
            index_all_women.append(ind_i)
            



index_m_women = []
for ind_i in range(data_len):
    if (agem[ind_i] >= 21 and agem[ind_i] <=35 and kidcount[ind_i] >= 2 and ageq2nd[ind_i] > 4 \
        and agefstm[ind_i] >=15 and aage[ind_i] == 0 and asex[ind_i] == 0\
        and aqtrbrth[ind_i] == 0  and asex2nd[ind_i] == 0 and aage2nd[ind_i] == 0 and msample[ind_i]==1):
            index_m_women.append(ind_i)
            



index_morekids_0 = []
index_morekids_1 = []
for ind_i in range(data_len):
    if (agem[ind_i] >= 21 and agem[ind_i] <=35 and kidcount[ind_i] >= 2 and ageq2nd[ind_i] > 4 \
        and agefstm[ind_i] >=15 and aage[ind_i] == 0 and asex[ind_i] == 0\
        and aqtrbrth[ind_i] == 0  and asex2nd[ind_i] == 0 and aage2nd[ind_i] == 0):
            if(morekids[ind_i]==0):
                index_morekids_0.append(ind_i)
            else:
                index_morekids_1.append(ind_i)
   
             
index_morekids_m_0 = []
index_morekids_m_1 = []
for ind_i in range(data_len):
    if (agem[ind_i] >= 21 and agem[ind_i] <=35 and kidcount[ind_i] >= 2 and ageq2nd[ind_i] > 4 \
        and agefstm[ind_i] >=15 and aage[ind_i] == 0 and asex[ind_i] == 0\
        and aqtrbrth[ind_i] == 0  and asex2nd[ind_i] == 0 and aage2nd[ind_i] == 0 and msample[ind_i]==1):
            if(morekids[ind_i]==0):
                index_morekids_m_0.append(ind_i)
            else:
                index_morekids_m_1.append(ind_i)
#%%
print('Table 2, col:1')

print('Children ever born ',np.mean(kidcount[index_all_women]))
print('More than 2 children ',np.mean(morekids[index_all_women]))
print('Boy Ist ',np.mean(boy1st[index_all_women]))
print('Boy 2nd ',np.mean(boy2nd[index_all_women]))
print('Two boys ',np.mean(boys2[index_all_women]))
print('Two girls ',np.mean(girls2[index_all_women]))
print('Same sex ',np.mean(samesex[index_all_women]))


print('\nTable 2, col:2')

print('Children ever born ',np.mean(kidcount[index_m_women]))
print('More than 2 children ',np.mean(morekids[index_m_women]))
print('Boy Ist ',np.mean(boy1st[index_m_women]))
print('Boy 2nd ',np.mean(boy2nd[index_m_women]))
print('Two boys ',np.mean(boys2[index_m_women]))
print('Two girls ',np.mean(girls2[index_m_women]))
print('Same sex ',np.mean(samesex[index_m_women]))

            
#%%

def fit_linear(x_tup,y):
    
    x_list=[]
    
    for tup in x_tup:
        x_list.append(tup.reshape((-1,1)))
    
    x_final = np.concatenate(x_list,axis=1)
    
    reg = linear_model.LinearRegression().fit(x_final, y)
    
    #print(reg.coef_)
    
    return reg.coef_


print('\nTable 6, Col:1')

ind_m = index_all_women
coeff = fit_linear((samesex[ind_m],),morekids[ind_m])
print('Samesex: ',coeff[0])

print('\nTable 6, Col:2')

ind_m = index_all_women

coeff = fit_linear((boy1st[ind_m],boy2nd[ind_m],samesex[ind_m],agem1[ind_m],\
            agefstm[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),morekids[ind_m])

print("Boy Ist",coeff[0],"\nBoy 2nd",coeff[1],"\nSame sex",coeff[2])


print('\nTable 6, Col:4')

ind_m = index_m_women
coeff = fit_linear((samesex[ind_m],),morekids[ind_m])
print('Samesex: ',coeff[0])

print('\nTable 6, Col:5')

ind_m = index_m_women

coeff = fit_linear((boy1st[ind_m],boy2nd[ind_m],samesex[ind_m],agem1[ind_m],\
            agefstm[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),morekids[ind_m])

print("Boy Ist",coeff[0],"\nBoy 2nd",coeff[1],"\nSame sex",coeff[2])


print('\nTable 7, Col:1')
ind_m = index_all_women

coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),workedm[ind_m])
print('Worked for pay: ',coeff[0])
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),weeksm[ind_m])
print('Weeks worked: ',coeff[0])
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),hoursm[ind_m])
print('Hours/week: ',coeff[0])
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),incomem[ind_m])
print('Labor income: ',coeff[0])
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),famincl[ind_m])
print('ln(Family income): ',coeff[0]) 


print('\nTable 7, Col:4')
ind_m = index_m_women
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),workedm[ind_m])
print('Worked for pay: ',coeff[0])
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),weeksm[ind_m])
print('Weeks worked: ',coeff[0])
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),hoursm[ind_m])
print('Hours/week: ',coeff[0])
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),incomem[ind_m])
print('Labor income: ',coeff[0])
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),famincl[ind_m])
print('ln(Family income): ',coeff[0]) 


print('\nTable 7, Col:7')
ind_m = index_m_women
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),workedd[ind_m])
print('Worked for pay: ',coeff[0])
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),weeksd[ind_m])
print('Weeks worked: ',coeff[0])
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),hoursd[ind_m])
print('Hours/week: ',coeff[0])
coeff = fit_linear((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m]),incomed[ind_m])
print('Labor income: ',coeff[0])





#%%

def fit_tree(x_tup,lab_x,y,lab_y,name):
    
    x_list=[]
    
    for tup in x_tup:
        x_list.append(tup.reshape((-1,1)))
    
    x_final = np.concatenate(x_list,axis=1)
    
    x_train,x_test,y_train,y_test = cross_validation.train_test_split(x_final,y,test_size=0.1,random_state=0)
    
    best_depth = 0
    best_score = 0
    for i in range(2,15):
    
        clf = tree.DecisionTreeRegressor(max_depth=i)
        
        clf = clf.fit(x_train,y_train)
        
        score=clf.score(x_test,y_test)
        
        if(score>best_score):
            best_score = score
            best_depth = i
    
    print('Best_Score: ',best_score,' Best_Depth: ',best_depth)
    
    clf = tree.DecisionTreeRegressor(max_depth=best_depth)
        
    clf = clf.fit(x_train,y_train)
    
    dot_data = tree.export_graphviz(clf, out_file=None, 
                     filled=True, rounded=True,  
                     feature_names=lab_x,  
                     #class_names=lab_y, 
                     special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render('./save_trees/'+name+'_%d'%(best_depth,))
    
    
    return clf


    
if not os.path.exists('save_trees'):
         os.makedirs('save_trees')

print('Plain Decision Tree for Treatment 0 and Treatment 1')
print('Making Decision Trees for Treatment 0_ALL')


ind_m = index_morekids_0
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],workedm[ind_m],['Value'],\
      name='Method_1/Treatment_ALL=0_worked_for_pay')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],weeksm[ind_m],['Value'],\
      name='Method_1/Treatment_ALL=0_Weeks_worked')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],hoursm[ind_m],['Value'],\
      name='Method_1/Treatment_ALL=0_Hours_per_week')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],incomem[ind_m],['Value'],\
      name='Method_1/Treatment_ALL=0_Labor_income')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],famincl[ind_m],['Value'],\
      name='Method_1/Treatment_ALL=0_ln(Family_income)')
    

print('\nMaking Decision Trees for Treatment 1_ALL')


ind_m = index_morekids_1
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],workedm[ind_m],['Value'],\
      name='Method_1/Treatment_ALL=1_worked_for_pay')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],weeksm[ind_m],['Value'],\
      name='Method_1/Treatment_ALL=1_Weeks_worked')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],hoursm[ind_m],['Value'],\
      name='Method_1/Treatment_ALL=1_Hours_per_week')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],incomem[ind_m],['Value'],\
      name='Method_1/Treatment_ALL=1_Labor_income')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],famincl[ind_m],['Value'],\
      name='Method_1/Treatment_ALL=1_ln(Family_income)')

print('Making Decision Trees for Treatment 0_Married')


ind_m = index_morekids_m_0
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],workedm[ind_m],['Value'],\
      name='Method_1/Treatment_Married=0_worked_for_pay')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],weeksm[ind_m],['Value'],\
      name='Method_1/Treatment_Married=0_Weeks_worked')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],hoursm[ind_m],['Value'],\
      name='Method_1/Treatment_Married=0_Hours_per_week')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],incomem[ind_m],['Value'],\
      name='Method_1/Treatment_Married=0_Labor_income')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],famincl[ind_m],['Value'],\
      name='Method_1/Treatment_Married=0_ln(Family_income)')
    

print('\nMaking Decision Trees for Treatment_Married 1_Married')


ind_m = index_morekids_m_1
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],workedm[ind_m],['Value'],\
      name='Method_1/Treatment_Married=1_worked_for_pay')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],weeksm[ind_m],['Value'],\
      name='Method_1/Treatment_Married=1_Weeks_worked')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],hoursm[ind_m],['Value'],\
      name='Method_1/Treatment_Married=1_Hours_per_week')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],incomem[ind_m],['Value'],\
      name='Method_1/Treatment_Married=1_Labor_income')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],famincl[ind_m],['Value'],\
      name='Method_1/Treatment_Married=1_ln(Family_income)')
        

print('Making Decision Trees for Treatment 0_Married_Husband')


ind_m = index_morekids_0
clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],workedd[ind_m],['Value'],\
      name='Method_1/Treatment_Married_Husband=0_worked_for_pay')
    
clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],weeksd[ind_m],['Value'],\
      name='Method_1/Treatment_Married_Husband=0_Weeks_worked')

clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],hoursd[ind_m],['Value'],\
      name='Method_1/Treatment_Married_Husband=0_Hours_per_week')
    
clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],incomed[ind_m],['Value'],\
      name='Method_1/Treatment_Married_Husband=0_Labor_income')

clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],famincl[ind_m],['Value'],\
      name='Method_1/Treatment_Married_Husband=0_ln(Family_income)')
    

print('\nMaking Decision Trees for Treatment_Married_Husband 1_Married_Husband')


ind_m = index_morekids_1
clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],workedm[ind_m],['Value'],\
      name='Method_1/Treatment_Married_Husband=1_worked_for_pay')
    
clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],weeksm[ind_m],['Value'],\
      name='Method_1/Treatment_Married_Husband=1_Weeks_worked')

clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],hoursm[ind_m],['Value'],\
      name='Method_1/Treatment_Married_Husband=1_Hours_per_week')
    
clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],incomem[ind_m],['Value'],\
      name='Method_1/Treatment_Married_Husband=1_Labor_income')

clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],famincl[ind_m],['Value'],\
      name='Method_1/Treatment_Married_Husband=1_ln(Family_income)')





print('\nPlain Decision Tree for TOT_ALL')


ind_m = index_all_women
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],workedm_tot[ind_m],['Value'],\
      name='Method_2/ALL_worked_for_pay')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],weeksm_tot[ind_m],['Value'],\
      name='Method_2/ALL_Weeks_worked')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],hoursm_tot[ind_m],['Value'],\
      name='Method_2/ALL_Hours_per_week')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],incomem_tot[ind_m],['Value'],\
      name='Method_2/ALL_Labor_income')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],famincl_tot[ind_m],['Value'],\
      name='Method_2/ALL_ln(Family_income)')
        
        
print('\nPlain Decision Tree for TOT_Married')


ind_m = index_all_women
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],workedm_tot[ind_m],['Value'],\
      name='Method_2/Married_worked_for_pay')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],weeksm_tot[ind_m],['Value'],\
      name='Method_2/Married_Weeks_worked')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],hoursm_tot[ind_m],['Value'],\
      name='Method_2/Married_Hours_per_week')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],incomem_tot[ind_m],['Value'],\
      name='Method_2/Married_Labor_income')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],famincl_tot[ind_m],['Value'],\
      name='Method_2/Married_ln(Family_income)')
        
        

print('\nPlain Decision Tree for TOT_Married_Husband')


ind_m = index_all_women
clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],workedm_tot[ind_m],['Value'],\
      name='Method_2/Married_Husband_worked_for_pay')
    
clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],weeksm_tot[ind_m],['Value'],\
      name='Method_2/Married_Husband_Weeks_worked')

clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],hoursm_tot[ind_m],['Value'],\
      name='Method_2/Married_Husband_Hours_per_week')
    
clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],incomem_tot[ind_m],['Value'],\
      name='Method_2/Married_Husband_Labor_income')

clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],famincl_tot[ind_m],['Value'],\
      name='Method_2/Married_Husband_ln(Family_income)')
        
      
print('\nPlain Decision Tree for Tau_Hat_ALL')


ind_m = index_all_women
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],workedm[ind_m],['Value'],\
      name='Method_3/All_worked_for_pay')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],weeksm[ind_m],['Value'],\
      name='Method_3/All_Weeks_worked')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],hoursm[ind_m],['Value'],\
      name='Method_3/All_Hours_per_week')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],incomem[ind_m],['Value'],\
      name='Method_3/All_Labor_income')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],famincl[ind_m],['Value'],\
      name='Method_3/All_ln(Family_income)')

print('\nPlain Decision Tree for Tau_Hat_Married')

ind_m = index_m_women
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],workedm[ind_m],['Value'],\
      name='Method_3/Married_worked_for_pay')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],weeksm[ind_m],['Value'],\
      name='Method_3/Married_Weeks_worked')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],hoursm[ind_m],['Value'],\
      name='Method_3/Married_Hours_per_week')
    
clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],incomem[ind_m],['Value'],\
      name='Method_3/Married_Labor_income')

clf = fit_tree((agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],famincl[ind_m],['Value'],\
      name='Method_3/Married_ln(Family_income)')

print('\nPlain Decision Tree for Tau_Hat_ALL_Husband')

ind_m = index_m_women
clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],workedm[ind_m],['Value'],\
      name='Method_3/Married_Husband_worked_for_pay')
    
clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],weeksm[ind_m],['Value'],\
      name='Method_3/Married_Husband_Weeks_worked')

clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],hoursm[ind_m],['Value'],\
      name='Method_3/Married_Husband_Hours_per_week')
    
clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],incomem[ind_m],['Value'],\
      name='Method_3/Married_Husband_Labor_income')

clf = fit_tree((aged[ind_m],agefstm[ind_m],boy1st[ind_m],\
            boy2nd[ind_m],blackd[ind_m],hispd[ind_m],othraced[ind_m])\
    ,['agem1','agefstm','boy1st',\
            'boy2nd','blackd','hispd','othraced'],famincl[ind_m],['Value'],\
      name='Method_3/Married_Husband_ln(Family_income)')




#%%
#from linearmodels.iv import IV2SLS
#
#def fit_2sls(x_tup_inst,x_tup_endo,x_tup_exo,y):
#    
#    x_list_inst=[]
#    for tup in x_tup_inst:
#        x_list_inst.append(tup.reshape((-1,1)))
#    x_final_inst = np.concatenate(x_list_inst,axis=1)
#    
#    x_list_endo=[]
#    for tup in x_tup_endo:
#        x_list_endo.append(tup.reshape((-1,1)))
#    x_final_endo = np.concatenate(x_list_endo,axis=1)
#    #x_final_endo = x_tup_endo[0].reshape((-1,1))
#    
#    x_list_exo=[]
#    for tup in x_tup_exo:
#        x_list_exo.append(tup.reshape((-1,1)))
#    x_final_exo = np.concatenate(x_list_exo,axis=1)
#    
#    model = IV2SLS(dependent=y.reshape((-1,1)),\
#            exog=x_final_exo,\
#            endog=x_final_endo,\
#            instruments=x_final_inst).fit(cov_type='unadind_iusted')
#    
#    print(model.summary)
#    
#    
#
#ind_m = index_all_women
#
##coeff = fit_2sls((morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],\
##            boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),workedm[ind_m])
##print('Worked for pay: ',coeff[0])
#
#
##coeff = fit_2sls((samesex[ind_m],),(morekids[ind_m],),(agem1[ind_m],agefstm[ind_m],boy1st[ind_m],boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),workedm[ind_m])
#
##coeff = fit_2sls((samesex[ind_m],),(morekids[ind_m],),\
##                 (agem1[ind_m],agefstm[ind_m],boy1st[ind_m],boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m]),\
##                 workedm[ind_m])
#
#
#coeff = fit_2sls((samesex[ind_m],),(morekids[ind_m],),\
#                 (agem1[ind_m],agefstm[ind_m],boy1st[ind_m],boy2nd[ind_m],),\
#                 workedm[ind_m])
#
#
##coeff = fit_2sls((samesex[ind_m],),\
##                 (morekids[ind_m],agem1[ind_m],agefstm[ind_m],boy1st[ind_m],boy2nd[ind_m],blackm[ind_m],hispm[ind_m],othracem[ind_m],),(boy2nd[ind_m]*0+1,),workedm[ind_m])
#
#
#





























