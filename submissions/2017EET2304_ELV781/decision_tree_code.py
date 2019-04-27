from sklearn import tree
from sklearn.model_selection import train_test_split
import os
import graphviz
import numpy as np
from pdb import set_trace as trace
from sklearn import linear_model
import matplotlib.pyplot as plt
from pdb import set_trace as trace
import pandas as pd
import gc

from sas7bdat import SAS7BDAT

data_80 = []
with SAS7BDAT('./AngEv98/m_d_806.sas7bdat', skip_header=False) as reader:
    for row in reader:
        data_80.append(row)
gc.collect()
header = data_80[0]
d = data_80[1:,:]

AGED = np.where(header=='AGED')[0][0]
ASEX2ND = np.where(header=='ASEX2ND')[0][0]
AGEM = np.where(header=='AGEM')[0][0]
KIDCOUNT = np.where(header=='KIDCOUNT')[0][0]
QTRBTHD = np.where(header=='QTRBTHD')[0][0]
ASEX = np.where(header=='ASEX')[0][0]
QTRBTHM = np.where(header=='QTRBTHM')[0][0]
AQTRBRTH = np.where(header=='AQTRBRTH')[0][0]
AGEMAR = np.where(header=='AGEMAR')[0][0]
YOBK = np.where(header=='YOBK')[0][0]
SEXK = np.where(header=='SEXK')[0][0]
AGEQ3RD = np.where(header=='AGEQ3RD')[0][0]
YOBM = np.where(header=='YOBM')[0][0]
AAGE = np.where(header=='AAGE')[0][0]
AGEQ2ND = np.where(header=='AGEQ2ND')[0][0]
AAGE2ND = np.where(header=='AAGE2ND')[0][0]
QTRMAR = np.where(header=='QTRMAR')[0][0]
AGEQK = np.where(header=='AGEQK')[0][0]
SEX2ND = np.where(header=='SEX2ND')[0][0]
QTRBKID = np.where(header=='QTRBKID')[0][0]

INCOME2D = np.where(header=='INCOME2D')[0][0]
INCOME1D = np.where(header=='INCOME1D')[0][0]
INCOME2M = np.where(header=='INCOME2M')[0][0]
WEEKSM = np.where(header=='WEEKSM')[0][0]
HOURSM = np.where(header=='HOURSM')[0][0]
FAMINC = np.where(header=='FAMINC')[0][0]
WEEKSD = np.where(header=='WEEKSD')[0][0]
TIMESMAR = np.where(header=='TIMESMAR')[0][0]
INCOME1M = np.where(header=='INCOME1M')[0][0]
HOURSD = np.where(header=='HOURSD')[0][0]
RACED =  np.where(header=='RACED')[0][0]
AGEM = np.where(header=='AGEM')[0][0]
RACEM =  np.where(header=='RACEM')[0][0]


timesmar = d[:, TIMESMAR]
agem1 = d[:, AGEM]
aged1 = d[:, AGED]
raced = d[:, RACED]
racem = d[:, RACEM]
qtrbthd = d[:, QTRBTHD]
sexk = d[:, SEXK]
aqtrbrth = d[:, AQTRBRTH]
ageq3rd = d[:, AGEQ3RD]
ageq2nd = d[:, AGEQ2ND]
yobk = d[:, YOBK]
aged = d[:, AGED]
agemar = d[:, AGEMAR]
ageqk = d[:, AGEQK]
sex2nd = d[:, SEX2ND]
kidcount = d[:, KIDCOUNT]
asex = d[:, ASEX]
aage = d[:, AAGE]
asex2nd = d[:, ASEX2ND]
agem = d[:, AGEM]
qtrbthm = d[:, QTRBTHM]
aage2nd = d[:, AAGE2ND]
qtrmar = d[:, QTRMAR]
yobm = d[:, YOBM]
qtrbkid = d[:, QTRBKID]


weeksm = np.asarray([int(x) for x in d[:, WEEKSM]])
weeksd = np.asarray([int(x) if x != '' else 0 for x in d[:, WEEKSD]])
hoursm = np.asarray([int(x) for x in d[:, HOURSM]])
hoursd = np.asarray([int(x) if x != '' else 0 for x in d[:, HOURSD]])
income1d = [int(x) if x != '' else -1 for x in d[:, INCOME1D]]
income2d = [int(x) if x != '' else -1 for x in d[:, INCOME2D]]
faminc = [int(x) for x in d[:, FAMINC]]
income1m = [int(x) for x in d[:, INCOME1M]]
income2m = [int(x) for x in d[:, INCOME2M]]


timesmar = [int(x) for x in timesmar]
sexk = [int(x) for x in sexk]
sex2nd = [int(x) if x!='' else -1 for x in sex2nd]
qtrbthm = [int(x) for x in qtrbthm]
qtrmar = [int(x) for x in qtrmar]
agemar = [int(x) for x in agemar]
qtrbkid = [int(x) for x in qtrbkid]
aged = np.asarray([int(x) if x !='' else 0 for x in aged])
agem = [int(x) for x in agem]
qtrbthd = [int(x) if x !='' else 0 for x in qtrbthd]
ageq3rd = [int(x) if x !='' else -1 for x in qtrbthd]
asex = [int(x) for x in asex]
aqtrbrth = [int(x) for x in aqtrbrth]
asex2nd = [int(x) if x != '' else -1 for x in asex2nd]
aage = [int(x) for x in aage]
aage2nd = [int(x) if x != '' else -1 for x in asex2nd]
agem = [int(x) for x in agem]


illegit=np.zeros(no_of_observations)
yom = np.zeros(no_of_observations)
for j in range(no_of_observations):
    if qtrmar[j]>0 :
        qtrmar[j]=qtrmar[j]-1
        if qtrbthm[j] <= qtrmar[j]:
            yom[j]=yobm[j]+agemar[j]
        elif qtrbthm[j] > qtrmar[j] :
            yom[j] = yobm[j] + agemar[j] + 1
            dom_q = yom[j] + (qtrmar[j]/4)
            do1b_q = yobk[j] + ((qtrbkid[j])/4)
            if (dom_q - do1b_q)>0 :
                 illegit[j]=1
            


boy1st = np.zeros(no_of_observations)
boy2nd = np.zeros(no_of_observations)
boys2 = np.zeros(no_of_observations)
workedm = np.zeros(no_of_observations)
workedd = np.zeros(no_of_observations)
girls2 = np.zeros(no_of_observations)
samesex = np.zeros(no_of_observations)
incomed = np.zeros(no_of_observations)
incomem = np.zeros(no_of_observations)
famincl = np.zeros(no_of_observations)
nonmomi = np.zeros(no_of_observations)
nonmomil = np.zeros(no_of_observations)

blackm = np.zeros(no_of_observations)
hispm = np.zeros(no_of_observations)
whitem = np.zeros(no_of_observations)
othracem = np.zeros(no_of_observations)

blackd = np.zeros(no_of_observations)
hispd = np.zeros(no_of_observations)
whited = np.zeros(no_of_observations)
othraced = np.zeros(no_of_observations)


for j in range(no_of_observations):
    if(sexk[j] == 0):
            boy1st[j]=1
    if(sex2nd[j] == 0):
            boy2nd[j]=1
    if(sexk[j] == 0 and sex2nd[j] ==0):
            boys2[j]=1
    if(sexk[j] == 1 and sex2nd[j] ==1):
            girls2[j]=1
        
    incomed[j] = income1d[j] + max(0,income2d[j])
    incomem[j] = income1m[j] + max(0,income2m[j])
    incomem[j] = incomem[j]*2.099173554
    
        

for j in range(no_of_observations):
    if(racem[j]==2):
        blackm[j]=1
    if(racem[j]==12):
        hispm[j]=1
    if(racem[j]==1):
        whitem[j]=1
        
    othracem[j] = 1 - (blackm[j]+hispm[j]+whitem[j])

for j in range(no_of_observations):
    incomed[j] = incomed[j]*2.099173554
    faminc[j] = faminc[j]*2.099173554
    famincl[j] = np.log(max(faminc[j],1))
    nonmomi[j] = faminc[j] - income1m[j]*2.099173554
    nonmomil[j] = np.log(max(1,nonmomi[j]))

yobd = np.zeros(no_of_observations)
for j in range(no_of_observations): 
    if qtrbthd[j] == 0 :
        yobd[j] = 80-aged[j];
    else:
        yobd[j] = 79-aged[j];

for j in range(no_of_observations):

    if(raced[j]==2):
        blackd[j]=1
    if(raced[j]==12):
        hispd[j]=1
    if(raced[j]==1):
        whited[j]=1

    othraced[j] = 1 - (blackd[j]+hispd[j]+whited[j])

for j in range(no_of_observations):
    
    if(weeksm[j]>0):
        workedm[j]=1


for j in range(no_of_observations):
        
    if(weeksd[j]>0):
        workedd[j]=1

            
            
        

ageqm = np.zeros(no_of_observations)
ageqd = np.zeros(no_of_observations)
agefstm = np.zeros(no_of_observations)
agefstd = np.zeros(no_of_observations)

for j in range(no_of_observations):
    ageqm[j] = 4*(80-yobm[j])-qtrbthm[j]-1;
    ageqd[j] = 4*(80-yobd[j])-qtrbthd[j]
    agefstm[j] = int((ageqm[j]- ageqk[j])/4)
    agefstd[j] = int((ageqd[j]-ageqk[j])/4)
   

morekids=np.zeros((no_of_observations))
samesex=np.zeros((no_of_observations))
for j in range(no_of_observations):
    if(kidcount[j]>2):
        morekids[j]=1
    if(sexk[j] == 1 and sex2nd[j] ==1)or (sexk[j] == 0 and sex2nd[j]==0):
        samesex[j]=1


msample=np.zeros((no_of_observations))

for j in range(no_of_observations):
    
    if(aged[j]!=0 and timesmar[j]==1 and illegit[j]==0 and agefstd[j]>=15 and agefstm[j]>=15):
        msample[j]=1


workedmCase3=np.zeros((no_of_observations))
weeksmCase3=np.zeros((no_of_observations))
hoursmCase3=np.zeros((no_of_observations))
incomemCase3=np.zeros((no_of_observations))
faminclCase3=np.zeros((no_of_observations))
for j in range(0,no_of_observations):
    if(morekids[j]==1):
        workedmCase3[j]=workedm[j]*2
        weeksmCase3[j]=weeksm[j]*2
        hoursmCase3[j]=hoursm[j]*2
        incomemCase3[j]=incomem[j]*2
        faminclCase3[j]=faminclCase3[j]*2
    else:
        workedmCase3[j]=workedm[j]*-2
        weeksmCase3[j]=weeksm[j]*-2
        hoursmCase3[j]=hoursm[j]*-2
        incomemCase3[j]=incomem[j]*-2
        faminclCase3[j]=faminclCase3[j]*-2
        
        



indexOfAllWomen = []
for j in range(no_of_observations):
    if agem[j] >= 21 and agem[j] <=35 and kidcount[j] >= 2 and ageq2nd[j] > 4 and agefstm[j] >=15 \
       and aage[j] == 0 and asex[j] == 0 and aqtrbrth[j] == 0  and asex2nd[j] == 0 and aage2nd[j] == 0:
            indexOfAllWomen.append(j)
            



indexMarriedWoman = []
for j in range(no_of_observations):
    if (agem[j] >= 21 and agem[j] <=35 and kidcount[j] >= 2 and ageq2nd[j] > 4 \
        and agefstm[j] >=15 and aage[j] == 0 and asex[j] == 0\
        and aqtrbrth[j] == 0  and asex2nd[j] == 0 and aage2nd[j] == 0 and msample[j]==1):
            indexMarriedWoman.append(j)
            



indexWhereMorekids = []
indexMorekids00 = []
for j in range(no_of_observations):
    if (agem[j] >= 21 and agem[j] <=35 and kidcount[j] >= 2 and ageq2nd[j] > 4 \
        and agefstm[j] >=15 and aage[j] == 0 and asex[j] == 0\
        and aqtrbrth[j] == 0  and asex2nd[j] == 0 and aage2nd[j] == 0):
            if(morekids[j]==0):
                indexWhereMorekids.append(j)
            else:
                indexMorekids00.append(j)
                
morekidsMarried0 = []
moreKidsMarried1 = []
for ind_i in range(no_of_observations):
    if (agem[ind_i] >= 21 and agem[ind_i] <=35 and kidcount[ind_i] >= 2 and ageq2nd[ind_i] > 4 \
        and agefstm[ind_i] >=15 and aage[ind_i] == 0 and asex[ind_i] == 0\
        and aqtrbrth[ind_i] == 0  and asex2nd[ind_i] == 0 and aage2nd[ind_i] == 0 and msample[ind_i]==1):
            if(morekids[ind_i]==0):
                morekidsMarried0.append(ind_i)
            else:
                moreKidsMarried1.append(ind_i)


if not os.path.exists('result_trees'):
    os.makedirs('result_trees')



def data_fit(x_tuple):
    len_col = len(x_tuple)
    len_row = np.asarray(x_tuple[0]).shape[0]
    out = np.zeros((len_row,len_col))
    for i in range(0,len_col):       
        out[:,i] = np.asarray(x_tuple[i])   
    return out

def decision_tree(x_final, y,title):    
    x_train,x_test,y_train,y_test = train_test_split(x_final,y,test_size=0.2,random_state=0)
    
    best_depth = best_score = 0
    for depth in range(2,15):
        dtr = tree.DecisionTreeRegressor(max_depth=depth)
        dtr.fit(x_train,y_train)    
        score=dtr.score(x_test,y_test)
        if(score>=best_score):
            best_score = score
            best_depth = depth
    
    dtr = tree.DecisionTreeRegressor(max_depth=best_depth)
    dtr.fit(x_train,y_train)
    dot_data = tree.export_graphviz(dtr, out_file=None, 
                     filled=True, rounded=True,  
                     special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render('./result_trees/'+title+'_'+str(best_depth))
    
    return dtr




def model_fit(x_tuple,y):   
    x_final = data_fit(x_tuple)   
    out = linear_model.LinearRegression().fit(x_final, y)    
    return out.coef_


print('Table 7 Result  C1')
obs_indx = indexOfAllWomen

effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),workedm[obs_indx])
print('Worked for pay: ',effctive_w[0])
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),weeksm[obs_indx])
print('Weeks worked: ',effctive_w[0])
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),hoursm[obs_indx])
print('Hours/week: ',effctive_w[0])
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),incomem[obs_indx])
print('Labor income: ',effctive_w[0])
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),famincl[obs_indx])
print('ln(Family income): ',effctive_w[0]) 


print('Table 7 Result C4')
obs_indx = indexMarriedWoman
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),workedm[obs_indx])
print('Worked for pay: ',effctive_w[0])
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),weeksm[obs_indx])
print('Weeks worked: ',effctive_w[0])
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),hoursm[obs_indx])
print('Hours/week: ',effctive_w[0])
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),incomem[obs_indx])
print('Labor income: ',effctive_w[0])
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),famincl[obs_indx])
print('ln(Family income): ',effctive_w[0]) 


print('Table 7 Result C7')
obs_indx = indexMarriedWoman
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),workedd[obs_indx])
print('Worked for pay: ',effctive_w[0])
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),weeksd[obs_indx])
print('Weeks worked: ',effctive_w[0])
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),hoursd[obs_indx])
print('Hours/week: ',effctive_w[0])
effctive_w = model_fit((morekids[obs_indx],agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]),incomed[obs_indx])
print('Labor income: ',effctive_w[0])



print('table6,c2')
obs_indx = indexOfAllWomen
weight_params = model_fit((boy1st[obs_indx],boy2nd[obs_indx],samesex[obs_indx],agem1[obs_indx],\
            agefstm[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),morekids[obs_indx])
print("boyFirst",weight_params[0],"\tboySecond",weight_params[1],"\tIVSamesex",weight_params[2])
print('table6,c4')

obs_indx = indexMarriedWoman
weight_params = model_fit((samesex[obs_indx],),morekids[obs_indx])
print('Samesex: ',weight_params[0])
print('table6,c5')
obs_indx = indexMarriedWoman

weight_params = model_fit((boy1st[obs_indx],boy2nd[obs_indx],samesex[obs_indx],agem1[obs_indx],\
            agefstm[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),morekids[obs_indx])

print("boyFirst",weight_params[0],"\tboySecond",weight_params[1],"\tIVSamesex",weight_params[2])



print('table6,c1')
obs_indx = indexOfAllWomen
weight_params = model_fit((samesex[obs_indx],),morekids[obs_indx])
print('Samesex: ',weight_params[0])
print('table6,c2')

obs_indx = indexOfAllWomen

weight_params = model_fit((boy1st[obs_indx],boy2nd[obs_indx],samesex[obs_indx],agem1[obs_indx],\
            agefstm[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),morekids[obs_indx])

print("boyFirst",weight_params[0],"\tboySecond",weight_params[1],"\tIVSamesex",weight_params[2])


print('table6,c4')
obs_indx = indexMarriedWoman
weight_params = model_fit((samesex[obs_indx],),morekids[obs_indx])
print('Samesex: ',weight_params[0])

print('table6,c5')
obs_indx = indexMarriedWoman
weight_params = model_fit((boy1st[obs_indx],boy2nd[obs_indx],samesex[obs_indx],agem1[obs_indx],\
            agefstm[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]),morekids[obs_indx])

print("boyFirst",weight_params[0],"\tboySecond",weight_params[1],"\tIVSamesex",weight_params[2])



"""Case1- With treatment 0 and 1
	Case2- TOT
	case3- Tau_hat
 """

obs_indx = indexWhereMorekids
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,workedm[obs_indx],\
      title='case1_allTreat=0_workedForPay')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,weeksm[obs_indx],\
      title='case1_allTreat=0_weeksWorked')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,hoursm[obs_indx],\
      title='case1_allTreat=0_hoursPerWeek')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,incomem[obs_indx],\
      title='case1_allTreat=0_laborIncome')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,famincl[obs_indx],\
      title='case1_allTreat=0_familyIncome')
    



obs_indx = indexMorekids00
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,workedm[obs_indx],\
      title='case1_allTreat=1_workedForPay')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,weeksm[obs_indx],\
      title='case1_allTreat=1_weeksWorked')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,hoursm[obs_indx],\
      title='case1_allTreat=1_hoursPerWeek')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,incomem[obs_indx],\
      title='case1_allTreat=1_laborIncome')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,famincl[obs_indx],\
      title='case1_allTreat=1_familyIncome')

print('Making Decision Trees for Treatment 0_marriedHusband')


obs_indx = indexWhereMorekids
decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,workedd[obs_indx],\
      title='case1_treatMarriedHusband=0_workedForPay')
    
decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,weeksd[obs_indx],\
      title='case1_treatMarriedHusband=0_weeksWorked')

decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,hoursd[obs_indx],\
      title='case1_treatMarriedHusband=0_hoursPerWeek')
    
decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,incomed[obs_indx],\
      title='case1_treatMarriedHusband=0_laborIncome')

decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,famincl[obs_indx],\
      title='case1_treatMarriedHusband=0_familyIncome')


obs_indx = morekidsMarried0
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,workedm[obs_indx],\
      title='case1_treatMarried=0_workedForPay')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,weeksm[obs_indx],\
      title='case1_treatMarried=0_weeksWorked')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,hoursm[obs_indx],\
      title='case1_treatMarried=0_hoursPerWeek')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,incomem[obs_indx],\
      title='case1_treatMarried=0_laborIncome')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,famincl[obs_indx],\
      title='case1_treatMarried=0_familyIncome')
    



print('Decision Tree result: treatMarriedHusband=1 ')


obs_indx = indexMorekids00
decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,workedm[obs_indx],\
      title='case1_treatMarriedHusband=1_workedForPay')
    
decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,weeksm[obs_indx],\
      title='case1_treatMarriedHusband=1_weeksWorked')

decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,hoursm[obs_indx],\
      title='case1_treatMarriedHusband=1_hoursPerWeek')
    
decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,incomem[obs_indx],\
      title='case1_treatMarriedHusband=1_laborIncome')

decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,famincl[obs_indx],\
      title='case1_treatMarriedHusband=1_familyIncome')




obs_indx = moreKidsMarried1
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,workedm[obs_indx],\
      title='case1_treatMarried=1_workedForPay')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,weeksm[obs_indx],\
      title='case1_treatMarried=1_weeksWorked')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,hoursm[obs_indx],\
      title='case1_treatMarried=1_hoursPerWeek')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,incomem[obs_indx],\
      title='case1_treatMarried=1_laborIncome')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,famincl[obs_indx],\
      title='case1_treatMarried=1_familyIncome')
        



print('Decision Tree result: alldworkedForPay ')


obs_indx = indexOfAllWomen
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,workedmCase3[obs_indx],\
      title='case2_alldworkedForPay')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,weeksmCase3[obs_indx],\
      title='case2_alldweeksWorked')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,hoursmCase3[obs_indx],\
      title='case2_alldhoursPerWeek')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,incomemCase3[obs_indx],\
      title='case2_alldlaborIncome')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,faminclCase3[obs_indx],\
      title='case2_alldfamilyIncome')
        
        
print('Decision Tree result: marriedworkedForPay ')


obs_indx = indexOfAllWomen
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,workedmCase3[obs_indx],\
      title='case2_marriedworkedForPay')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,weeksmCase3[obs_indx],\
      title='case2_marriedweeksWorked')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,hoursmCase3[obs_indx],\
      title='case2_marriedhoursPerWeek')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,incomemCase3[obs_indx],\
      title='case2_marriedlaborIncome')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,faminclCase3[obs_indx],\
      title='case2_marriedfamilyIncome')
        
        

print('Decision Tree result: marriedHusband_workedForPay ')


obs_indx = indexOfAllWomen
decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,workedmCase3[obs_indx],\
      title='case2_marriedHusband_workedForPay')
    
decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,weeksmCase3[obs_indx],\
      title='case2_marriedHusband_weeksWorked')

decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,hoursmCase3[obs_indx],\
      title='case2_marriedHusband_hoursPerWeek')
    
decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,incomemCase3[obs_indx],\
      title='case2_marriedHusband_laborIncome')

decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,faminclCase3[obs_indx],\
      title='case2_marriedHusband_familyIncome')


print('Decision Tree result: alldworkedForPay')

obs_indx = indexOfAllWomen
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,workedm[obs_indx],\
      title='case3_alldworkedForPay')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,weeksm[obs_indx],\
      title='case3_alldweeksWorked')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,hoursm[obs_indx],\
      title='case3_alldhoursPerWeek')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,incomem[obs_indx],\
      title='case3_alldlaborIncome')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,famincl[obs_indx],\
      title='case3_alldfamilyIncome')

print('Plain Decision Tree for Tau_Hat_Married')

obs_indx = indexMarriedWoman
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,workedm[obs_indx],\
      title='case3_marriedworkedForPay')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,weeksm[obs_indx],\
      title='case3_marriedweeksWorked')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,hoursm[obs_indx],\
      title='case3_marriedhoursPerWeek')
    
decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,incomem[obs_indx],\
      title='case3_marriedlaborIncome')

decision_tree(data_fit((agem1[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackm[obs_indx],hispm[obs_indx],othracem[obs_indx]))\
    ,famincl[obs_indx],\
      title='case3_marriedfamilyIncome')

print('Plain Decision Tree for Tau_Hat_ALL_Husband')

obs_indx = indexMarriedWoman
decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,workedm[obs_indx],\
      title='case3_marriedHusband_workedForPay')
    
decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,weeksm[obs_indx],\
      title='case3_marriedHusband_weeksWorked')

decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,hoursm[obs_indx],\
      title='case3_marriedHusband_hoursPerWeek')
    
decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,incomem[obs_indx],\
      title='case3_marriedHusband_laborIncome')

decision_tree(data_fit((aged[obs_indx],agefstm[obs_indx],boy1st[obs_indx],\
            boy2nd[obs_indx],blackd[obs_indx],hispd[obs_indx],othraced[obs_indx]))\
    ,famincl[obs_indx],\
      title='case3_marriedHusband_familyIncome')