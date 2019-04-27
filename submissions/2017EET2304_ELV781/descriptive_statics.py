import numpy as np
import matplotlib.pyplot as plt
# from pdb import set_trace as trace
import pandas as pd
import gc
from sas7bdat import SAS7BDAT
from sklearn import linear_model
from sklearn import tree
from sklearn.model_selection import train_test_split
import os
import graphviz

# data_80 = [] 
# with SAS7BDAT('./AngEv98/m_d_806.sas7bdat', skip_header=False) as reader:
#     for row in reader:
#         data_80.append(row)

# np.save('data_80.npy',np.asarray(data_80))
# del data_80
# gc.collect()

#Data Loading
#Data Preprocessing(Converting data from string to numerical value and replacing none type with appropriate value)
from data_preprocessing import *

cont = kids_cont = morekids2 = boy1st = boy2nd = boys2 = girls2 = samesex = age = twin = age_at_first_b = 0
weeks_worked = Worked_for_pay = hours_per_week = labor_income = family_income = weeks_worked_hus = 0

for j in range(no_of_observations):
    if agem[j] >= 21 and agem[j] <=35 and kidcount[j] >= 2 and ageq2nd[j] > 4 and agefstm[j] >=15 and \
       aage[j] == 0 and asex[j] == 0 and aqtrbrth[j] == 0  and asex2nd[j] == 0 and aage2nd[j] == 0:
        cont += 1
        
        kids_cont += kidcount[j]
        if kidcount[j] > 2:
            morekids2 += 1

        if(sexk[j] == 0):
            boy1st += 1

        if(sex2nd[j] == 0):
            boy2nd +=1

        if(sexk[j] == 0 and sex2nd[j] ==0):
            boys2 +=1

        if(sexk[j] == 1 and sex2nd[j] ==1):
            girls2 +=1

        if(sexk[j] == 1 and sex2nd[j] ==1)or (sexk[j] == 0 and sex2nd[j]==0):
            samesex += 1
        age += agem[j]

        if(ageq3rd[j] == ageq2nd[j]) and (ageq3rd[j] != -1 and ageq2nd[j] !=-11):
            twin +=1

        age_at_first_b += agefstm[j]

        if(weeksm[j] > 0):
            weeks_worked += weeksm[j]
            Worked_for_pay += 1

        hours_per_week += hoursm[j]

        labor_income += incomem[j]

        family_income += faminc[j]


print('\n--------------------------------------------------')
print('------------- Results for ALL WOMEN --------------')
print('--------------------------------------------------\n')

print("Children ever born = {}\nMore than 2 children = {}\nBoy Ist = {}\nBoy 2nd = {}".format(\
    kids_cont/cont, morekids2/cont, boy1st/cont, boy2nd/cont))
print("Two boys = {}\nTwo girls ={}\nSame sex = {}\nTwins-2 = {}".format(\
    boys2/cont, girls2/cont,samesex/cont, twin/cont))

print("Age = {} \nAge at first birth = {} \nWorked for pay = {} \nWeeks worked = {}".format(\
    age/cont, age_at_first_b/cont, Worked_for_pay/cont, weeks_worked/cont))
print("Hours/week = {}\nLabor income = {}\nFamily income = {}".format(hours_per_week/cont,\
     labor_income/cont, family_income/cont))



msample = np.zeros(no_of_observations)
for j in range(no_of_observations):
    if (aged[j] > 0 and timesmar[j] == 1 and marital[j] == 0 and illegit[j]==0 \
        and agefstd[j] >= 15 and agefstm[j] >= 15):
        msample[j] = 1


count = morekids2 = boy1st = boy2nd = boys2 = girls2 = samesex = age = twin = age_at_first_b = weeks_worked = 0
Worked_for_pay = hours_per_week = labor_income = family_income = non_family_income = weeks_worked_hus = 0
age_dad = hours_per_week_dad = labor_income_dad = age_at_first_b_dad = Worked_for_pay_hus = kids_count = 0

for j in range(no_of_observations):
    if agem[j] >= 21 and agem[j] <=35 and kidcount[j] >= 2 and ageq2nd[j] > 4 and agefstm[j] >=15 \
       and aage[j] == 0 and asex[j] == 0 and aqtrbrth[j] == 0  and asex2nd[j] == 0 and aage2nd[j] == 0:
        if msample[j] == 1:

            kids_count  += kidcount[j]
            count +=1

            if kidcount[j] > 2:
                morekids2 += 1

            if(sexk[j] == 0):
                boy1st += 1

            if(sex2nd[j] == 0):
                boy2nd +=1

            if(sexk[j] == 0 and sex2nd[j] ==0):
                boys2 +=1

            if(sexk[j] == 1 and sex2nd[j] ==1):
                girls2 +=1

            if(sexk[j] == 1 and sex2nd[j] ==1)or (sexk[j] == 0 and sex2nd[j]==0):
                samesex += 1

            age += agem[j]

            if(ageq3rd[j] == ageq2nd[j]) and (ageq3rd[j] != -1 and ageq2nd[j] !=-11):
                twin +=1

            age_at_first_b += agefstm[j]
            
            age_at_first_b_dad += agefstd[j]

            age_dad += aged[j]

            if(weeksm[j] > 0):
                weeks_worked += weeksm[j]
                Worked_for_pay += 1

            if(weeksd[j] > 0):
                weeks_worked_hus += weeksd[j]
                Worked_for_pay_hus += 1
                
            hours_per_week += hoursm[j]

            hours_per_week_dad += hoursd[j]

            labor_income += incomem[j]

            labor_income_dad += incomed[j]

            family_income += faminc[j]

            non_family_income += nonmomi[j]



print('\n--------------------------------------------------')
print('------------- Results for MARRIED WOMEN ----------')
print('--------------------------------------------------\n')

print("Children ever born = {}\nMore than 2 children = {}\nBoy 1st = {}\nBoy 2nd = {}".format(\
    kids_count/count, morekids2/count, boy1st/count, boy2nd/count))
print("Two boys = {}\nTwo girls ={}\nSame sex = {}\nTwins-2 = {}".format(\
    boys2/count, girls2/count,samesex/count, twin/count))

print("Age = {} \nAge atfirst birth = {} \nWorked for pay = {} \nWeeks worked = {}".format(\
    age/count, age_at_first_b/count, Worked_for_pay/count, weeks_worked/count))
print("Hours/week = {}\nLabor income = {}\nFamily income = {}\nNon-wife income ={}".format(hours_per_week/count,\
     labor_income/count, family_income/count, non_family_income/count))
print("Age dad = {} \nAge dad at first birth = {} \nWorked for pay = {} \nWeeks worked = {}".format(\
    age_dad/count, age_at_first_b_dad/count, Worked_for_pay_hus/count, weeks_worked_hus/count))
print("Hours/week = {}\nLabor income = {}".format(\
      hours_per_week_dad/count,labor_income_dad/count))

