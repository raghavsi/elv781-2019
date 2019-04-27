from data_loader import *

sexk = d[:, SEXK]
sex2nd = d[:, SEX2ND]
ageq3rd = d[:, AGEQ3RD]
qtrbthd = d[:, QTRBTHD]
agem = d[:, AGEM]
aged = d[:, AGED]
kidcount = d[:, KIDCOUNT]
ageq2nd = d[:, AGEQ2ND]
ageqk = d[:, AGEQK]
asex = d[:, ASEX]
aage = d[:, AAGE]
aqtrbrth = d[:, AQTRBRTH]
asex2nd = d[:, ASEX2ND]
aage2nd = d[:, AAGE2ND]
qtrmar = d[:, QTRMAR]
yobm = d[:, YOBM]
agemar = d[:, AGEMAR]
qtrbthm = d[:, QTRBTHM]
qtrbkid = d[:, QTRBKID]
yobk = d[:, YOBK]
marital = d[:, MARITAL]
timesmar = d[:, TIMESMAR]



weeksm = [int(x) for x in d[:, WEEKSM]]
weeksd = [int(x) if x != '' else 0 for x in d[:, WEEKSD]]
hoursm = [int(x) for x in d[:, HOURSM]]
hoursd = [int(x) if x != '' else 0 for x in d[:, HOURSD]]
income1d = [int(x) if x != '' else -1 for x in d[:, INCOME1D]]
income2d = [int(x) if x != '' else -1 for x in d[:, INCOME2D]]
faminc = [int(x) for x in d[:, FAMINC]]
income1m = [int(x) for x in d[:, INCOME1M]]
income2m = [int(x) for x in d[:, INCOME2M]]
timesmar = [int(x) for x in timesmar]
marital = [int(x) for x in marital]
qtrbthm = [int(x) for x in qtrbthm]
qtrmar = [int(x) for x in qtrmar]
agemar = [int(x) for x in agemar]
qtrbkid = [int(x) for x in qtrbkid]
aged = [int(x) if x !='' else 0 for x in aged]
agem = [int(x) for x in agem]
qtrbthd = [int(x) if x !='' else 0 for x in qtrbthd]
aage = [int(x) for x in aage]
ageq3rd = [int(x) if x != None else -1 for x in ageq3rd]
ageq2nd = [int(x) if x !=None else -1 for x in ageq2nd]
sex2nd = [int(x) if x!='' else -1 for x in sex2nd]
sexk = [int(x) for x in sexk]
asex = [int(x) for x in asex]
aqtrbrth = [int(x) for x in aqtrbrth]
asex2nd = [int(x) if x != '' else -1 for x in asex2nd]
aage2nd = [int(x) if x != '' else -1 for x in asex2nd]



illegit=np.zeros(no_of_observations);
yom = np.zeros(no_of_observations)
dom_q = np.zeros(no_of_observations)
do1b_q = np.zeros(no_of_observations)
for j in range(no_of_observations):
    if qtrmar[j]>0 :
        qtrmar[j]=qtrmar[j]-1
        if qtrbthm[j] <= qtrmar[j]:
            yom[j]=yobm[j]+agemar[j]
        elif qtrbthm[j] > qtrmar[j] :
            yom[j] = yobm[j] + agemar[j] + 1
        dom_q[j] = yom[j] + (qtrmar[j]/4)
        do1b_q[j] = yobk[j] + ((qtrbkid[j])/4)
        if (dom_q[j] - do1b_q[j])>0 :
            illegit[j]=1

yobd = np.zeros(no_of_observations)
for j in range(no_of_observations): 
    if qtrbthd[j] == 0 :
        yobd[j] = 80-aged[j];
    else:
        yobd[j] = 79-aged[j];

ageqm = np.zeros(no_of_observations)
ageqd = np.zeros(no_of_observations)
agefstm = np.zeros(no_of_observations)
agefstd = np.zeros(no_of_observations)
for j in range(no_of_observations):
    ageqm[j] = 4*(80-yobm[j])-qtrbthm[j]-1;
    ageqd[j] = 4*(80-yobd[j])-qtrbthd[j]
    agefstm[j] = int((ageqm[j]- ageqk[j])/4)
    agefstd[j] = int((ageqd[j]-ageqk[j])/4)


incomed = np.zeros(no_of_observations)
incomem = np.zeros(no_of_observations)
famincl = np.zeros(no_of_observations)
nonmomi = np.zeros(no_of_observations)
nonmomil = np.zeros(no_of_observations)

for j in range(no_of_observations):
    incomed[j] = income1d[j] + max(0,income2d[j])
    incomem[j] = income1m[j] + max(0,income2m[j])
    incomem[j] = incomem[j]*2.099173554
    incomed[j] = incomed[j]*2.099173554
    faminc[j] = faminc[j]*2.099173554
    famincl[j] = np.log(max(faminc[j],1))
    nonmomi[j] = faminc[j] - income1m[j]*2.099173554
    nonmomil[j] = np.log(max(1,nonmomi[j]))


