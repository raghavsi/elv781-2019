import numpy as np



# data_80 = np.load('data_80.npy')
# header = data_80[0]
# d = data_80[1:,:]
# no_of_observations = d.shape[0]
# # print(no_of_observations)
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as trace
import pandas as pd
import gc

from sas7bdat import SAS7BDAT
data_80 = []
with SAS7BDAT('./AngEv98/m_d_806.sas7bdat', skip_header=False) as reader:
    for row in reader:
        #print( row)
        data_80.append(row)

np.save('data_80.npy',np.asarray(data_80))
gc.collect()


header = data_80[0]
d = data_80[1:,:]

AGEM = np.where(header== 'AGEM')[0][0]
AGED = np.where(header== 'AGED')[0][0]
KIDCOUNT = np.where(header== 'KIDCOUNT')[0][0]
AGEQ2ND = np.where(header== 'AGEQ2ND')[0][0]
AGEQK = np.where(header== 'AGEQK')[0][0]
ASEX = np.where(header== 'ASEX')[0][0]
AQTRBRTH = np.where(header== 'AQTRBRTH')[0][0]
ASEX2ND = np.where(header== 'ASEX2ND')[0][0]
AAGE2ND = np.where(header== 'AAGE2ND')[0][0]
QTRMAR = np.where(header== 'QTRMAR')[0][0]
YOBM = np.where(header== 'YOBM')[0][0]
AGEMAR = np.where(header== 'AGEMAR')[0][0]
QTRBTHM = np.where(header== 'QTRBTHM')[0][0]
QTRBKID = np.where(header== 'QTRBKID')[0][0]
YOBK = np.where(header== 'YOBK')[0][0]
QTRBTHD = np.where(header== 'QTRBTHD')[0][0]
AAGE = np.where(header== 'AAGE')[0][0]
AGEQ3RD = np.where(header== 'AGEQ3RD')[0][0]
SEXK = np.where(header== 'SEXK')[0][0]
SEX2ND = np.where(header== 'SEX2ND')[0][0]
WEEKSM = np.where(header== 'WEEKSM')[0][0]
WEEKSD = np.where(header== 'WEEKSD')[0][0]
HOURSM = np.where(header== 'HOURSM')[0][0]
HOURSD = np.where(header== 'HOURSD')[0][0]
INCOME1D = np.where(header== 'INCOME1D')[0][0]
INCOME2D = np.where(header== 'INCOME2D')[0][0]
FAMINC = np.where(header== 'FAMINC')[0][0]
INCOME1M = np.where(header== 'INCOME1M')[0][0]
INCOME2M = np.where(header== 'INCOME2M')[0][0]
MARITAL = np.where(header== 'MARITAL')[0][0]
TIMESMAR = np.where(header== 'TIMESMAR')[0][0]