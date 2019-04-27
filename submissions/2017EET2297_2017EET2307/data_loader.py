#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:49:05 2019

@author: aravind
"""

import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as trace
import pandas as pd
import gc

#data_90=pd.read_sas('./AngEv98/m_d_903.sas7bdat')

from sas7bdat import SAS7BDAT


data_80 = []

with SAS7BDAT('./AngEv98/m_d_806.sas7bdat', skip_header=False) as reader:
    for row in reader:
        #print( row)
        data_80.append(row)

np.save('data_80.npy',np.asarray(data_80))

del data_80
gc.collect()

data_90 = []

with SAS7BDAT('./AngEv98/m_d_903.sas7bdat', skip_header=False) as reader:
    for row in reader:
        #print( row)
        data_90.append(row)

np.save('data_90.npy',np.asarray(data_90))


print('Done.....')
