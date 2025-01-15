#functions to create csv file extracted from the data

import numpy as np
import matplotlib.pyplot as plt
import data_preprocess
import pandas as pd


first_day = pd.read_csv('data/exchange1_20210519.csv')
first_day_order = data_preprocess.extract(first_day)
del first_day

second_day = pd.read_csv('data/exchange1_20210520.csv')
second_day_order = data_preprocess.extract(second_day)
del second_day

third_day = pd.read_csv('data/exchange1_20210521.csv')
third_day_order = data_preprocess.extract(third_day)
del third_day

three_day = np.concatenate((first_day_order, second_day_order,third_day_order), axis=0)

np.savetxt("data/data_order.csv", three_day, 
              delimiter = ",")
