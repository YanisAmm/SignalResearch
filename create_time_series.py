#functions to create csv file extracted from the data

import numpy as np
import matplotlib.pyplot as plt
import data_preprocess
import pandas as pd


first_day = pd.read_csv('data/exchange1_20210519.csv')
time_first_day = data_preprocess.get_time_point(first_day)

del first_day

second_day = pd.read_csv('data/exchange1_20210520.csv')
time_second_day = data_preprocess.get_time_point(second_day)

del second_day

third_day = pd.read_csv('data/exchange1_20210521.csv')
time_third_day = data_preprocess.get_time_point(third_day)

del third_day

three_day_time = np.concatenate((time_first_day, time_second_day,time_third_day), axis=0)

np.savetxt("data/time.csv", three_day_time, 
              delimiter = ",")