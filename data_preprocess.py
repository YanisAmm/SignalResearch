# functions for data preprocessing

import numpy as np
from sklearn.preprocessing import StandardScaler

# extract the csv file going to work with, for sake of memory and time, we will reduce
# the dimension of the inputs
def extract(data_original):
    data = data_original.iloc[:,5].to_numpy()
    data = np.concatenate((data.reshape([-1,1]),data_original.iloc[:,6].to_numpy().reshape([-1,1])), axis=1)
    data = np.concatenate((data,data_original.iloc[:,3].to_numpy().reshape([-1,1])), axis=1)
    data = np.concatenate((data,data_original.iloc[:,4].to_numpy().reshape([-1,1])), axis=1)
    for i in range(1,5):
        data = np.concatenate((data,data_original.iloc[:,5+4*i].to_numpy().reshape([-1,1])), axis=1)
        data = np.concatenate((data,data_original.iloc[:,6+4*i].to_numpy().reshape([-1,1])), axis=1)
        data = np.concatenate((data,data_original.iloc[:,3+4*i].to_numpy().reshape([-1,1])), axis=1)
        data = np.concatenate((data,data_original.iloc[:,4+4*i].to_numpy().reshape([-1,1])), axis=1)
    
    return data

# extract the time points of the series
def get_time_point(data_original):
    # convert the time to seconds 
    
    time_serie = data_original.iloc[:,0]
    data = np.zeros((len(time_serie), 1))
    for i in range(len(time_serie)):
        #Depending on the day, the time point is increased by 0, 1 or 2 day
        if(float(time_serie[i][8:10]) == 19):
            count = 0
        if(float(time_serie[i][8:10]) == 20):
            count = 24*3600
        if(float(time_serie[i][8:10]) == 21):
            count = 24*3600*2
        hour = float(time_serie[i][11:13])
        minute = float(time_serie[i][14:16])
        second = float(time_serie[i][17:19])
        ms = float(time_serie[i][20:23])
        data[i] = count + 3600*hour + 60*(minute-5) + second + ms/1000
    return data

# turn the unevenly spaced time series into evenly spaced time series
def rescale_data(data_order, data_message, data_level, time_window):
    # start time is 5am641ms and end time is 3 days later
    start_time = 0.641 
    end_time = 172800 
    data = np.concatenate((data_message.reshape([-1, 1]), data_order), axis=1)
    data_new = np.empty([int((end_time - start_time) / time_window) + 1, int(4 * data_level)])
    time_scale = np.arange(start_time, end_time, time_window)
    k = 0
    for i in range(time_scale.shape[0]):
        while k < data.shape[0] and data[k, 0] < time_scale[i]:
            k = k + 1
        
        data_new[i] = data[k - 1, 1:]


    return data_new

# generate (x, y) pair for cnn model. x: k by look back range by n. y: k by forecast size by n.
# n is 4 times level. k is the num_rows in the code.
def generate_data(data, forecast_size, look_back):
    m, n = data.shape
    num_rows = m - look_back - forecast_size
    data_x = np.empty([num_rows, look_back, n])
    data_y = np.empty([num_rows, forecast_size, n])

    for i in range(num_rows):
        data_x[i, :, :] = data[i: i + look_back, :]
        data_y[i, :, :] = data[i + look_back: i + look_back + forecast_size, :]

    return data_x, data_y
    

# split the data 1:1 for train and test
def train_test_split(data_x, data_y, forecast_size):
    n = data_x.shape[0]
    #Train on the first two days and test on the last day 
    train_num = int(n * 0.66)
    train_x = data_x[:train_num]
    train_y = data_y[:train_num]
    test_x = data_x[train_num + forecast_size:]
    test_y = data_y[train_num + forecast_size:]
    return train_x, train_y, test_x, test_y
    
# generate matrix price, volumes and target probability
def data_for_trading_model(data_x, data_y, profit_threshold, share, fees): 
    n = data_x.shape[0]
    price = np.empty_like(data_x[:,:,0::2])
    price_ask = data_x[:,:,0::4]
    price_bid = data_x[:,:,2::4]
    np.copyto(price, np.concatenate((price_bid[:,:,::-1], price_ask), axis=-1))
    volume = np.empty_like(data_x[:,:,1::2])
    volume_ask = data_x[:,:,1::4]
    volume_bid = data_x[:,:,3::4]
    np.copyto(volume, np.concatenate((volume_bid[:,:,::-1], volume_ask), axis=-1))
    volume = np.log(volume)
    profit = share*(np.max(data_y[:, :, 2], axis=1) - (1+fees)*data_x[:, -1, 0])
    
        
    prob = profit >= profit_threshold
    for i in range(n):
        price_mean = np.mean(price[i, :, :], axis=0)
        price[i, :, :] = price[i, :, :] - price_mean
        volume_mean = np.mean(volume[i, :, :], axis=0)
        volume[i, :, :] = volume[i, :, :] - volume_mean
    
    return price, volume, prob    


    
