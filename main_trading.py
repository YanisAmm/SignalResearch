


import numpy as np
import matplotlib.pyplot as plt
import data_preprocess
import deep_learning_models
import trading_strategies

# load the order and message book. We use the level 5 limit order book data.'
data_level = 5


data_order = np.loadtxt('data/data_order.csv', delimiter=',')
data_message = np.loadtxt('data/time.csv', delimiter=',')

# set time window, forecast size, and look back range, share and fees.
time_window = 2
forecast_size = 50
look_back = 100
share = 0.1
fees = 0.00001
# turn data into evenly spaced, then split the one day dataset into two half day datasets.
evenly_spaced_data = data_preprocess.rescale_data(data_order, data_message, data_level, time_window)
data_x, data_y = data_preprocess.generate_data(evenly_spaced_data, forecast_size, look_back)
del evenly_spaced_data


# set profit threshold and generate the data for training and testing.
profit_threshold_for_model = 10
train_x, train_y, test_x, test_y = data_preprocess.train_test_split(data_x, data_y, forecast_size)
train_price, train_volume, train_prob = data_preprocess.data_for_trading_model(train_x, train_y, profit_threshold_for_model, share, fees)
test_price, test_volume, test_prob = data_preprocess.data_for_trading_model(test_x, test_y, profit_threshold_for_model, share, fees)


# set batch size and learning rate. 
# train the model and predict the probability of making a profit by longing one share of stock
batch_size = 16
learning_rate = 0.001
cnn_model_for_long_1 = deep_learning_models.cnn_classification_trading_model(look_back, data_level, learning_rate)
cnn_model_for_long_1.fit([train_price, train_volume], train_prob, epochs=1, batch_size=batch_size,
                        validation_data=[[test_price, test_volume], test_prob])
pred_long_prob_1 = cnn_model_for_long_1.predict([test_price, test_volume]).flatten()


# set the probability threshold and profit target.
# use the predicted probability to trade.
prob_threshold = 0.9
profit_threshold = 10
cnn_long_1_profit = trading_strategies.one_share_trade(test_x[:, -1, 0], test_y[:, :, 2], 'long', pred_long_prob_1, prob_threshold, profit_threshold, share, fees)
del train_price, train_volume, train_prob, test_price, test_volume, test_prob





# print the final profit
print('long profit: ', cnn_long_1_profit[-1])
