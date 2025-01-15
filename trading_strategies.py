# This is the trading strategy. 
# we can trade after each predict > probability threshold.
# For each prediction, assume data is x1, x2, ..., xlook_back, xlook_back+1, xlook_back+2, ..., xlook_back + forecast_size.
# we use x1, x2, ..., xlook_back to make a prediction. The trading start at x_lookback+1.

import numpy as np


def one_share_trade(current_price, future_price, trading_type, pred_prob, prob_threshold, profit_threshold, share, fees):
    # trading_type : long 
    m, forecast_size = future_price.shape  # future price is a matrix of future price for each xlook_back, xlook_back+1, xlook_back+2, ..., xlook_back + forecast_size..
    trading_index = 0 # first prediction
    profit = np.empty(m + forecast_size + 1) # profit history
    profit[0] = 0 # start at 0
    number_of_trades = 0
    number_pos_trades = 0
    invested_money = 0
    while trading_index < m:
        if pred_prob[trading_index] >= prob_threshold: # if probability is high enough
            position_open = True # open position
            number_of_trades = number_of_trades +1
            holding_time = 0 # stock holding time <= 100
            for i in range(forecast_size): # from e101 to e200
                if ((trading_type == 'long' and future_price[trading_index, i] >= current_price[trading_index] + profit_threshold) or 
                (trading_type == 'short' and future_price[trading_index, i] + profit_threshold <= current_price[trading_index])):
                # if we can make a profit, close position.
                    position_open = False
                    holding_time = i + 1
                    break
            
            # if we cannot make a profit within forecast_size timestep, close position anyway.
            if position_open:
                position_open = False
                holding_time = forecast_size
            
            # update profit history 
            profit[trading_index + 1: trading_index + holding_time] = profit[trading_index]
            if trading_type == 'long':
                profit[trading_index + holding_time] = profit[trading_index] + share*(future_price[trading_index, holding_time - 1] - (1+fees)*current_price[trading_index])
                invested_money = invested_money + share*(1+fees)*current_price[trading_index]
                print('profit of trade is :',share*(future_price[trading_index, holding_time - 1] - (1+fees)*current_price[trading_index]))
                if((future_price[trading_index, holding_time - 1] - current_price[trading_index]) >=0):
                    number_pos_trades = number_pos_trades+1
            else:
                profit[trading_index + holding_time] = profit[trading_index] - future_price[trading_index, holding_time - 1] + current_price[trading_index]
            trading_index = trading_index + holding_time
        else: # probability is low so do nothing
            profit[trading_index + 1] = profit[trading_index]
            trading_index = trading_index + 1

    profit[trading_index + 1:] = profit[trading_index]
    print('number of trades is:', number_of_trades)
    print('ratio of positive trades is:', float(number_pos_trades/number_of_trades))
    print('ROI is:',float(100*profit[-1]/invested_money))
    return profit    
