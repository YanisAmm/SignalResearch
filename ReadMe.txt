This folder contain 6 python files

- main_trading.py: This file has to be executed to i) clean and preprocess the data, ii) create the machine learning layer, iii) train the model and iv) initiate the trading.

- data_preprocess.py: This file contains the functions to preprocess and clean the data. This file is called by main_trading.py, create_time_series and create_orderbook.py.

- deep_learning_models.py: This file contains the machine learning layer used by main_trading.py.

- trading_ strategies.py: This file performs the trading operations. It is used by main_trading.py.

- create_ orderbook.py: This file cleans the orderbook to reduce the amount of data to process and creates a new file called data_ order.csv.

- create_ time_ series.py: This file converts the matching time main_trading.py in seconds and write the results in time.csv.

As time.csv and data_order.csv have already been created, it is possible to directly run main_trading.py.

This folder also contains the report: Queueco_Written_Assessment.pdf 

The data folder contains:

- exchange1_202105XX.csv : Original data for the first instrument with XX = 19 || 20 || 21
- time.csv, a column vector that contains the time points in seconds
- data_order.csv, a matrix that contains the level 5 order_book, extracted from the 10 order_book given

----------------------------------------Instructions-------------------------------------------------------------------------

1> import exchange1_20210519.csv, exchange1_20210520.csv and exchange1_20210521.csv in the folder data

2> Execute with python create_orderbook.py to create the file data_order.csv (this action can take few minutes)

3> Execute with python create_time_series.py to create the file time.csv (this action can take few minutes)

4> Execute with python main_trading.py to run the trading algorithm (this action can take few minutes)

