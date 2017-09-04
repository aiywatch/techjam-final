import pandas as pd
import numpy as np
import math
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta, FR
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers



SELECTED_CAT = ['Automobiles and Vehicles', 'Clothing Stores', 'Service Providers',
            'Transportation', 'Utilities']


ccinfo = pd.read_csv('data/cc_info.csv', index_col='card_no') #, parse_dates=['txn_date'])
cclog = pd.read_csv('data/cc_log.csv', parse_dates=['txn_dt'])
cat_map = pd.read_csv('data/Final_categories.csv') #, parse_dates=['txn_date'])

selected_cat_id = cat_map[cat_map['Categories'].isin(SELECTED_CAT)]


cclog.drop(['txn_tm', 'card_acpt_cty', 'card_type'], axis=1, inplace=True)
ccinfo.drop(['card_type', 'opn_dt', 'exp_dt', 'main_zip_cd'], axis=1, inplace=True)






cclog['month'] = cclog['txn_dt'].dt.month
cclog['year'] = cclog['txn_dt'].dt.year

cclog['month_rank'] = 0
cclog.loc[cclog['year'] == 2016, 'month_rank'] = cclog.loc[cclog['year'] == 2016, 'month']
cclog.loc[cclog['year'] == 2017, 'month_rank'] = cclog.loc[cclog['year'] == 2017, 'month'] + 12

cclog_filter_cat = cclog[cclog['mrch_tp_cd'].isin(selected_cat_id['MCC'].values)]
cclog_filter_cat['cat_name'] = ''


for i, cat_name in enumerate(SELECTED_CAT):
    temp = selected_cat_id[selected_cat_id['Categories'] == cat_name]
    select_row = cclog_filter_cat['mrch_tp_cd'].isin(temp['MCC'].values)
    cclog_filter_cat.loc[select_row, 'cat_name'] = cat_name


sort_time = cclog_filter_cat.sort_values(['card_no', 'cat_name', 'txn_dt'],ascending=False).groupby(['card_no', 'cat_name']).head(1)
sort_time['days_from_last'] = (datetime.datetime.now() - sort_time['txn_dt']).dt.days


month_group = cclog_filter_cat.groupby(['cat_name', 'month_rank', 'card_no'])

group_sum = month_group.sum()[['bill_amt']]
group_count = month_group.count()[['txn_dt']]

group_log = pd.merge(group_sum, group_count, how='outer',
                          left_index=True, right_index=True)

cclog_filter_cat_10up = cclog_filter_cat[cclog_filter_cat['month_rank'] > 13]
card_group = cclog_filter_cat_10up.groupby(['cat_name', 'card_no'])
group_card_sum = card_group.sum()[['bill_amt']]
group_card_count = card_group.count()[['txn_dt']]
group_agg = card_group.agg(['mean', 'count'])[['bill_amt']]

group_log_card = pd.merge(group_card_sum, group_card_count, how='outer',
                          left_index=True, right_index=True)



all_spend = pd.DataFrame()
for cat_name in SELECTED_CAT:
    temp = pd.DataFrame(index=ccinfo.index)
    temp['total_count'] = 0
    temp['total_spend'] = 0
    temp['mean'] = 0
    for month_rank in range(19):
        card_sum = group_log.loc[(cat_name, month_rank)]
        temp[['bill'+str(month_rank), 'count'+str(month_rank)]] = card_sum
        
    temp[['total_spend', 'total_count']] = group_log_card.loc[(cat_name)]
    temp['mean'] = group_agg.loc[(cat_name), ('bill_amt','mean')]
        
    sort_time_temp = sort_time[sort_time['cat_name'] == cat_name].set_index('card_no')
    temp['days_from_last'] = sort_time_temp['days_from_last']
    
    temp['cat_name'] = cat_name
    all_spend = all_spend.append(temp.reset_index(), ignore_index=True)














all_spend.fillna(0, inplace=True)

all_spend['cat_int'] = all_spend['cat_name'].apply(lambda x: SELECTED_CAT.index(x))

#all_spend = pd.get_dummies(all_spend, columns = ['cat_int'] )


dataset = pd.merge(ccinfo, all_spend, how='right',
                   left_index=True, right_on='card_no')


#X = dataset.drop(['bill18', 'count18', 'card_no', 'cat_name'], axis=1).values

############# start Model

X = dataset[['total_count', 'brth_estb_yr', 'gnd_ind', 'mean', 'bill6', 'count6', 'bill15', 'count15',
             'bill16', 'count16', 'bill17', 'count17', 'cat_int']]

#X['mean_count'] = (X['count15'] + X['count16'] + X['count17']) / 3
#X['mean_bill'] = (X['bill15'] + X['bill16'] + X['bill17']) / 3


X = pd.get_dummies(X, columns = ['cat_int'] ).values


y = dataset['count18'].values >= 1


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



#model_class = LGBMClassifier(n_estimators=400, num_leaves=80, 
#                             objective='sigmoid')
#model_class.fit(X_train, y_train, eval_metric="logloss")

from xgboost import XGBClassifier
model_class = XGBClassifier(min_child_weight=10, max_depth=3, gamma=1,
                           reg_alpha=0.015)

model_class.fit(X,y)


#y_pred = model_class.predict_proba(X_test)[:,1] > 0.15
#
#accuracy_score(y_test, y_pred)
#cm= classification_report(y_test, y_pred)


#feature_importances = model_class.feature_importances_

########

dataset_reg = dataset[dataset['count18'] > 0]


X = dataset_reg[['total_count', 'gnd_ind', 'mean', 'bill6', 'count6', 'bill15', 'count15',
             'bill16', 'count16', 'bill17', 'count17', 'cat_int']]

#X = dataset_reg[['total_count', 'gnd_ind', 'mean', 'count14', 'count15',
#              'count16',  'count17', 'cat_int']]

#X['mean_count'] = (X['count15'] + X['count16'] + X['count17']) / 3
#X['mean_bill'] = (X['bill15'] + X['bill16'] + X['bill17']) / 3

X = pd.get_dummies(X, columns = ['cat_int'] ).values


y = dataset_reg['count18']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)




model_reg_f = LGBMRegressor(n_estimators=100, num_leaves=80)
model_reg_f.fit(X, y) 

y_pred = model_reg_f.predict(X_test).round()
mean_absolute_error(y_test, y_pred)

#feature_importances = model_reg_f.feature_importances_


#######

dataset_reg = dataset[dataset['count18'] > 0]

X = dataset_reg[['total_count', 'gnd_ind', 'mean', 'bill6', 'count6', 'bill15', 'count15',
             'bill16', 'count16', 'bill17', 'count17', 'cat_int']]
#X['mean_count'] = (X['count15'] + X['count16'] + X['count17']) / 3
#X['mean_bill'] = (X['bill14'] + X['bill15'] + X['bill16'] + X['bill17']) / 4

X = pd.get_dummies(X, columns = ['cat_int'] ).values


y = dataset_reg['bill18']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)




model_reg_s = LGBMRegressor(n_estimators=400, num_leaves=80, learning_rate=0.05)
model_reg_s.fit(X, y)

y_pred = model_reg_s.predict(X_test)
mean_absolute_error(y_test, y_pred)

#feature_importances = model_reg_s.feature_importances_



#sc_X = StandardScaler()
#X = sc_X.fit_transform(X)
#X_test = sc_X.transform(X_test)


#feature_num = X.shape[1]
#
#model_reg_s = Sequential()
#
#model_reg_s.add(Dense(output_dim=30, init='normal', activation='relu', input_dim=feature_num,
#           kernel_regularizer=regularizers.l2(0.010)))
#model_reg_s.add(Dense(output_dim=30, init='normal', activation='relu',
#           kernel_regularizer=regularizers.l2(0.010)))
#model_reg_s.add(Dense(output_dim=30, init='normal', activation='relu',
#           kernel_regularizer=regularizers.l2(0.010)))
#model_reg_s.add(Dense(output_dim=1, init='normal'))
#
#model_reg_s.compile(optimizer='adam', loss='mean_squared_error')
#model_reg_s.fit(X, y, batch_size=32, nb_epoch=100)




####### Start Predict


dataset.sort_values(['card_no', 'cat_int'], inplace=True)
dataset.reset_index(drop=True, inplace=True)


X = dataset[['total_count', 'brth_estb_yr', 'gnd_ind', 'mean', 'bill7', 'count7', 'bill16', 'count16',
             'bill17', 'count17', 'bill18', 'count18', 'cat_int']]

#X['mean_count'] = (X['count18'] + X['count16'] + X['count17']) / 3
#X['mean_bill'] = (X['bill18'] + X['bill16'] + X['bill17']) / 3

X = pd.get_dummies(X, columns = ['cat_int'] ).values


#y_buy = model_class.predict(X)
y_buy = model_class.predict_proba(X)[:,1] > 0.1

#####
dataset_buy = dataset[y_buy]

X = dataset_buy[['total_count', 'gnd_ind', 'mean', 'bill7', 'count7', 'bill16', 'count16',
             'bill17', 'count17', 'bill18', 'count18', 'cat_int']]
#X['mean_count'] = (X['count18'] + X['count16'] + X['count17']) / 3
#X['mean_bill'] = (X['bill18'] + X['bill16'] + X['bill17']) / 3

X = pd.get_dummies(X, columns = ['cat_int'] ).values


y_pred_f = model_reg_f.predict(X).round()

#####

#X = sc_X.transform(X)

y_pred_s = model_reg_s.predict(X).round()

##### End prediction

dataset.loc[y_buy, 'count_ans'] = y_pred_f
dataset.loc[y_buy, 'spend_ans'] = y_pred_s

dataset.fillna(0, inplace=True)

ans = dataset[['count_ans', 'spend_ans']]

ans_df = pd.DataFrame({'auto_f': ans['count_ans'][::5].reset_index(drop=True), 'cloth_f': ans['count_ans'][1::5].reset_index(drop=True), 
                        'serv_f': ans['count_ans'][2::5].reset_index(drop=True), 'tran_f': ans['count_ans'][3::5].reset_index(drop=True), 
                        'util_f': ans['count_ans'][4::5].reset_index(drop=True), 'auto_s': ans['spend_ans'][::5].reset_index(drop=True), 
                        'cloth_s': ans['spend_ans'][1::5].reset_index(drop=True), 'serv_s': ans['spend_ans'][2::5].reset_index(drop=True), 
                        'tran_s': ans['spend_ans'][3::5].reset_index(drop=True), 'util_s': ans['spend_ans'][4::5].reset_index(drop=True), })

ans_df = ans_df[['auto_f', 'cloth_f', 'serv_f', 'tran_f', 'util_f',
                   'auto_s', 'cloth_s', 'serv_s', 'tran_s', 'util_s']]

ans_df.to_csv('Team_19.csv', header=False, index=False, sep=',')






    
