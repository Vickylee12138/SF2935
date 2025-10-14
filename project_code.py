# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import gc

from sklearn.model_selection import train_test_split, KFold

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# %%
class Config:
    # data_dir = '../input/optiver-realized-volatility-prediction/'
    data_dir = 'F:/SF2935/project/'
    seed = 42

# %%
train = pd.read_csv(Config.data_dir + 'train.csv')
train.head()

# %%
train.stock_id.unique()

# %%
test = pd.read_csv(Config.data_dir + 'test.csv')
test.head()

# %%
display(train.groupby('stock_id').size())

print("\nUnique size values")
display(train.groupby('stock_id').size().unique())

# %%
def get_trade_and_book_by_stock_and_time_id(stock_id, time_id=None, dataType = 'train'):
    book_example = pd.read_parquet(f'{Config.data_dir}book_{dataType}.parquet/stock_id={stock_id}')
    trade_example =  pd.read_parquet(f'{Config.data_dir}trade_{dataType}.parquet/stock_id={stock_id}')
    if time_id:
        book_example = book_example[book_example['time_id']==time_id]
        trade_example = trade_example[trade_example['time_id']==time_id]
    book_example.loc[:,'stock_id'] = stock_id
    trade_example.loc[:,'stock_id'] = stock_id
    return book_example, trade_example

# %% [markdown]
# #### Feature engineering

# %% [markdown]
# 

# %%
def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))


def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

def calculate_wap1(df):
    a1 = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    b1 = df['bid_size1'] + df['ask_size1']
    a2 = df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    b2 = df['bid_size2'] + df['ask_size2']
    
    x = (a1/b1 + a2/b2)/ 2
    
    return x


def calculate_wap2(df):
        
    a1 = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    a2 = df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    b = df['bid_size1'] + df['ask_size1'] + df['bid_size2']+ df['ask_size2']
    
    x = (a1 + a2)/ b
    return x

def realized_volatility_per_time_id(file_path, prediction_column_name):

    stock_id = file_path.split('=')[1]

    df_book = pd.read_parquet(file_path)
    df_book['wap1'] = calculate_wap1(df_book)
    df_book['wap2'] = calculate_wap2(df_book)

    df_book['log_return1'] = df_book.groupby(['time_id'])['wap1'].apply(log_return)
    df_book['log_return2'] = df_book.groupby(['time_id'])['wap2'].apply(log_return)
    df_book = df_book[~df_book['log_return1'].isnull()]

    df_rvps =  pd.DataFrame(df_book.groupby(['time_id'])[['log_return1', 'log_return2']].agg(realized_volatility)).reset_index()
    df_rvps[prediction_column_name] = 0.6 * df_rvps['log_return1'] + 0.4 * df_rvps['log_return2']

    df_rvps['row_id'] = df_rvps['time_id'].apply(lambda x:f'{stock_id}-{x}')
    
    return df_rvps[['row_id',prediction_column_name]]

# %%
def get_agg_info(df):
    agg_df = df.groupby(['stock_id', 'time_id']).agg(mean_sec_in_bucket = ('seconds_in_bucket', 'mean'), 
                                                     mean_price = ('price', 'mean'),
                                                     mean_size = ('size', 'mean'),
                                                     mean_order = ('order_count', 'mean'),
                                                     max_sec_in_bucket = ('seconds_in_bucket', 'max'), 
                                                     max_price = ('price', 'max'),
                                                     max_size = ('size', 'max'),
                                                     max_order = ('order_count', 'max'),
                                                     min_sec_in_bucket = ('seconds_in_bucket', 'min'), 
                                                     min_price = ('price', 'min'),
                                                     #min_size = ('size', 'min'),
                                                     #min_order = ('order_count', 'min'),
                                                     median_sec_in_bucket = ('seconds_in_bucket', 'median'), 
                                                     median_price = ('price', 'median'),
                                                     median_size = ('size', 'median'),
                                                     median_order = ('order_count', 'median')
                                                    ).reset_index()
    
    return agg_df

# %% [markdown]
# #### Most of the feature engineering code

# %%
def get_stock_stat(stock_id : int, dataType = 'train'):
    
    book_subset, trade_subset = get_trade_and_book_by_stock_and_time_id(stock_id, dataType=dataType)
    book_subset.sort_values(by=['time_id', 'seconds_in_bucket'])

    ## book data processing
    
    book_subset['bas'] = (book_subset[['ask_price1', 'ask_price2']].min(axis = 1)
                                / book_subset[['bid_price1', 'bid_price2']].max(axis = 1)
                                - 1)                               

    
    book_subset['wap1'] = calculate_wap1(book_subset)
    book_subset['wap2'] = calculate_wap2(book_subset)
    
    book_subset['log_return_bid_price1'] = np.log(book_subset['bid_price1'].pct_change() + 1)
    book_subset['log_return_ask_price1'] = np.log(book_subset['ask_price1'].pct_change() + 1)
    # book_subset['log_return_bid_price2'] = np.log(book_subset['bid_price2'].pct_change() + 1)
    # book_subset['log_return_ask_price2'] = np.log(book_subset['ask_price2'].pct_change() + 1)
    book_subset['log_return_bid_size1'] = np.log(book_subset['bid_size1'].pct_change() + 1)
    book_subset['log_return_ask_size1'] = np.log(book_subset['ask_size1'].pct_change() + 1)
    # book_subset['log_return_bid_size2'] = np.log(book_subset['bid_size2'].pct_change() + 1)
    # book_subset['log_return_ask_size2'] = np.log(book_subset['ask_size2'].pct_change() + 1)
    book_subset['log_ask_1_div_bid_1'] = np.log(book_subset['ask_price1'] / book_subset['bid_price1'])
    book_subset['log_ask_1_div_bid_1_size'] = np.log(book_subset['ask_size1'] / book_subset['bid_size1'])
    

    book_subset['log_return1'] = (book_subset.groupby(by = ['time_id'])['wap1'].
                                  apply(log_return).
                                  reset_index(drop = True).
                                  fillna(0)
                                 )
    book_subset['log_return2'] = (book_subset.groupby(by = ['time_id'])['wap2'].
                                  apply(log_return).
                                  reset_index(drop = True).
                                  fillna(0)
                                 )
    
    stock_stat = pd.merge(
        book_subset.groupby(by = ['time_id'])['log_return1'].agg(realized_volatility).reset_index(),
        book_subset.groupby(by = ['time_id'], as_index = False)['bas'].mean(),
        on = ['time_id'],
        how = 'left'
    )
    
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return2'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return_bid_price1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return_ask_price1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return_bid_size1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return_ask_size1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_ask_1_div_bid_1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_ask_1_div_bid_1_size'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    
    
    stock_stat['stock_id'] = stock_id
    
    # Additional features that can be added. Referenced from https://www.kaggle.com/yus002/realized-volatility-prediction-lgbm-train/data
    
    # trade_subset_agg = get_agg_info(trade_subset)
    
    #     stock_stat = pd.merge(
    #         stock_stat,
    #         trade_subset_agg,
    #         on = ['stock_id', 'time_id'],
    #         how = 'left'
    #     )
    
    ## trade data processing 
    
    return stock_stat

def get_data_set(stock_ids : list, dataType = 'train'):

    stock_stat = Parallel(n_jobs=-1)(
        delayed(get_stock_stat)(stock_id, dataType) 
        for stock_id in stock_ids
    )
    
    stock_stat_df = pd.concat(stock_stat, ignore_index = True)

    return stock_stat_df

# %%
def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

# %%
book_stock_1, trade_stock_1 = get_trade_and_book_by_stock_and_time_id(1, 5)
display(book_stock_1.shape)
display(trade_stock_1.shape)

# %%
book_stock_1.head()

# %%
trade_stock_1.head()

# %%
%%time
train_stock_stat_df = get_data_set(train.stock_id.unique(), dataType = 'train')
train_stock_stat_df.head()

# %%
train_data_set = pd.merge(train, train_stock_stat_df, on = ['stock_id', 'time_id'], how = 'left')
train_data_set.head()

# %%
train_data_set.info()

# %%
%%time
test_stock_stat_df = get_data_set(test['stock_id'].unique(), dataType = 'test')
test_stock_stat_df

# %%
test_data_set = pd.merge(test, test_stock_stat_df, on = ['stock_id', 'time_id'], how = 'left')
test_data_set.fillna(-999, inplace=True)
test_data_set

# %%
# train_data_set.to_pickle('train_features_df.pickle')
# test_data_set.to_pickle('test_features_df.pickle')

# %%
x = gc.collect()

# %%
X_display = train_data_set.drop(['stock_id', 'time_id', 'target'], axis = 1)
X = X_display.values
y = train_data_set['target'].values

X.shape, y.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=Config.seed, shuffle=False)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# # Optuna Tuned XGBoost

# %%
rs = Config.seed

# %%
import optuna
from optuna.samplers import TPESampler

def objective(trial, data=X, target=y):
    
    def rmspe(y_true, y_pred):
        return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=rs, shuffle=False)
    
    param = {
        'tree_method':'gpu_hist', 
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.009,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15,17,20]),
        'random_state': trial.suggest_categorical('random_state', [24, 48,2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)}
    
    model = XGBRegressor(**param)
    
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")
    model.fit(X_train ,y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    
    preds = model.predict(X_test)
    
    rmspe = rmspe(y_test, preds)
    
    return rmspe

# %%
study = optuna.create_study(sampler=TPESampler(), direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=1000, gc_after_trial=True)

# %%
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# %%
optuna.visualization.plot_optimization_history(study)

# %%
optuna.visualization.plot_param_importances(study)

# %%
best_xgbparams = study.best_params
best_xgbparams

# %%
xgb = XGBRegressor(**best_xgbparams, tree_method='gpu_hist')

# %%
%%time
xgb.fit(X_train ,y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)

preds = xgb.predict(X_test)
R2 = round(r2_score(y_true = y_test, y_pred = preds), 5)
RMSPE = round(rmspe(y_true = y_test, y_pred = preds), 5)
print(f'Performance of the Tuned XGB prediction: R2 score: {R2}, RMSPE: {RMSPE}')

# %% [markdown]
# # Optuna Tuned LGBM

# %%
def objective(trial):
    
    def rmspe(y_true, y_pred):
        return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs, shuffle=False)
    valid = [(X_test, y_test)]
    
    param = {
        "device": "gpu",
        "metric": "rmse",
        "verbosity": -1,
        'learning_rate':trial.suggest_loguniform('learning_rate', 0.005, 0.5),
        "max_depth": trial.suggest_int("max_depth", 2, 500),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "n_estimators": trial.suggest_int("n_estimators", 100, 4000),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100000, 700000),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)}

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")
    model = LGBMRegressor(**param)
    
    model.fit(X_train, y_train, eval_set=valid, verbose=False, callbacks=[pruning_callback], early_stopping_rounds=100)

    preds = model.predict(X_test)
    
    rmspe = rmspe(y_test, preds)
    return rmspe

# %%
study = optuna.create_study(sampler=TPESampler(), direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=1000, gc_after_trial=True)

# %%
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# %%
optuna.visualization.plot_optimization_history(study)

# %%
optuna.visualization.plot_param_importances(study)

# %%
best_lgbmparams = study.best_params
best_lgbmparams

# %%
lgbm = LGBMRegressor(**best_lgbmparams, device='gpu')

# %%
%%time
lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False, early_stopping_rounds=100)

preds = xgb.predict(X_test)
R2 = round(r2_score(y_true = y_test, y_pred = preds), 6)
RMSPE = round(rmspe(y_true = y_test, y_pred = preds), 6)
print(f'Performance of the Tuned LIGHTGBM prediction: R2 score: {R2}, RMSPE: {RMSPE}')

# %% [markdown]
# # Stacking Regressor

# %%
def_xgb = XGBRegressor(tree_method='gpu_hist', random_state = rs, n_jobs= - 1)

def_lgbm = LGBMRegressor(device='gpu', random_state=rs)

# %%
from sklearn.ensemble import StackingRegressor


estimators = [('def_xgb', def_xgb),
              ('def_lgbm', def_lgbm),
              ('tuned_xgb', xgb)]

clf = StackingRegressor(estimators=estimators, final_estimator=lgbm, verbose=1)

# %%
%%time
clf.fit(X_train, y_train)

# %%
preds = clf.predict(X_test)
R2 = round(r2_score(y_true = y_test, y_pred = preds),6)
RMSPE = round(rmspe(y_true = y_test, y_pred = preds), 6)
print(f'Performance of the STACK prediction: R2 score: {R2}, RMSPE: {RMSPE}')

# %%
from sklearn.linear_model import LinearRegression
Lin = LinearRegression()
Lin.fit(X_train, y_train)
preds = Lin.predict(X_test)
R2 = round(r2_score(y_true = y_test, y_pred = preds),6)
RMSPE = round(rmspe(y_true = y_test, y_pred = preds), 6)
print(f'Performance of the Linear prediction: R2 score: {R2}, RMSPE: {RMSPE}')

# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# dt = DecisionTreeRegressor(random_state=42)
dt = DecisionTreeRegressor(
    max_depth=4,      
    random_state=42
)
dt.fit(X_train, y_train)
preds = dt.predict(X_test)
R2 = round(r2_score(y_true=y_test, y_pred=preds), 6)
RMSPE = round(rmspe(y_true=y_test, y_pred=preds), 6)
print(f'Performance of the Decision Tree prediction: R2 score: {R2}, RMSPE: {RMSPE}')

# %%
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

bagging_lin = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=30, 
    random_state=42,
    n_jobs=-1,
)

bagging_lin.fit(X_train, y_train)
preds = bagging_lin.predict(X_test)
R2 = round(r2_score(y_true=y_test, y_pred=preds), 6)
RMSPE = round(rmspe(y_true=y_test, y_pred=preds), 6)
print(f'Performance of Bagging Linear Regression: R2 score: {R2}, RMSPE: {RMSPE}')

# %%
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_jobs=-1, n_estimators=30)
RF.fit(X_train, y_train)
preds = RF.predict(X_test)
R2 = round(r2_score(y_true = y_test, y_pred = preds),6)
RMSPE = round(rmspe(y_true = y_test, y_pred = preds), 6)
print(f'Performance of the Random Forest prediction: R2 score: {R2}, RMSPE: {RMSPE}')

# %%
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

models = {
    # 'AdaBoost': AdaBoostRegressor(
    #     estimator=DecisionTreeRegressor(max_depth=4, random_state=42),
    #     n_estimators=100,
    #     learning_rate=0.1,
    #     random_state=42
    # ),
    'Linear': LinearRegression(
    ),
    'Bagging': BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=4, random_state=42),
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        random_state=42
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        # max_features='sqrt', 
        random_state=42
    )
}


results = []

for name, model in models.items():
    print(f"Training {name}...")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
   
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmspe_val = rmspe(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    overfitting_gap = train_r2 - r2
    

    results.append({
        'Model': name,
        'R2': round(r2, 4),
        'RMSE': round(rmse, 4),
        'RMSPE': round(rmspe_val, 4),
        'Training_Time': round(training_time, 2),
        'CV_Mean_R2': round(cv_mean, 4),
        'CV_Std': round(cv_std, 4),
        'Overfitting_Gap': round(overfitting_gap, 4)
    })
    
    print(f"{name} completed in {training_time:.2f} seconds")

results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)
print(results_df.to_string(index=False))

print("\n" + "="*50)
print("PERFORMANCE RANKINGS")
print("="*50)

r2_rank = results_df[['Model', 'R2']].sort_values('R2', ascending=False)
print("R2 Score Ranking:")
for i, (_, row) in enumerate(r2_rank.iterrows(), 1):
    print(f"{i}. {row['Model']}: {row['R2']}")

rmspe_rank = results_df[['Model', 'RMSPE']].sort_values('RMSPE')
print("\nRMSPE Ranking (lower is better):")
for i, (_, row) in enumerate(rmspe_rank.iterrows(), 1):
    print(f"{i}. {row['Model']}: {row['RMSPE']}")


stability_rank = results_df[['Model', 'CV_Std']].sort_values('CV_Std')
print("\nStability Ranking (CV Std, lower is better):")
for i, (_, row) in enumerate(stability_rank.iterrows(), 1):
    print(f"{i}. {row['Model']}: {row['CV_Std']}")


overfit_rank = results_df[['Model', 'Overfitting_Gap']].sort_values('Overfitting_Gap')
print("\nOverfitting Resistance Ranking (gap, lower is better):")
for i, (_, row) in enumerate(overfit_rank.iterrows(), 1):
    print(f"{i}. {row['Model']}: {row['Overfitting_Gap']}")

print("\n" + "="*50)
print("COMPREHENSIVE ANALYSIS")
print("="*50)

best_r2_model = results_df.loc[results_df['R2'].idxmax()]
best_rmspe_model = results_df.loc[results_df['RMSPE'].idxmin()]
fastest_model = results_df.loc[results_df['Training_Time'].idxmin()]
most_stable_model = results_df.loc[results_df['CV_Std'].idxmin()]
least_overfitting_model = results_df.loc[results_df['Overfitting_Gap'].idxmin()]

print(f"Best R2: {best_r2_model['Model']} ({best_r2_model['R2']})")
print(f"Best RMSPE: {best_rmspe_model['Model']} ({best_rmspe_model['RMSPE']})")
print(f"Fastest Training: {fastest_model['Model']} ({fastest_model['Training_Time']}s)")
print(f"Most Stable: {most_stable_model['Model']} (CV Std: {most_stable_model['CV_Std']})")
print(f"Least Overfitting: {least_overfitting_model['Model']} (Gap: {least_overfitting_model['Overfitting_Gap']})")

print("\n" + "="*50)
print("VALIDATION OF TABLE CHARACTERISTICS")
print("="*50)

print("Noise Sensitivity (based on CV stability):")
for _, row in stability_rank.iterrows():
    sensitivity = "Low" if row['CV_Std'] < 0.05 else "Moderate" if row['CV_Std'] < 0.1 else "High"
    print(f"- {row['Model']}: {sensitivity}")

print("\nOverfitting Risk:")
for _, row in overfit_rank.iterrows():
    risk = "Low" if row['Overfitting_Gap'] < 0.05 else "Moderate" if row['Overfitting_Gap'] < 0.1 else "High"
    print(f"- {row['Model']}: {risk}")

# %%
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Fixed evaluation function
def evaluate_model_performance(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Evaluate model performance including training time, R2 scores and overfitting gap
    """
    print(f"\n=== {model_name} Performance Evaluation ===\n")
    
    # Ensure correct data format
    X_train = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    X_test = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
    y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
    y_test = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
    
    print(f"Training set shape: X_train{X_train.shape}, y_train{y_train.shape}")
    print(f"Test set shape: X_test{X_test.shape}, y_test{y_test.shape}")
    
    # Training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training Time: {training_time:.2f} seconds")
    
    # Training and test predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(f"Prediction shapes: y_train_pred{y_train_pred.shape}, y_test_pred{y_test_pred.shape}")
    
    # R2 scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Training R2: {train_r2:.5f}")
    print(f"Test R2: {test_r2:.5f}")
    
    # Overfitting gap
    overfitting_gap = train_r2 - test_r2
    print(f"Overfitting Gap (Train R2 - Test R2): {overfitting_gap:.5f}")
    
    # RMSPE
    train_rmspe = rmspe(y_train, y_train_pred)
    test_rmspe = rmspe(y_test, y_test_pred)
    print(f"Training RMSPE: {train_rmspe:.5f}")
    print(f"Test RMSPE: {test_rmspe:.5f}")
    
    return {
        'training_time': training_time,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'overfitting_gap': overfitting_gap,
        'train_rmspe': train_rmspe,
        'test_rmspe': test_rmspe
    }

# Cross-validation evaluation function
def evaluate_cross_validation(model, X, y, cv_folds=5, model_name="Model"):
    """
    Evaluate model performance using cross-validation
    """
    print(f"\n=== {model_name} Cross-Validation Evaluation ===\n")
    
    # Ensure correct data format
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    y = np.array(y) if not isinstance(y, np.ndarray) else y
    
    print(f"Data shape: X{X.shape}, y{y.shape}")
    
    start_time = time.time()
    
    # Cross-validation with R2 scoring
    try:
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2', n_jobs=-1)
    except Exception as e:
        print(f"Cross-validation error: {e}")
        print("Using single thread for cross-validation...")
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2', n_jobs=1)
    
    cv_time = time.time() - start_time
    
    print(f"Cross-validation time: {cv_time:.2f} seconds")
    print(f"Cross-validation R2 scores:")
    print(f"  Fold scores: {[f'{score:.5f}' for score in cv_scores]}")
    print(f"  Mean R2: {cv_scores.mean():.5f} (+/- {cv_scores.std() * 2:.5f})")
    print(f"  Standard deviation: {cv_scores.std():.5f}")
    
    return {
        'cv_mean_r2': cv_scores.mean(),
        'cv_std_r2': cv_scores.std(),
        'cv_scores': cv_scores,
        'cv_time': cv_time
    }

# Prepare data for training (ensure correct format)
def prepare_data_for_training(train_data_set):
    """
    Prepare training data ensuring correct format
    """
    # Select feature columns
    feature_columns = ['log_return1', 'bas', 'log_return2', 'log_return_bid_price1', 
                      'log_return_ask_price1', 'log_return_bid_size1', 'log_return_ask_size1',
                      'log_ask_1_div_bid_1', 'log_ask_1_div_bid_1_size']
    
    X = train_data_set[feature_columns].values
    y = train_data_set['target'].values
    
    print(f"Data preparation completed: X{X.shape}, y{y.shape}")
    print(f"Feature columns: {feature_columns}")
    
    return X, y, feature_columns

# Prepare data
print("Preparing data...")
X, y, feature_columns = prepare_data_for_training(train_data_set)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=Config.seed, shuffle=True  # Changed to shuffle=True
)

print(f"Final data shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Baseline XGBoost model with default parameters
print("\n" + "="*50)
print("Evaluating Baseline XGBoost Model Performance...")
print("="*50)

base_xgb = XGBRegressor(
    random_state=Config.seed,
    tree_method='gpu_hist',
    n_estimators=500,  # Reduced number of trees for faster training
    learning_rate=0.1,
    max_depth=6,
    verbosity=0
)

# Evaluate baseline model
base_performance = evaluate_model_performance(
    base_xgb, X_train, X_test, y_train, y_test, "Baseline XGBoost"
)

# Cross-validation for baseline model (using training set)
base_cv = evaluate_cross_validation(
    XGBRegressor(
        random_state=Config.seed,
        tree_method='gpu_hist',
        n_estimators=300,  # Use fewer trees for cross-validation
        learning_rate=0.1,
        max_depth=6,
        verbosity=0
    ), 
    X_train, y_train, cv_folds=3, model_name="Baseline XGBoost"  # Reduced folds for speed
)

# LightGBM model evaluation
print("\n" + "="*50)
print("Evaluating LightGBM Model Performance...")
print("="*50)

lgb_model = LGBMRegressor(
    random_state=Config.seed,
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    device='gpu',
    verbose=-1
)

# Evaluate LightGBM model
lgb_performance = evaluate_model_performance(
    lgb_model, X_train, X_test, y_train, y_test, "LightGBM"
)

# LightGBM cross-validation
lgb_cv = evaluate_cross_validation(
    LGBMRegressor(
        random_state=Config.seed,
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        device='gpu',
        verbose=-1
    ),
    X_train, y_train, cv_folds=3, model_name="LightGBM"
)

# If tuned parameters exist, evaluate tuned model
try:
    if 'best_xgbparams' in locals() or 'best_xgbparams' in globals():
        print("\n" + "="*50)
        print("Evaluating Tuned XGBoost Model Performance...")
        print("="*50)
        
        # Use previously found best parameters
        tuned_params = best_xgbparams.copy()
        if 'n_estimators' in tuned_params and tuned_params['n_estimators'] > 1000:
            tuned_params['n_estimators'] = 800  # Limit number of trees
        
        tuned_xgb = XGBRegressor(**tuned_params, tree_method='gpu_hist', verbosity=0)
        
        # Evaluate tuned model
        tuned_performance = evaluate_model_performance(
            tuned_xgb, X_train, X_test, y_train, y_test, "Tuned XGBoost"
        )
        
        # Cross-validation for tuned model
        tuned_cv_params = tuned_params.copy()
        if 'n_estimators' in tuned_cv_params:
            tuned_cv_params['n_estimators'] = min(400, tuned_cv_params['n_estimators'])
            
        tuned_cv = evaluate_cross_validation(
            XGBRegressor(**tuned_cv_params, tree_method='gpu_hist', verbosity=0),
            X_train, y_train, cv_folds=3, model_name="Tuned XGBoost"
        )
except Exception as e:
    print(f"Tuned model evaluation failed: {e}")

# Create performance summary table
performance_summary = []

# Add baseline XGBoost results
performance_summary.append({
    'Model': 'Baseline XGBoost',
    'Training Time (s)': base_performance['training_time'],
    'Train R2': base_performance['train_r2'],
    'Test R2': base_performance['test_r2'],
    'Overfitting Gap': base_performance['overfitting_gap'],
    'CV Mean R2': base_cv['cv_mean_r2'],
    'CV Std R2': base_cv['cv_std_r2'],
    'Test RMSPE': base_performance['test_rmspe']
})

# Add LightGBM results
performance_summary.append({
    'Model': 'LightGBM',
    'Training Time (s)': lgb_performance['training_time'],
    'Train R2': lgb_performance['train_r2'],
    'Test R2': lgb_performance['test_r2'],
    'Overfitting Gap': lgb_performance['overfitting_gap'],
    'CV Mean R2': lgb_cv['cv_mean_r2'],
    'CV Std R2': lgb_cv['cv_std_r2'],
    'Test RMSPE': lgb_performance['test_rmspe']
})

# Add tuned model results if available
if 'tuned_performance' in locals():
    performance_summary.append({
        'Model': 'Tuned XGBoost',
        'Training Time (s)': tuned_performance['training_time'],
        'Train R2': tuned_performance['train_r2'],
        'Test R2': tuned_performance['test_r2'],
        'Overfitting Gap': tuned_performance['overfitting_gap'],
        'CV Mean R2': tuned_cv['cv_mean_r2'],
        'CV Std R2': tuned_cv['cv_std_r2'],
        'Test RMSPE': tuned_performance['test_rmspe']
    })

# Display performance summary
summary_df = pd.DataFrame(performance_summary)
print("\n" + "="*60)
print("Model Performance Summary")
print("="*60)
print(summary_df.round(5))

# Visualization comparison
try:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models = summary_df['Model'].values
    test_r2 = summary_df['Test R2'].values
    cv_mean_r2 = summary_df['CV Mean R2'].values
    
    x = range(len(models))
    width = 0.35
    
    # R2 score comparison
    axes[0, 0].bar(x, test_r2, width, label='Test R2', alpha=0.7, color='skyblue')
    axes[0, 0].bar([i + width for i in x], cv_mean_r2, width, label='CV Mean R2', alpha=0.7, color='lightcoral')
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('R2 Score')
    axes[0, 0].set_title('R2 Score Comparison')
    axes[0, 0].set_xticks([i + width/2 for i in x])
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Overfitting gap
    axes[0, 1].bar(models, summary_df['Overfitting Gap'], alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('Overfitting Gap')
    axes[0, 1].set_title('Overfitting Gap (Train R2 - Test R2)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training time
    axes[1, 0].bar(models, summary_df['Training Time (s)'], alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # RMSPE comparison
    axes[1, 1].bar(models, summary_df['Test RMSPE'], alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('RMSPE')
    axes[1, 1].set_title('Test RMSPE Comparison')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Visualization creation failed: {e}")

print("\n" + "="*60)
print("Evaluation Completed!")
print("="*60)

# Output key conclusions
best_model_idx = summary_df['Test R2'].idxmax()
best_model = summary_df.loc[best_model_idx, 'Model']
best_test_r2 = summary_df.loc[best_model_idx, 'Test R2']
best_rmspe = summary_df.loc[best_model_idx, 'Test RMSPE']

print(f"\nKey Conclusions:")
print(f"- Best Model: {best_model} (Test R2: {best_test_r2:.5f})")
print(f"- Best Model RMSPE: {best_rmspe:.5f}")
print(f"- Least Overfitting Model: {summary_df.loc[summary_df['Overfitting Gap'].idxmin(), 'Model']}")
print(f"- Fastest Training Model: {summary_df.loc[summary_df['Training Time (s)'].idxmin(), 'Model']}")

# Additional detailed analysis
print(f"\nDetailed Analysis:")
print(f"- Cross-validation consistency:")
for model in summary_df['Model']:
    model_data = summary_df[summary_df['Model'] == model].iloc[0]
    cv_std = model_data['CV Std R2']
    consistency = "High" if cv_std < 0.05 else "Medium" if cv_std < 0.1 else "Low"
    print(f"  {model}: CV Std = {cv_std:.5f} ({consistency} consistency)")

print(f"- Overfitting analysis:")
for model in summary_df['Model']:
    model_data = summary_df[summary_df['Model'] == model].iloc[0]
    gap = model_data['Overfitting Gap']
    severity = "Low" if gap < 0.05 else "Medium" if gap < 0.1 else "High"
    print(f"  {model}: Overfitting gap = {gap:.5f} ({severity} overfitting)")


