'''
script for exploring the available data for the datathon
'''


#%% imports

import numpy as np
import pandas as pd
import os

#%%
# I put Arif's data and the Agent A data in the clinical_data folder and put .csv files in the gitignore in case making the data public is an issue
AgentA_data = pd.read_csv(os.path.join('..', 'clinical_data', 'Amsterdam_AgentA.csv'))
# RL_data = pd.read_csv(os.path.join('..', 'clinical_data', 'data_v1_max_72_h.csv'))
#parquet file
RL_data = pd.read_parquet(os.path.join('..', 'clinical_data', 'data_v3_max_72_h.parquet'))
RL_data.reset_index()
RL_data_old = pd.read_csv(os.path.join('..', 'clinical_data', 'data_v1_max_72_h.csv'))

# %%
# list all columns in AgentA that have data
AgentA_columns = []
for col in AgentA_data.columns:
    if AgentA_data[col].notnull().sum() > 0:
        AgentA_columns.append(col)
print('there are {} columns with data in AgentA_data'.format(len(AgentA_columns)))

RL_data_columns = []
for col in RL_data.columns:
    if RL_data[col].notnull().sum() > 0:
        RL_data_columns.append(col)
print('there are {} columns with data in RL_data'.format(len(RL_data_columns)))

# print the columns that are in one dataset but not the other
only_in_AgentA = set(AgentA_columns) - set(RL_data_columns)
only_in_RL = set(RL_data_columns) - set(AgentA_columns)
print('columns only in AgentA_data:', only_in_AgentA)
print('columns only in RL_data:', only_in_RL)

# %%
#for every column in RL_data_columns, print the column and the dtype on one line
for col in RL_data_columns:
    print(col, ',' , RL_data[col].dtype)
# %%

# find rows where 'Pressure Control' is a value in any column
pressure_control_rows = RL_data.isin(['Pressure Control']).any(axis=1)
print('there are {} rows with Pressure Control as a value'.format(pressure_control_rows.sum()))
# %%
