import numpy as np
import pandas as pd
import src.data.data_collector as dc

base_path = "/Users/ian/Documents/exploratory/bridges/data/external/BaseData"
paths = []


def make_wide_df(base_path, element):
    df = dc.get_df(base_path, element)
    #  list the columns with time-varying data; these will be shifted to long format
    condition_cols = []
    condition_cols.extend(df.columns[df.columns.str.startswith('Condition State.')])  # get the inspection year and value
    condition_cols = condition_cols + ['BRKEY']
    condition_df = pd.melt(df.loc[:, condition_cols],
                       id_vars='BRKEY',
                       var_name='condition_col',
                       value_name='condition_state')
    condition_df['condition'] = [int(x.split('.')[1]) for x in condition_df['condition']]
    duration_cols = []
    duration_cols.extend(df.columns[df.columns.str.startswith('Condition State Duration')])
    duration_cols = duration_cols + ['BRKEY']
    duration_df = pd.melt(df.loc[:, duration_cols],
                          id_vars='BRKEY',
                          var_name='duration_col',
                          value_name='duration_values')
    try:
        [int(x.split('.')[1]) for x in duration_df['duration_col']]
    event_cols = condition_cols + duration_cols + ['BRKEY']
    #  conduct the melt process independently for both the condition and duration

    time_invar_cols = list(df.iloc[:, 0:18].columns)
    time_invar_df = df[df.columns.intersection(time_invar_cols)]
    wide_df = pd.merge(time_invar_df, event_df, on='BRKEY')
    return [wide_df, event_df, time_invar_df]





    #
    # time_cols.extend(df.columns[df.columns.str.contains('Condition', case=False)])  # get condition in inspection year
    # time_cols.extend(df.columns[df.columns.str.contains('Rehab', case=False)])  # get rehab status