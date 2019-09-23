import numpy as np
import pandas as pd
import src.data.data_collector as dc

base_path = "/Users/ian/Documents/exploratory/bridges/data/external/BaseData"
paths = []


def make_wide_df(base_path, part):
    df = dc.get_df(base_path, part)
    #  list the columns with time-varying data; these will be shifted to long format
    time_cols = []
    time_cols.extend(df.columns[df.columns.str.contains('Inspection', case=False)])  # get the inspection year and value
    time_cols.extend(df.columns[df.columns.str.contains('Condition', case=False)])  # get condition in inspection year
    time_cols.extend(df.columns[df.columns.str.contains('Rehab', case=False)])  # get rehab status
    cols = time_cols + ['BRKEY']
    time_var_df = pd.melt(df.loc[:,cols],
                        id_vars = 'BRKEY',
                        var_name = 'status',
                        value_name = 'values')
    time_invar_df = df.drop(time_cols, axis=1)
    wide_df = pd.merge(time_invar_df, time_var_df, on='BRKEY')
    return [wide_df, time_var_df, time_invar_df]

