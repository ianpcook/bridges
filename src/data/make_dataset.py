# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # read in the XLS data
    df = pd.read_excel('/Users/ian/Documents/exploratory/bridges/data/external/test data.xlsx')
    df['id'] = df.index
    # select just the condition columns so we can melt the df from wide to long
    melt_vars = ['Condition State', 'Inspection Year']
    # make a base dataframe for all years; will spread out time for bridge existence over full span of inspections
    year_df = pd.DataFrame([*range(1985, 2019, 1)], columns={'year'})
    ts_df = pd.DataFrame(df['id'])
    for var in melt_vars:
        var_cols = [col for col in df.columns if var in col]
        keep_cols = var_cols + ['id']
        var_df = pd.melt(df[keep_cols], id_vars=['id'], var_name='event_counter', value_name=var.split(' ')[0]+'_value')
        var_df['event_counter'] = var_df['event_counter'].str.extract(r'(\d+)', expand=False).astype(int)
        var_df = var_df.sort_values(by=['id', 'event_counter'])
        if 'event_counter' in ts_df.columns:
            ts_df = pd.merge(left=ts_df, right=var_df, on=['id', 'event_counter'], how='inner')
        else:
            ts_df = pd.merge(left=ts_df, right=var_df, on=['id'], how='inner')
    # create duration counts and binary degradation indicators
    ts_df = ts_df[ts_df['Condition_value'] != 0]
    dur_df = pd.DataFrame()
    for id in ts_df['id'].unique():
        sub_df = ts_df[ts_df['id']==id]
        sub_df['inspection_gap'] = (sub_df['Inspection_value'].shift(1) - sub_df['Inspection_value'])*-1
        sub_df['condition_change'] = (sub_df['Condition_value'].shift(1) - sub_df['Condition_value'])
        dur_df = dur_df.append(sub_df)
        print('Finished bridge: ', id)
    # get rid of the edge cases
    dur_df['condition_change'].fillna(0, inplace=True)  # assume the condition didn't change in either direction
    dur_df['inspection_gap'].fillna(2, inplace=True)  # assume two years since the last inspection
    dur_df['condition_change'][dur_df['condition_change'] < 0] = 0
    dur_df = dur_df.groupby(by=['id', 'Condition_value']).sum().reset_index()
    dur_df.rename(columns={'condition_change': 'degraded',
                           'inspection_gap': 'duration'}, inplace=True)  # more appropriate column
    # shift back to indicate which states we've seen degrade
    dur_df['degraded_obs'] = dur_df['degraded'].shift(1).fillna(0)
    dur_df.drop(['Condition_value', 'event_counter', 'Inspection_value', 'degraded'], axis=1, inplace=True)
    # merge the time-set df with the full df
    feature_cols = ['ADT (2018)', 'BPN', 'Deck Area',
                    'FC', 'LENGTH', 'Material', 'Span Length',
                    'Truck Precentage', 'YEAR BUILT', 'id']
    feature_df = df[feature_cols]
    full_df = pd.merge(left=dur_df, right=feature_df, on='id', how='inner')
    full_df = pd.concat([full_df, pd.get_dummies(full_df['Material'])], axis=1)  # dummy out the Material
    full_df = pd.concat([full_df, pd.get_dummies(full_df['FC'])], axis=1)  # dummy out the FC column
    full_df.drop(['Material', 'FC'], axis=1, inplace=True)
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    full_df.to_csv('/Users/ian/Documents/exploratory/bridges/data/processed/bridges_coxph.csv', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path('__file__').resolve().parents[0]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir/'input_filepath')
