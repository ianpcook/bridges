import os
import pandas as pd

base_path = "/Users/ian/Documents/exploratory/bridges/data/external/BaseData"
paths = []
def path_collector(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".xlsm"):
                 #print(os.path.join(root, file))
                 s = os.path.join(root, file)
                 print(s)
                 paths.append(s)
    return paths


def xls_collector(part):
    full_df = pd.DataFrame()
    paths = path_collector(base_path)
    file_list = [path for path in paths if part in path.split('/')[-1].split(' ')[0]]
    i = 0
    if part == 'Culvert':
        part_ref = 'CULV'
    else:
        part_ref = part
    for file in file_list:
        i += 1
        try:
            temp_df = pd.DataFrame()
            temp_df = pd.read_excel(file,
                                    sheet_name=part_ref,
                                    skiprows=0,
                                    low_memory=True,
                                    header=1,
                                    index_col=0)
            temp_df = temp_df.dropna(thresh=100, axis=0)
            #  thresh set to arbitrary high number to deal with excel read-in;
            #  using "all" does not drop the rows b/c it thinks there's non-NaN content in them
        except FileNotFoundError:
            temp_df = pd.DataFrame()
        full_df = full_df.append(temp_df,
                                 sort=False)
        print(str(i) + ' of ' + str(len(file_list)) + ' files processed.')
        print('df length is now: '+str(len(full_df)))
    full_df.dropna(how='all', axis=1)
    full_df = full_df.loc[:, ~full_df.columns.str.startswith('Blank')]  # kill blank columns
    full_df = full_df.loc[:, ~full_df.columns.str.startswith('Unnamed')]  # kill unnamed columns
    return full_df


def get_df(base_path, part):
    if os.path.exists(base_path+'/'+part+'_full.csv'):
        df = pd.read_csv(base_path+'/'+part+'_full.csv')
    else:
        df = xls_collector(part=part)
    df = df.loc[:, ~df.columns.str.startswith('Blank')]  # kill blank columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]  # kill unnamed columns
    return df

#
# deck_df = xls_collector(part)
# deck_df.to_csv(base_path+'deck_full.csv')
# deck_df = pd.read_csv(base_path+'deck_full.csv')
