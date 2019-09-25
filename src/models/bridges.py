import folium
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

def build_df(file1_path, file2_path):
    pb = pd.read_csv(file1_path, low_memory=False)
    br = pd.read_csv(file2_path, low_memory=False)
    df = pd.merge(br, pb, how='inner', left_on='Bridge Reference Number', right_on='BRKEY')
    return df


def prep_split_df(df):
    df = df.sort_values(by='Bridge Reference Number')
    df = df.drop_duplicates(subset='Bridge Reference Number')
    df = df[~(df['Structurally Deficient'] == 'DEMO')]
    df = df[~(df['Structurally Deficient'] == '0')]
    df['Administrative Area'] = df['Administrative Area'].astype(int)
    df['Average Daily Traffic'] = df['Average Daily Traffic'].astype(float)
    keep_cols = ['Owner Code', 'Length', 'Deck Area', 'Number of Sections', 'Structure Type',
                 'Year Built', 'Why Bridge Posted of Closed', 'Cond Rate Bridge Deck',
                 'Cond Rate Bridge Superstructure', 'Cond Rate Bridge Substructure',
                 'Cond Rate Bridge Culvert', 'Sufficiency Rating', 'Administrative Area',
                 'Average Daily Traffic', 'OFFSET', 'YEARRECON', 'SERVTYPON', 'MAINSPANS',
                 'APPSPANS', 'DECKWIDTH', 'DKSURFTYPE', 'DKMEMBTYPE', 'DKPROTECT',
                 'DEPT_MAIN_MATERIAL_TYPE', 'DEPT_MAIN_PHYSICAL_TYPE', 'DEPT_MAIN_SPAN_INTERACTION',
                 'DEPT_MAIN_STRUC_CONFIG', 'BYPASSLEN',
                 'AROADWIDTH', 'ROADWIDTH', 'NBI_RATING', 'MAIN_DEF_RATE',
                 'MATERIALMAIN', 'MAXSPAN', 'LANES', 'TRUCKPCT']
    select = [x for x in df.columns if x in keep_cols]
    X = df.loc[:, select]
    # crappy hack to just get this done in time for the Wolf visit
    X.fillna(value=0, inplace=True)
    y = df.loc[:, 'Structurally Deficient']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    le=LabelEncoder()
    # Iterating over all the common columns in train and test
    for col in X_test.columns.values:
        # Encoding only categorical variables
        if X_test[col].dtypes == 'object':
            # Using whole data to form an exhaustive list of levels
            # data = X_train[col].append(X_test[col])
            X_train[col] = le.fit_transform(X_train.loc[:,col].astype(str))
            X_test[col] = le.fit_transform(X_test.loc[:,col].astype(str))
    return df, X_train, X_test, y_train, y_test


def run_model(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(multi_class='multinomial', solver='newton-cg',
                            max_iter=1000, verbose=1, n_jobs=-1,
                            warm_start=True).fit(X_train, y_train)
    pprint('The model\'s score on the training set is: '+str(accuracy_score(y_train, lr.predict(X_train))))
    pprint('The model\'s score on the test set is: '+str(accuracy_score(y_test, lr.predict(X_test))))
    pprint('The confusion matrix on the test set: ')
    y_test_pred = lr.predict(X_test)
    pprint(pd.DataFrame(confusion_matrix(y_test, y_test_pred), index=['t:Fair', 't:Good', 't:Poor', 't:SD'],
                 columns=['p:Fair', 'p:Good', 'p:Poor', 'p:SD']))
    y_train_pred = lr.predict(X_train)
    return y_test_pred, y_train_pred


def make_mapping_df(df, X_train, X_test, y_test_pred, y_train_pred):
    map_df = pd.concat([X_train, X_test], axis=0)
    predictions = pd.DataFrame(np.hstack((y_train_pred, y_test_pred))).rename(columns={0: 'SD_predicted'})
    map_df['SD_predicted'] = predictions['SD_predicted'].values
    map_df = pd.merge(map_df, df['Structurally Deficient'], left_index=True, right_index=True, how='inner')
    map_df.rename(columns={'Structurally Deficient': 'SD_rated'}, inplace=True)
    map_df['equal'] = map_df['SD_rated'] == map_df['SD_predicted']
    map_df.sort_index(inplace=True)
    map_df['lat'] = df['DEC_LAT'].values
    map_df['long'] = df['DEC_LONG'].values
    condition_map = {}
    condition_map['Good'] = 0
    condition_map['Fair'] = 1
    condition_map['Poor'] = 2
    condition_map['SD'] = 3
    map_df['rated_val'] = map_df['SD_rated'].apply(lambda x: condition_map.get(x, ''))
    map_df['pred_val'] = map_df['SD_predicted'].apply(lambda x: condition_map.get(x, ''))
    map_df['rated_pred_diff'] = map_df['rated_val'] - map_df['pred_val']
    return map_df


def make_map(map_df):
    map_df.dropna(inplace=True)
    folium_map = folium.Map(location=[map_df['lat'].mean(), map_df['long'].mean()],
                            zoom_start=8,
                            tiles='openstreetmap')
    one_worse = folium.FeatureGroup(name='Predicted worse than rated')
    # add the locations where the prediction has a worse condition that the rating
    pred_1_worse_df = map_df[map_df['rated_val'] - map_df['pred_val'] == -1]
    for index, row in pred_1_worse_df.iterrows():
        popup_text = "Given Rating: {}<br> Predicted Rating: {}"
        popup_text = popup_text.format(row["SD_rated"],
                                       row["SD_predicted"],
                                       )
        #radius = row['pred_val']
        one_worse.add_child(folium.Marker(location=(row['lat'],
                                row['long']),
                      popup=popup_text,
                      # color='#ff0000',
                      # radius=radius,
                      # fill=True
                      icon=folium.Icon(color='red', icon='wrench', prefix='fa')
                      ))#.add_to(folium_map)
    # add the locations where the prediction has a better condition than the rating
    one_better = folium.FeatureGroup(name='Rated slightly better than predicted')
    pred_1_better_df = map_df[map_df['rated_val'] - map_df['pred_val'] == 1]
    for index, row in pred_1_better_df.iterrows():
        popup_text = "Given Rating: {}<br> Predicted Rating: {}"
        popup_text = popup_text.format(row["SD_rated"],
                                       row["SD_predicted"],
                                       )
        radius = 5
        one_better.add_child(folium.CircleMarker(location=(row['lat'],
                                row['long']),
                      popup=popup_text,
                      color='#888888',
                      radius=radius,
                      fill=True
                      #icon=folium.Icon(color='lightgray', prefix='fa')
                      ))#.add_to(folium_map)
    two_better = folium.FeatureGroup(name='Rated much better than predicted')
    pred_2_better_df = map_df[map_df['rated_val'] - map_df['pred_val'] == 2]
    for index, row in pred_2_better_df.iterrows():
        popup_text = "Given Rating: {}<br> Predicted Rating: {}"
        popup_text = popup_text.format(row["SD_rated"],
                                       row["SD_predicted"],
                                       )
        # radius = row['pred_val']
        two_better.add_child(folium.Marker(location=(row['lat'],
                                row['long']),
                      popup=popup_text,
                      # color='#ff0000',
                      # radius=radius,
                      # fill=True
                      icon=folium.Icon(color='darkgreen', icon='pause', prefix='fa')
                      ))#.add_to(folium_map)
    folium_map.add_child(one_worse)
    folium_map.add_child(one_better)
    folium_map.add_child(two_better)
    folium.TileLayer('Mapbox Bright').add_to(folium_map)
    folium.TileLayer('openstreetmap').add_to(folium_map)
    folium.TileLayer('Mapbox Control Room').add_to(folium_map)
    folium.TileLayer('Stamen Terrain').add_to(folium_map)
    folium_map.add_child(folium.LayerControl())
    folium_map.save('./pa_bridges_map.html')


def main():
    FILE1_PATH = './data/external/Pennsylvania_Bridges.csv'
    FILE2_PATH = './data/external/bridges.csv'
    df = build_df(file1_path=FILE1_PATH, file2_path=FILE2_PATH)
    df, X_train, X_test, y_train, y_test = prep_split_df(df)
    y_test_pred, y_train_pred = run_model(X_train, X_test, y_train, y_test)
    map_df = make_mapping_df(df, X_train, X_test, y_test_pred, y_train_pred)
    make_map(map_df)


if __name__ == '__main__':
    main()
