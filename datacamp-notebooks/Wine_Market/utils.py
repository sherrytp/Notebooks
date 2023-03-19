import numpy as np
import pandas as pd
import IPython
from tqdm import tqdm
import matplotlib.pyplot as plt

def print_tab_info(df):
    tab_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0: 'null values (nb)'}))
    # tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.rename(index={0:'null values (%)'}))  %pct of NA
    tab_info.to_excel('courier tab_info.xlsx')
    return df.columns, df.isnull().sum()


def combine_date_time(df, datecol, timecol):
    """
    Combine date and time columns in a pandas dataframe
    """
    return df.apply(lambda row: row[datecol].replace(
        hour=row[timecol].hour,
        minute=row[timecol].minute),
                    axis=1)


def day_of_week(df):
    """
    This class adds the feature day of the week to the original dataset
    :param df: raw dataframe of shipping start date
    returns: dataframe with a new column calculating days -> int
    """
    df['day_of_week'] = df.index.dayofweek
    return df


def days_from_nearest_holiday(i, start_year=2018):
    """
    Create the days from holiday feature using function that gives the number of days from a date to the nearest holiday
    :param df: raw dataframe of shipping start date
    :param start_year: start year of the dataframe
    """
    holiday = []
    years = [x for x in range(start_year, 2021, 1)]
    for date in holidays.UnitedStates(years=years).items():
        holiday.append(date[0])
    x = [(abs(i.date() - h)).days for h in holiday]
    return min(x)


def seconds_calculator(df):
    """not really helpful anymore with updated database"""
    leg1_hour = []
    leg1_min = []
    leg1_second = []
    leg1_time = []
    for i in df['leg1_transit_time_in_hours_mins_seconds'].apply(lambda x: x.split(':')):
        leg1_hour.append(i[0])
        leg1_min.append(i[1])
        leg1_second.append(i[2])
    for x in range(df.shape[0]):
        leg1_time = leg1_hour[x] * 3600 + leg1_min[x] * 60 + leg1_second[x]
    df['leg1_second'] = leg1_time

    leg2_hour = []
    leg2_min = []
    leg2_second = []
    leg2_time = []
    for i in df['leg2_transit_time_in_hours_mins_seconds'].apply(lambda x: x.split(':')):
        leg2_hour.append(i[0])
        leg2_min.append(i[1])
        leg2_second.append(i[2])
    for x in range(df.shape[0]):
        leg2_time = leg2_hour[x] * 3600 + leg2_min[x] * 60 + leg2_second[x]
    df['leg2_second'] = leg2_time
    return df


def geopoint_distance(df):
    """
        This function combines addresses of city, state, country for origins and destinations,
        then calculates the geographical features of given address.
        :param: city and state to upper string
        :param: country_code change to full country names based on a given dictionary
        :param: read input from pickle of all available origin and destination addresses
        returns: geopoint of latitude and longitude of origin and destination, spherical distance between two
        """
    df.origin_state = df['origin_state'].apply(lambda x: str(x))
    df.destination_state = df['destination_state'].apply(lambda x: str(x))
    df.origin_city = df['origin_city'].apply(lambda x: str(x).upper())
    df.destination_city = df['destination_city'].apply(lambda x: str(x).upper())
    df['origin_country_code'].replace(country_codes, inplace=True)
    df['destination_country_code'].replace(country_codes, inplace=True)
    df['origin'] = df['origin_city'] + ', ' + df['origin_state'].fillna('') + ', ' + df['origin_country_code']
    df['destination'] = \
        df['destination_city'] + ', ' + df['destination_state'].fillna('') + ', ' + df['destination_country_code']

    # 1 - convenient function to delay between geocoding calls 'application' or 'myGeocoder'
    locator = Nominatim(user_agent='application', timeout=10)
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
    # 2- - create location only by geocode, create longitude, latitude, altitude from location (returns tuple)
    # dom['origin_point'] = dom['origin'].progress_apply(geocode).progress_apply(lambda loc: tuple(loc.point) if loc else None)
    # dom['destination_point'] = dom['destination'].apply(geocode).apply(lambda loc: tuple(loc.point) if loc else None)
    # 3 - split point column into latitude, longitude and altitude columns
    # df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df['point'].tolist(), index=df.index)
    f = lambda loc: tuple(loc.point) if loc else None

    uniq = pd.concat([df['origin'], df['destination']], ignore_index=True)
    unique_input, unique_input_indices = np.unique(uniq, return_inverse=True)
    unique_locs = np.array([f(geocode(i)) for i in tqdm(unique_input)])
    locs = unique_locs[unique_input_indices]
    end = len(df)
    df['point_origin'] = locs[:end]
    df['point_destination'] = locs[end:]

    dist = []
    for x, y in zip(df['point_origin'], df['point_destination']):
        dist.append(distance(x, y).miles)
    df['distance'] = dist
    return df


def plot_distance(data, flag=False):
    """Histogram of Distance Travelled, figsize(6,6)"""
    plt.style.use('fivethirtyeight')
    plt.hist(data.dropna(), bins=500, edgecolor='k')
    if flag == 'True':
        plt.yscale('log')
    else:
        plt.yscale('linear')
    plt.xlabel('Spherical Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distance')
    plt.show()


def weather_delay():
    """
    Import weather data and complete feature encoding corresponding to dates.
    Resource: https://transtats.bts.gov/
    Time Range: 2018-08 to 2020-06, updated semi-annually
    """
    pass
    airline = pd.read_csv('702328417_T_ONTIME_REPORTING.csv')
    table = airline.groupby('FL_DATE')['WEATHER_DELAY'].mean()
    start_date = pd.to_datetime('2018-08-01')  # or min of airline_df
    end_date = pd.to_datetime('2020-06-30')  # max of airline_df
    if all([df["date"].isin(pd.date_range(start_date, end_date))]):
        df.join(table).fillna(0)
    base_model.fit(train_features, train_labels)
    base_accuracy = evaluate(base_model, test_features, test_labels)
    best_random = rf_random.best_estimator_


def opencage_geocode(df):
    geocoder = OpenCageGeocode('8a2831d3b1c54f7c9fdce385f384200c')
    # for index, row in dom.iterrows(): iterate over rows in dataframe   
    #     query = str(City)+','+str(State)
    # Can also split by ['lat']['lng']
    df['point_origin'] = geocoder.geocode(df['origin'])[0]['geometry']
    df['point_destination'] = geocoder.geocode(df['destination'])[0]['geometry']


country_codes = {
    "AE": 'United Arab Emirates',
    "AM": 'Armenia',
    "AR": 'Argentina',
    "AT": 'Austria',
    "AU": 'Australia',
    "BB": 'Barbados',
    "BD": 'Bangladesh',
    "BE": 'Belgium',
    "BG": 'Bulgaria',
    "BM": 'Bermuda',
    "BN": 'Brunei Darussalam',
    "BQ": 'Bonaire, Sint Eustatius and Saba',
    "BR": 'Brazil',
    "CA": 'Canada',
    "CD": 'Congo',
    "CH": 'Switzerland',
    "CL": 'Chile',
    "CM": 'Cameroon',
    "CN": 'China',
    "CO": 'Colombia',
    "CR": 'Costa Rica',
    "CW": 'Curaçao',
    "CY": 'Cyprus',
    "CZ": 'Czechia',
    "DE": 'Germany',
    "DK": 'Denmark',
    "DO": 'Dominican Republic',
    "EC": 'Ecuador',
    "EE": 'Estonia',
    "EG": 'Egypt',
    "ES": 'Spain',
    "FI": 'Finland',
    "FR": 'France',
    "GB": 'United Kingdom of Great Britain and Northern Ireland',
    "GD": 'Grenada',
    "GE": 'Georgia',
    "GH": 'Ghana',
    "GR": 'Greece',
    "GT": 'Guatemala',
    "HK": 'Hong Kong',
    "HR": 'Croatia',
    "HT": 'Haiti',
    "HU": 'Hungary',
    "ID": 'Indonesia',
    "IE": 'Ireland',
    "IL": 'Israel',
    "IN": 'India',
    "IS": 'Iceland',
    "IT": 'Italy',
    "JM": 'Jamaica',
    "JO": 'Jordan',
    "JP": 'Japan',
    "KE": 'Kenya',
    "KH": 'Cambodia',
    "KR": 'Korea',
    "KW": 'Kuwait',
    "KY": 'Cayman Islands',
    "KZ": 'Kazakhstan',
    "LB": 'Lebanon',
    "LK": 'Sri Lanka',
    "LT": 'Lithuania',
    "LU": 'Luxembourg',
    "LV": 'Latvia',
    "MD": 'Moldova',
    "MO": 'Macao',
    "MP": 'Northern Mariana Islands',
    "MT": 'Malta',
    "MU": 'Mauritius',
    "MX": 'Mexico',
    "MY": 'Malaysia',
    "NG": 'Nigeria',
    "NL": 'Netherlands',
    "NO": 'Norway',
    "NP": 'Nepal',
    "NZ": 'New Zealand',
    "OM": 'Oman',
    "PA": 'Panama',
    "PE": 'Peru',
    "PH": 'Philippines',
    "PK": 'Pakistan',
    "PL": 'Poland',
    "PR": 'Puerto Rico',
    "PT": 'Portugal',
    "QA": 'Qatar',
    "RO": 'Romania',
    "RU": 'Russian Federation',
    "SA": 'Saudi Arabia',
    "SE": 'Sweden',
    "SG": 'Singapore',
    "SI": 'Slovenia',
    "SK": 'Slovakia',
    "SL": 'Sierra Leone',
    "SV": 'El Salvador',
    "TH": 'Thailand',
    "TR": 'Turkey',
    "TT": 'Trinidad and Tobago',
    "TW": 'Taiwan, Province of China',
    "UA": 'Ukraine',
    "UG": 'Uganda',
    "US": 'United States of America',
    "UY": 'Uruguay',
    "VE": 'Venezuela',
    "VI": 'Virgin Islands (U.S.)',
    "VN": 'Vietnam',
    "ZA": 'South Africa',
    "ZW": 'Zimbabwe'
}


def AAD(exp, pred):
    # 100/N*sum(abs(exp-calc)) /exp
    ls = []
    for i, x in zip(pred, exp):
        if x == 0:
            ls.append(0)
        else:
            ls.append(np.abs(x - i) / i)
    return 100 / len(exp) * np.sum(ls)


def evaluate_models(x, y):
    """
    Split datasets into XY train-test sets with a random state of 0.3.
    Evaluate models by comparing AAD Errors with input parameters.
    Exhaustive search over specified parameter values for an estimator.
    :param: important members are fit, predict.
    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    score = make_scorer(AAD, greater_is_better=False)
    # Import parameters and model of xgboosting
    xgb = XGBRegressor(objective='reg:squarederror', gamma=1, random_state=0)
    xgb_param_grid = {
        'max_depth': [i for i in range(2, 52, 2)],
        'n_estimators': [i for i in range(6, 21, 1)]}
    grid_search_xgb = GridSearchCV(estimator=xgb,
                                   param_grid=xgb_param_grid, scoring=score, cv=3,
                                   n_jobs=-1, verbose=2, return_train_score=True)
    grid_search_xgb.fit(X_train, y_train)
    # Import parameters and model of lightgm boosting
    lgb = LGBMRegressor(objective='regression', num_leaves=10, max_depth=0)
    lgb_param_grid = {
        'random_state': [i for i in range(0, 20, 5)],
        # 'max_depth': [6, 10, 20, 70, 80],
        'max_depth': [i for i in range(2, 12, 2)],
        'n_estimators': [i for i in range(15, 41, 1)],
        'num_leaves': [i for i in range(25, 60, 5)]}
    grid_search_lgb = GridSearchCV(estimator=lgb,
                                   param_grid=lgb_param_grid, scoring=score, cv=3,
                                   n_jobs=-1, verbose=2, return_train_score=True)
    grid_search_lgb.fit(X_train, y_train)
    # Import parameters and model of catboosting 
    cat = CatBoostRegressor(learning_rate=0.085164, iterations=2, depth=2)
    cat_param_grid = {'iterations' : [i for i in range(2, 50, 2)]} 
    grid_search_cat = GridSearchCV(estimator=cat,
                               param_grid=cat_param_grid, scoring=score, cv=3,
                               n_jobs=-1, verbose=2, return_train_score=True)
    grid_search_cat.fit(X_train, y_train)
    gridpred_xgb = grid_search_xgb.predict(X_test)
    gridpred_lgb = grid_search_lgb.predict(X_test)
    gridpred_cat = grid_search_cat.predict(X_test) 
    xgb_aad = AAD(y_test, gridpred_xgb)
    lgb_aad = AAD(y_test, gridpred_lgb)
    cat_aad = AAD(y_test, gridpred_cat)
    print('Model Performance Accuracy Error: {:0.4f} degrees of XGB. \n'.format(grid_search_xgb.best_score_))
    print('Model Performance Accuracy Error: {:0.4f} degrees og LGBM. \n'.format(grid_search_lgb.best_score_))
    print('Model Parameters of XGB for best AAD is: ', xgb_aad, grid_search_xgb.best_params_)
    print('Model Parameters of LGBM for best AAD is: ', lgb_aad, grid_search_lgb.best_params_)
    print('Model Parameters of CatBoost for best AAD is: ', cat_aad, grid_search_cat.best_params_)
    X_train.to_pickle('./pickle/X_train.pkl')
    X_test.to_pickle('./pickle/X_test.pkl')
    y_train.to_pickle('./pickle/y_train.pkl')
    y_test.to_pickle('./pickle/y_test.pkl')
    if (lgb_aad < xgb_aad) & (lgb_aad < cat_aad): 
        return gridpred_lgb, y_test
    else:
        return gridpred_xgb, y_test


def stack_models(model_list):
    X_train = pd.read_pickle('./pickle/xgb19.1375/X_train.pkl')
    X_test = pd.read_pickle('./pickle/xgb19.1375/X_test.pkl')
    y_train = pd.read_pickle('./pickle/xgb19.1375/y_train.pkl')
    y_test = pd.read_pickle('./pickle/xgb19.1375/y_test.pkl')
    regressor_list = []
    for i in model_list:
        regressor_list.append(make_pipeline(i))
    stack_gen = StackingCVRegressor(regressors=regressor_list, meta_regressor=make_pipeline(xg),
                                    use_features_in_secondary=True)
    stack_gen.fit(X_train, y_train)
    stack_pred = stack_gen.predict(X_test)
    print('Model Performance Accuracy Error: '.format(AAD(y_test, stack_gen)))
    return stack_pred
# Stacking Example?
#     param_grid = {'max_depth': [6, 10]}
#     xg = xgb.XGBRegressor(objective='reg:squarederror', gamma=1, n_estimators=20)
#     lgb = lgb.LGBMRegressor(objective='regression', num_leaves=10, max_depth=0)
#     model_list = [xg, lgb]
#     stack_models(model_list=model_list)
#    #     pickle.dump(grid, open('./pickle/GridSearchCV.sav'), 'wb')
    #     load_model = pickle.load(open(filename, 'w'))
