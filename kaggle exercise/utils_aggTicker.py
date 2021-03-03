import datetime as dt 
import pandas as pd
import plotly.graph_objs as go

def data_loader(path, ticker, date): 
    """
    returns a dataframe containing all transactions of a time range for one specific ticker 
    """
    file_list = []
    df_list = []
    for i in date: 
        file_list.append(path + ticker + '-' + str(i))
    for i in file_list: 
        df = pd.read_csv(i, sep = '\t', index_col = 'tradeid')
        df_list.append(df)
    return pd.concat(df_list)

def volume_agg(df, parse_min: str, ) -> pd.DataFrame: 
    result = df.groupby(pd.Grouper(key='time', freq=parse_min)).agg({'amount': 'sum'})
    result['min'] = result.index.minute
    result['hour'] = result.index.hour 
    result[parse_min] = result['hour']*60 + result['min']
    table = result.groupby(parse_min)['amount'].sum()
    return table 

def volatility_agg(df, parse_min: str, ) -> pd.DataFrame: 
    table = df.groupby(pd.Grouper(key='time', freq=parse_min)).agg({'price': 'std'})
#     or .groupby('min')['price'].std()
    table['min'] = table.index.minute
    table['hour'] = table.index.hour 
    table[parse_min] = table['hour']*60 + table['min']
    return table.groupby(parse_min)['price'].sum() 
