import pandas as pd

def df_stats(df):
    """
    df_stats -- prints stats on dtype, NA and unique values in dataframe

    Input:
    df -- pandas.Dataframe
    """
    info = pd.DataFrame({
        'dtype':df.dtypes,
        '#NA':df.isnull().sum(),
        '%NA':map(int,df.isnull().sum()/len(df)*100),
        '#unique':df.nunique()
    })
    print('Shape:{}\n{}'.format(df.shape,info))
    
    

df_data = {}
def df_changes(df, label='', init=False):
    """
    df_changes -- prints recent changes to the dataframe
    
    Input:
    df -- pandas.Dataframe
    label -- string (default='') can be used to distinguish different dataframes
    init -- bool (default=False) resets the stored information on the dataframe
    """
    global df_data

    # Initialize message to print
    msg = ''

    # Set initial data
    if init:
        df_data[label+'shape'] = (0,0)
        df_data[label+'columns'] = set()
        df_data[label+'dtypes'] = dict()
        msg += 'Now tracking changes to dataframe (label=\'{}\')\n'.format(label)

    
    # Changes in # rows
    r0,c0 = df_data.get(label+'shape',(0,0))
    r1,c1 = df.shape
    if r1 > r0: msg += 'Added {} row{}\n'.format(r1-r0,('s' if r1-r0>1 else ''))
    if r0 > r1: msg += 'Removed {} row{}\n'.format(r0-r1,('s' if r0-r1>1 else ''))
        
    # Changes in columns
    d0 = df_data.get(label+'dtypes',{})
    d1 = dict(df.dtypes)
    c0 = set(d0.keys())
    c1 = set(d1.keys())
    if len(c1-c0)>0: 
        msg += 'Added {} column{}:\n'.format(len(c1-c0),('s' if len(c1-c0)>1 else ''))
        for key in c1-c0:
            msg += '- {} ({})\n'.format(key,d1[key])
    if len(c0-c1)>0: 
        msg += 'Removed {} column{}:\n'.format(len(c0-c1),('s' if len(c0-c1)>1 else ''))
        for key in c0-c1:
            msg += '- {} ({})\n'.format(key,d0[key])

    # Changes in data types
    d0 = df_data.get(label+'dtypes',{})
    d1 = dict(df.dtypes)
    keys = set(d0).intersection(set(d1))
    for key in keys:
        if d0[key] != d1[key]: msg += 'Changed dtype \'{}\': {} -> {}\n'.format(key,d0[key],d1[key])
    
    # Store new values in global dict
    df_data[label+'shape'] = df.shape
    df_data[label+'columns'] = set(df.columns)
    df_data[label+'dtypes'] = dict(df.dtypes)

    # Print the message
    if len(msg)>0: print(msg[0:-1])