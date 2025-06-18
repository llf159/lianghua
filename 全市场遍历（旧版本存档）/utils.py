def ensure_datetime_index(df):
    df = df.sort_values('date')
    df = df.set_index('date')
    return df
