# Tony's Code? 
# Code for extracting weekly returns from WRDS api:

import wrds
import pandas as pd
import datetime

# Connect to WRDS
db = wrds.Connection()

# Function to get S&P 500 and company returns
def get_sp500_and_company_returns():
    # Get the list of S&P 500 companies
    sp500_constituents = db.get_table('comp', 'idxcst_his', columns=['gvkey', 'iid', '"from"', '"thru"'])
    sp500_constituents = sp500_constituents[sp500_constituents['iid'] == '01']

    # Get the S&P 500 index returns
    sp500_returns = db.raw_sql("""
        SELECT caldt AS date, vwretd AS ret
        FROM crsp.dsp500
        WHERE caldt BETWEEN '2000-01-01' AND '2010-12-31'
    """)

    # Filter for Wednesdays
    sp500_returns['date'] = pd.to_datetime(sp500_returns['date'])
    sp500_returns = sp500_returns[sp500_returns['date'].dt.weekday == 2]

    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=['Ticker', 'Date', 'Company Return', 'S&P 500 Return'])

    # Loop through each company in the S&P 500
    for _, row in sp500_constituents.iterrows():
        gvkey = row['gvkey']

        # Get the lpermno for the gvkey
        permno_data = db.raw_sql(f"""
            SELECT lpermno
            FROM crsp.ccmxpf_linktable
            WHERE gvkey = '{gvkey}' AND linktype IN ('LU', 'LC') AND
                  linkdt <= '2010-12-31' AND (linkenddt IS NULL OR linkenddt >= '2000-01-01')
        """)

        if permno_data.empty:
            continue

        lpermno = int(permno_data.iloc[0]['lpermno'])

        # Get the company returns
        company_returns = db.raw_sql(f"""
            SELECT date, ret
            FROM crsp.dsf
            WHERE permno = {lpermno} AND date BETWEEN '2000-01-01' AND '2010-12-31'
        """)

        # Filter for Wednesdays
        company_returns['date'] = pd.to_datetime(company_returns['date'])
        company_returns = company_returns[company_returns['date'].dt.weekday == 2]

        # Merge with S&P 500 returns
        merged = pd.merge(company_returns, sp500_returns, on='date', suffixes=('_company', '_sp500'))
        merged['Ticker'] = gvkey
        merged = merged[['Ticker', 'date', 'ret_company', 'ret_sp500']]
        merged.columns = ['Ticker', 'Date', 'Company Return', 'S&P 500 Return']

        # Append to the main DataFrame
        df = pd.concat([df, merged])

    return df

# Get the DataFrame
df = get_sp500_and_company_returns()
print(df)