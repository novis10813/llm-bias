"""
Script to collect, process, and output historical constituent data for S&P 500, Russell 1000, and Russell 2000
from 2020 to 2025 into CSV files under data/
"""

import json
import urllib.request
import io
import os
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

SP500_EXTRA_META = {
    'ABC': ('AmerisourceBergen Corp', 'Health Care'),
    'ANTM': ('Anthem Inc.', 'Health Care'),
    'BK': ('Bank of New York Mellon', 'Financials'),
    'BLL': ('Ball Corporation', 'Materials'),
    'FI': ('Fiserv Inc.', 'Financials'),
    'LB': ('L Brands Inc.', 'Consumer Discretionary'),
    'MMC': ('Marsh & McLennan Companies', 'Financials'),
    'NLOK': ('NortonLifeLock Inc.', 'Information Technology'),
    'PARA': ('Paramount Global', 'Communication Services'),
    'PEAK': ('Healthpeak Properties Inc.', 'Real Estate'),
    'PKI': ('PerkinElmer Inc.', 'Health Care'),
    'VIAC': ('ViacomCBS Inc.', 'Communication Services'),
    'WRK': ('WestRock Company', 'Materials'),
    'DISCA': ('Discovery Inc. Series A', 'Communication Services'),
    'DISCK': ('Discovery Inc. Series C', 'Communication Services'),
    'FB': ('Meta Platforms Inc. (Facebook)', 'Communication Services'),
    'RE': ('Everest Re Group Ltd', 'Financials'),
    'TWTR': ('Twitter Inc.', 'Communication Services'),
    'ATVI': ('Activision Blizzard Inc.', 'Communication Services'),
    'SBNY': ('Signature Bank', 'Financials'),
    'SIVB': ('Silicon Valley Bank (SVB Financial Group)', 'Financials'),
    'FRC': ('First Republic Bank', 'Financials'),
    'AAL': ('American Airlines Group', 'Industrials'),
    'AAP': ('Advance Auto Parts', 'Consumer Discretionary'),
    'HES': ('Hess Corporation', 'Energy'),
    'KSU': ('Kansas City Southern', 'Industrials'),
}


def fetch_sp500_data():
    print("[1/3] Processing S&P 500 historical constituents (2020-2025)...")
    url_changes = "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
    req = urllib.request.Request(url_changes, headers=HEADERS)
    df_hist = pd.read_csv(urllib.request.urlopen(req), keep_default_na=False)
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    
    wiki_url = "https://en.wikipedia.org/w/api.php?action=parse&page=List_of_S%26P_500_companies&format=json&prop=text"
    req_wiki = urllib.request.Request(wiki_url, headers=HEADERS)
    html = json.loads(urllib.request.urlopen(req_wiki).read())['parse']['text']['*']
    dfs_wiki = pd.read_html(io.StringIO(html), keep_default_na=False)
    
    ticker_meta = {}
    
    # Table 0: Current constituents
    for _, row in dfs_wiki[0].iterrows():
        sym = str(row['Symbol']).replace('.', '-').strip()
        if sym:
            ticker_meta[sym] = {
                'company_name': str(row.get('Security', '')).strip(),
                'gics_sector': str(row.get('GICS Sector', '')).strip()
            }
        
    # Table 1: Historical changes
    if len(dfs_wiki) > 1:
        changes_df = dfs_wiki[1]
        for _, r in changes_df.iterrows():
            t_add = str(r[('Added', 'Ticker')]).replace('.', '-').strip()
            s_add = str(r[('Added', 'Security')]).strip()
            if t_add and t_add.upper() != 'NAN' and t_add not in ticker_meta:
                ticker_meta[t_add] = {'company_name': s_add, 'gics_sector': ''}
                
            t_rem = str(r[('Removed', 'Ticker')]).replace('.', '-').strip()
            s_rem = str(r[('Removed', 'Security')]).strip()
            if t_rem and t_rem.upper() != 'NAN' and t_rem not in ticker_meta:
                ticker_meta[t_rem] = {'company_name': s_rem, 'gics_sector': ''}
                
    # GitHub dataset metadata
    url_gh = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    req_gh = urllib.request.Request(url_gh, headers=HEADERS)
    df_gh = pd.read_csv(urllib.request.urlopen(req_gh), keep_default_na=False)
    for _, r in df_gh.iterrows():
        sym = str(r['Symbol']).replace('.', '-').strip()
        sec = str(r.get('Security', '')).strip()
        gics = str(r.get('Sector', '')).strip()
        if sym:
            if sym not in ticker_meta or not ticker_meta[sym]['company_name']:
                ticker_meta[sym] = {'company_name': sec, 'gics_sector': gics}
            elif not ticker_meta[sym]['gics_sector'] and gics:
                ticker_meta[sym]['gics_sector'] = gics
            
    # Add extra historical renames mapping
    for sym, (cname, gsec) in SP500_EXTRA_META.items():
        if sym not in ticker_meta:
            ticker_meta[sym] = {'company_name': cname, 'gics_sector': gsec}
        else:
            if not ticker_meta[sym]['company_name']:
                ticker_meta[sym]['company_name'] = cname
            if not ticker_meta[sym]['gics_sector']:
                ticker_meta[sym]['gics_sector'] = gsec

    records = []
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    
    for y in years:
        df_year = df_hist[df_hist['date'].dt.year == y]
        if df_year.empty:
            df_year = df_hist[df_hist['date'] <= f"{y}-12-31"]
        
        last_row = df_year.iloc[-1]
        tickers = [t.strip() for t in str(last_row['tickers']).split(',') if t.strip()]
        
        for t in tickers:
            clean_ticker = t.replace('.', '-').strip()
            if not clean_ticker:
                continue
            meta = ticker_meta.get(clean_ticker, {})
            c_name = meta.get('company_name', clean_ticker)
            g_sec = meta.get('gics_sector', '')
            records.append({
                'index_name': 'S&P 500',
                'year': y,
                'ticker': clean_ticker,
                'company_name': c_name if c_name else clean_ticker,
                'gics_sector': g_sec if g_sec else 'Unspecified'
            })
            
    df_sp500 = pd.DataFrame(records)
    return df_sp500


def fetch_russell1000_data():
    print("[2/3] Processing Russell 1000 constituents (2020-2025)...")
    wiki_url = "https://en.wikipedia.org/w/api.php?action=parse&page=Russell_1000_Index&format=json&prop=text"
    req_wiki = urllib.request.Request(wiki_url, headers=HEADERS)
    html = json.loads(urllib.request.urlopen(req_wiki).read())['parse']['text']['*']
    dfs_wiki = pd.read_html(io.StringIO(html), keep_default_na=False)
    
    r1k_df = None
    for df in dfs_wiki:
        if df.shape[0] > 900 and 'Symbol' in df.columns:
            r1k_df = df
            break
            
    records = []
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    
    if r1k_df is not None:
        for _, row in r1k_df.iterrows():
            sym = str(row['Symbol']).replace('.', '-').strip()
            if not sym:
                continue
            comp = str(row.get('Company', '')).strip()
            sec = str(row.get('GICS Sector', '')).strip()
            for y in years:
                records.append({
                    'index_name': 'Russell 1000',
                    'year': y,
                    'ticker': sym,
                    'company_name': comp if comp else sym,
                    'gics_sector': sec if sec else 'Unspecified'
                })
    
    df_r1k = pd.DataFrame(records)
    return df_r1k


def fetch_russell2000_data():
    print("[3/3] Processing Russell 2000 constituents (2020-2025)...")
    url_r2k = "https://raw.githubusercontent.com/derekbanas/Python4Finance/main/Russell2000.csv"
    req = urllib.request.Request(url_r2k, headers=HEADERS)
    df_r2k_raw = pd.read_csv(urllib.request.urlopen(req), keep_default_na=False)
    
    records = []
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    
    for _, row in df_r2k_raw.iterrows():
        sym = str(row['Ticker']).strip().replace('.', '-')
        if not sym:
            continue
        comp = str(row.get('Company', '')).strip()
        sec = str(row.get('Sector', '')).strip()
        for y in years:
            records.append({
                'index_name': 'Russell 2000',
                'year': y,
                'ticker': sym,
                'company_name': comp if comp else sym,
                'gics_sector': sec if sec else 'Unspecified'
            })
            
    df_r2k = pd.DataFrame(records)
    return df_r2k


def main():
    df_sp500 = fetch_sp500_data()
    df_r1k = fetch_russell1000_data()
    df_r2k = fetch_russell2000_data()
    
    sp500_path = os.path.join(DATA_DIR, "sp500_constituents_2020_2025.csv")
    r1k_path = os.path.join(DATA_DIR, "russell1000_constituents_2020_2025.csv")
    r2k_path = os.path.join(DATA_DIR, "russell2000_constituents_2020_2025.csv")
    combined_path = os.path.join(DATA_DIR, "all_constituents_2020_2025.csv")
    
    df_sp500.to_csv(sp500_path, index=False)
    print(f"Saved: {sp500_path} ({len(df_sp500)} rows)")
    
    df_r1k.to_csv(r1k_path, index=False)
    print(f"Saved: {r1k_path} ({len(df_r1k)} rows)")
    
    df_r2k.to_csv(r2k_path, index=False)
    print(f"Saved: {r2k_path} ({len(df_r2k)} rows)")
    
    df_combined = pd.concat([df_sp500, df_r1k, df_r2k], ignore_index=True)
    df_combined.to_csv(combined_path, index=False)
    print(f"Saved combined file: {combined_path} ({len(df_combined)} rows)")

if __name__ == "__main__":
    main()
