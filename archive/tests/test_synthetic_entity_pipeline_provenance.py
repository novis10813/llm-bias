import hashlib
from llm_bias.synthetic_entity_bias.entities import load_entity_pool

def test_entity_pool_flags_overlap_conflict_duplicate_and_normalization(tmp_path):
 p=tmp_path/'sp500_constituents_2020_2025.csv'; p.write_text('index_name,year,ticker,company_name,gics_sector\nS&P 500,2020,a.b,Old,Tech\nS&P 500,2020,A-B,New,Tech\nS&P 500,2020,A-B,New,Tech\nRussell 1000,2020,A-B,Other,Tech\n',encoding='utf-8')
 row=load_entity_pool([p])[0]
 assert {'conflicting_company_name','membership_overlap','duplicate_source_row','ticker_normalization_collision'} <= set(row.anomalies)

def test_hash_is_stable_for_ticker_ids():
 assert hashlib.sha256('A\nB'.encode()).hexdigest()==hashlib.sha256('A\nB'.encode()).hexdigest()
