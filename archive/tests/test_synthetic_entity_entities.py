from llm_bias.synthetic_entity_bias.entities import load_entity_pool

def test_pool_normalizes_and_aggregates(tmp_path):
 p=tmp_path/'sp500_constituents_2020_2025.csv'; p.write_text('index_name,year,ticker,company_name,gics_sector\nS&P 500,2020,a.b,Old,Tech\nS&P 500,2021,A-B,New,Tech\n',encoding='utf-8')
 rows=load_entity_pool([p],seed=1)
 assert len(rows)==1 and rows[0].ticker=='A-B' and rows[0].company_name=='New'
 assert rows[0].years==(2020,2021)
