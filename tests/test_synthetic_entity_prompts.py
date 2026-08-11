import pytest
from types import SimpleNamespace
from llm_bias.synthetic_entity_bias.prompts import render_prompt
class Tokenizer:
 chat_template='x'
 def apply_chat_template(self,messages,**kwargs): return messages[0]['content']
 def __call__(self,text,**kwargs): return SimpleNamespace(input_ids=list(range(len(text))),offset_mapping=[(i,i+1) for i in range(len(text))])
def test_prompt_rejects_truncation():
 with pytest.raises(ValueError): render_prompt(Tokenizer(),'positive',entity='ACME',max_seq_len=2)
