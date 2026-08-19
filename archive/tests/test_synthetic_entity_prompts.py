import pytest
from types import SimpleNamespace
from llm_bias.synthetic_entity_bias.prompts import render_prompt
class Tokenizer:
 chat_template='x'
 def apply_chat_template(self,messages,**kwargs): return messages[0]['content']
 def __call__(self,text,**kwargs): return SimpleNamespace(input_ids=list(range(len(text))),offset_mapping=[(i,i+1) for i in range(len(text))])
def test_prompt_rejects_truncation():
    with pytest.raises(ValueError): render_prompt(Tokenizer(),'positive',entity='ACME',max_seq_len=2)


def test_all_expanded_templates_render():
    from llm_bias.synthetic_entity_bias.spec import TEMPLATES, TEMPLATE_SENTIMENTS
    assert len(TEMPLATES) == 12
    assert set(TEMPLATES.keys()) == set(TEMPLATE_SENTIMENTS.keys())
    for template_name in TEMPLATES:
        rendered = render_prompt(Tokenizer(), template_name, entity="NVIDIA Corporation")
        assert rendered.entity == "NVIDIA Corporation"
        assert rendered.template == template_name
        assert rendered.entity_span[1] > rendered.entity_span[0]
        assert rendered.answer_position > rendered.entity_span[1]
