"""Immutable prompt rendering and tokenizer contract validation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from llm_bias.core.prompt_input import format_prompt, input_ids
from .spec import BASELINE_ENTITY, LABELS, SCORING_INSTRUCTION, TEMPLATES

@dataclass(frozen=True)
class RenderedPrompt:
 template: str; ticker: str; entity: str; raw_content: str; formatted: str; entity_span: tuple[int,int]; answer_position: int; input_ids: tuple[int,...]

def raw_prompt(template: str, entity: str) -> str:
 if template not in TEMPLATES: raise ValueError(f"unknown template {template!r}")
 if "[ENTITY]" not in TEMPLATES[template]: raise AssertionError("immutable template missing slot")
 return TEMPLATES[template].replace("[ENTITY]", entity) + "\n\n" + SCORING_INSTRUCTION

def _ids(tokenizer: Any, text: str) -> list[int]:
 """Legacy facade for the shared tokenizer contract."""
 return input_ids(tokenizer, text, add_special_tokens=False)

def render_prompt(tokenizer: Any, template: str, *, entity: str, ticker: str="", use_chat_template: bool=True, max_seq_len: int|None=None) -> RenderedPrompt:
 raw=raw_prompt(template,entity)
 formatted=format_prompt(tokenizer,raw,use_chat_template=use_chat_template,enable_thinking=False)
 enc=tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
 ids=list(enc.input_ids if hasattr(enc,"input_ids") else enc["input_ids"])
 offsets=enc.offset_mapping if hasattr(enc,"offset_mapping") else enc["offset_mapping"]
 if ids and isinstance(ids[0],list): ids=ids[0]; offsets=offsets[0]
 if max_seq_len is not None and len(ids)>max_seq_len: raise ValueError(f"formatted prompt exceeds max_seq_len={max_seq_len}")
 prefix=TEMPLATES[template].split("[ENTITY]",1)[0]
 raw_start=formatted.find(raw)
 if raw_start < 0: raise ValueError("raw user content is absent from formatted prompt")
 start_char=raw_start + len(prefix); end_char=start_char+len(entity)
 candidates=[]
 for i,(a,b) in enumerate(offsets):
  if b <= a: continue
  if a < start_char or b > end_char: continue
  candidates.append(i)
 if not candidates or candidates != list(range(candidates[0],candidates[-1]+1)): raise ValueError("entity token span is not contiguous or fully covered")
 if offsets[candidates[0]][0] != start_char or offsets[candidates[-1]][1] != end_char or sum(offsets[i][1]-offsets[i][0] for i in candidates) != len(entity): raise ValueError("entity token offsets do not exactly cover entity characters")
 answer_position=len(ids)-1
 if candidates[-1] >= answer_position: raise ValueError("entity span must precede answer position")
 return RenderedPrompt(template,ticker,entity,raw,formatted,(candidates[0],candidates[-1]+1),answer_position,tuple(int(x) for x in ids))

def _continuation_ids(tokenizer: Any, prompt: str, candidate: str) -> list[int]:
 prefix=_ids(tokenizer,prompt); combined=_ids(tokenizer,prompt+candidate)
 if combined[:len(prefix)]!=prefix: raise ValueError("formatted prompt is not an exact token prefix of prompt+label")
 suffix=combined[len(prefix):]
 if not suffix: raise ValueError("label produced no continuation token")
 return suffix

def validate_token_contract(tokenizer: Any, rendered: list[RenderedPrompt], *, use_chat_template=True) -> dict[str,Any]:
 if not rendered: raise ValueError("no prompts to validate")
 ids_by_prompt=[]; decoded={}
 for prompt in rendered:
  one=[]
  for label in LABELS:
   suffix=_continuation_ids(tokenizer,prompt.formatted,label)
   if len(suffix)!=1: raise ValueError(f"label {label!r} is not one continuation token")
   one.append(int(suffix[0])); decoded[label]=tokenizer.decode([suffix[0]],skip_special_tokens=False,clean_up_tokenization_spaces=False)
  ids_by_prompt.append(one)
 if any(ids != ids_by_prompt[0] for ids in ids_by_prompt[1:]): raise ValueError("score label token IDs differ across formatted prompts")
 return {"label_token_ids":dict(zip(LABELS,ids_by_prompt[0],strict=True)),"decoded":decoded,"n_prompts":len(rendered),"anomalies":[]}
