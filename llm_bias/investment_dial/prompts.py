"""Explicit local evidence contract and Appendix A JSON elicitation."""
import random

from llm_bias.core.prompt_input.encoding import format_prompt, input_ids, continuation_token_ids

VERSION = "investment-dial-local-v1"
PREFIX = '{"decision": "'


def validate_data(data):
    if data.get("schema_version") != 1 or not isinstance(data.get("source"), str) or not data["source"].strip():
        raise ValueError("require schema_version=1 and source provenance")
    companies = data.get("companies", [])
    if not isinstance(companies, list) or not companies:
        raise ValueError("require companies")
    tickers, splits = set(), set()
    for company in companies:
        for key in ("ticker", "name", "split"):
            if not isinstance(company.get(key), str) or not company[key].strip():
                raise ValueError(f"missing company {key}")
        ticker = company["ticker"].strip().upper()
        if ticker in tickers or company["split"] not in {"screen", "A", "B", "test"}:
            raise ValueError("duplicate company or invalid split")
        tickers.add(ticker)
        splits.add(company["split"])
        pairs = company.get("evidence_pairs", [])
        if not isinstance(pairs, list) or len(pairs) < 4:
            raise ValueError("require at least four matched evidence pairs")
        for pair in pairs:
            if not isinstance(pair, dict) or any(not isinstance(pair.get(k), str) or not pair[k].strip() for k in ("positive", "negative")):
                raise ValueError("invalid evidence pair")
    if splits != {"screen", "A", "B", "test"}:
        raise ValueError("require disjoint screen/A/B/test companies")


def build_trials(data, *, seed=42, repeats=2, graded=False):
    validate_data(data)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows = []
    for company in sorted(data["companies"], key=lambda c: c["ticker"]):
        rng = random.Random(f"{seed}:{company['ticker'].strip().upper()}")
        for repeat in range(repeats):
            pairs = rng.sample(company["evidence_pairs"], 4)
            order = list(range(4))
            rng.shuffle(order)
            for positive_count in (range(5) if graded else [2]):
                items = ([pairs[i]["positive"] for i in range(positive_count)] +
                         [pairs[i]["negative"] for i in range(4 - positive_count)])
                evidence = [items[i] for i in order]
                for reverse in (False, True):
                    options = '"sell" or "buy"' if reverse else '"buy" or "sell"'
                    prompt = (
                        "Refer to the evidence below to make a final investment decision for the given stock.\n\n"
                        f"Stock Ticker: [{company['ticker']}]\n\nStock Name: [{company['name']}]\n\n"
                        "— Evidence —\n\n" + "\n".join(f"- {item}" for item in evidence) + "\n\n—\n\n"
                        "Your final response must be a single, valid JSON object. The JSON object must contain\n"
                        "the following two keys:\n\n"
                        f'"decision": {options}\n\n'
                        '"reason": A brief justification for your decision\n\n'
                        'You must choose either "buy" or "sell" – "hold" is NOT an allowed answer. Pick the\n'
                        "direction the evidence leans toward, even if the evidence is mixed. Your response\n"
                        "should start with { and end with }. Do not include any other text."
                    )
                    rows.append({"id": f"{company['ticker']}:{repeat}:{positive_count}:{int(reverse)}",
                                 "ticker": company["ticker"], "split": company["split"],
                                 "positive_count": positive_count, "reverse_options": reverse, "prompt": prompt})
    return rows


def encode_trials(rows, tokenizer, *, max_prompt_tokens=4096):
    encoded = []
    for row in rows:
        text = format_prompt(tokenizer, row["prompt"], use_chat_template=True, enable_thinking=False)
        ids = input_ids(tokenizer, text, add_special_tokens=False)
        decision_text = text + PREFIX
        decision_ids = input_ids(tokenizer, decision_text, add_special_tokens=False)
        answers = [continuation_token_ids(tokenizer, decision_text, word) for word in ("buy", "sell")]
        if any(len(answer) != 1 for answer in answers) or answers[0] == answers[1]:
            raise ValueError("buy/sell must be distinct single-token continuations at fixed JSON prefix")
        if not ids or len(decision_ids) > max_prompt_tokens:
            raise ValueError("empty or oversized prompt; truncation forbidden")
        encoded.append(row | {"prompt_ids": ids, "decision_ids": decision_ids,
                              "answer_ids": [answer[0] for answer in answers]})
    return encoded
