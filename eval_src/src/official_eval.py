"""Verbatim port of the official LaMP-QA evaluator, for cross-checking.

This mirrors `evaluation/evaluator.py` + `evaluate_responses.py` from
github.com/LaMP-Benchmark/LaMP-QA as closely as possible:

  * offline `vllm.LLM` (not an OpenAI server), `max_model_len=32000`
  * `SamplingParams(temperature, top_p=0.95, max_tokens=4096, logprobs=1)`
  * batch-level retry: every prompt whose output fails to parse is re-run, with
    the temperature raised by 0.1 each round, up to `max_retries`, after which
    it is assigned `match_score = 0`
  * scores evaluated per category file, exactly as the official CLI is invoked

`src/judge.py` is the fast concurrent client used during training; this script
exists to verify the two agree.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json5  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

_EVAL_PROMPT_SYSTEM = """You are a fair and insightful judge with exceptional reasoning and analytical abilities. Your task is to evaluate a user's question, a generated response to that question, and an aspect that is important to the user. Based on this information, identify if the aspect is addressed in the generated response. Provide a clear and accurate assessment.

# your input:
    - question: the question asked by the user.
    - details: the detailed explanation of the question from the user.
    - response: a generated response to the user's question
    - aspect: the aspect that is important to the user, consisting of the following fields:
        - aspect: the title for the aspect.
        - reason: the reason that this aspect is important for the user.
        - evidence: the evidence from the user detailed explanation that the aspect extracted from.

# your output: Your output should be only a valid json object in ```json ``` block without any explanations that contains the following fields:
    - match_score: A score between 0 to 2 that indicates how well the generated response addresses this aspect, where: 0 means the response does not cover this aspect, 1 means the response somewhat covers this aspect, and 2 means the response covers this aspect very well.
"""

_EVAL_PROMPT_USER = """
question: {question}
details: {details}
response: {response}
aspect: {aspects}

Your output should be only a valid json object in ```json ``` block without any explanations.
"""


def parse_json(json_str):
    json_str = json_str.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(json_str, strict=False)
    except Exception:
        pass
    return json5.loads(json_str)


def create_eval_prompt(question, details, response, aspect, tokenizer):
    aspect = (
        f'-aspect: {aspect["aspect"]}\n'
        f'    -reason: {aspect["reason"]}\n'
        f'    -evidence: {aspect["evidence"]}'
    )
    conversation = [
        {"role": "system", "content": _EVAL_PROMPT_SYSTEM},
        {
            "role": "user",
            "content": _EVAL_PROMPT_USER.format(
                question=question, details=details, response=response, aspects=aspect
            ),
        },
    ]
    return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)


def evaluator(queries, responses, details, aspects, llm, max_retries=100):
    temperature = 0.0
    tokenizer = llm.get_tokenizer()
    retries = 0

    prompts, ids = [], []
    for i, (query, response, detail, aspect) in enumerate(
        zip(queries, responses, details, aspects)
    ):
        for j, asp in enumerate(aspect):
            prompts.append(create_eval_prompt(query, detail, response, asp, tokenizer))
            ids.append({"q_id": i, "a_id": j})

    outputs_dict = {}
    while prompts:
        retries += 1
        sampling_params = SamplingParams(
            temperature=temperature, top_p=0.95, max_tokens=4096, logprobs=1
        )
        outputs = llm.generate(prompts, sampling_params)
        wrongs = []
        for id_, prompt, output in zip(ids, prompts, outputs):
            outputs_dict.setdefault(id_["q_id"], {})
            try:
                obj = parse_json(output.outputs[0].text)
                _ = obj["match_score"]
                outputs_dict[id_["q_id"]][id_["a_id"]] = obj
            except Exception:
                if retries > max_retries:
                    outputs_dict[id_["q_id"]][id_["a_id"]] = {"match_score": 0}
                    continue
                wrongs.append((id_, prompt))
        prompts = [p for _, p in wrongs]
        ids = [i for i, _ in wrongs]
        print(f"  retry round {retries}: {len(prompts)} prompts left", flush=True)
        if temperature < 1.0:
            temperature += 0.1

    scores = []
    for i, aspect in enumerate(aspects):
        score_query = 0
        for j, _asp in enumerate(aspect):
            score_query += outputs_dict[i][j]["match_score"]
        scores.append({"id": i, "score": score_query / (len(aspect) * 2)})
    return {
        "score": sum(s["score"] for s in scores) / len(scores),
        "per_question_scores": scores,
    }


SHORT = {
    "Art_and_Entertainment": "Art",
    "Lifestyle_and_Personal_Development": "Lifestyle",
    "Society_and_Culture": "Society",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--responses", nargs="+", required=True)
    ap.add_argument("--out_dir", default="outputs/official_eval")
    ap.add_argument("--evaluator_llm", default=os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct"))
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=32000)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.dataset, encoding="utf-8") if l.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    llm = LLM(
        args.evaluator_llm,
        max_model_len=args.max_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.90,
    )

    for resp_path in args.responses:
        with open(resp_path, encoding="utf-8") as f:
            outputs = json.load(f)
        tag = os.path.basename(resp_path).replace(".json", "")
        summary = {}
        for cfg in ("Art_and_Entertainment", "Lifestyle_and_Personal_Development",
                    "Society_and_Culture"):
            # the official CLI is run once per {category}_{split}.json file
            subset = [r for r in records if r["config"] == cfg]
            result = evaluator(
                [r["question"] for r in subset],
                [str(outputs[r["id"]][0]["output"]) for r in subset],
                [r["details"] for r in subset],
                [r["aspects"] for r in subset],
                llm,
            )
            summary[SHORT[cfg]] = result["score"]
            print(f"[{tag}] {SHORT[cfg]}: {result['score']:.4f}  (n={len(subset)})", flush=True)
        summary["Avg. (macro)"] = sum(summary[c] for c in ("Art", "Lifestyle", "Society")) / 3
        summary["n"] = len(records)
        with open(os.path.join(args.out_dir, f"{tag}.scores.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": summary}, f, indent=2)
        print(f"[{tag}] MACRO {summary['Avg. (macro)']:.4f}", flush=True)


if __name__ == "__main__":
    main()
