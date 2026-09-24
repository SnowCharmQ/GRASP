"""Generate responses against a vLLM-served policy under the official LaMP-QA
RAG protocol.

Only the ``official_rag`` mode is implemented: the policy is prompted with the
official system+user turn and must return a ``personalized_answer`` inside a
```json``` block. Output is the LaMP-QA response format: ``{qid: [{"output": ...}]}``.

Requests are fired concurrently against the vLLM OpenAI server so the serving
GPUs stay saturated. Responses whose JSON fails to parse are re-generated at a
raised temperature, matching the official baselines' retry behaviour.
"""

import argparse
import asyncio
import json
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_prompts import (  # noqa: E402
    OFFICIAL_RAG_SYSTEM,
    build_official_messages,
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(record, num_contexts):
    return build_official_messages(
        OFFICIAL_RAG_SYSTEM, record["question"], record.get("profile", []),
        num_contexts)


def _extra_body(args):
    """vllm-only sampling knobs, omitted entirely when left at defaults."""
    eb = {}
    if getattr(args, "top_k", 0):
        eb["top_k"] = args.top_k
    if getattr(args, "min_p", 0.0):
        eb["min_p"] = args.min_p
    if getattr(args, "presence_penalty", 0.0):
        eb["presence_penalty"] = args.presence_penalty
    if getattr(args, "repetition_penalty", 0.0):
        eb["repetition_penalty"] = args.repetition_penalty
    if getattr(args, "no_thinking", False):
        # Qwen3: switch off the template's reasoning branch. The reasoning
        # tokens spend the max_tokens budget and truncate the answer. No-op for
        # templates that ignore the kwarg.
        eb["chat_template_kwargs"] = {"enable_thinking": False}
    return eb


async def _one(client, sem, model, messages, temperature, top_p, max_tokens,
               retries=5, extra_body=None):
    async with sem:
        for attempt in range(retries):
            try:
                kwargs = dict(model=model, messages=messages,
                              temperature=temperature, top_p=top_p)
                if max_tokens and max_tokens > 0:
                    kwargs["max_tokens"] = max_tokens
                if extra_body:
                    kwargs["extra_body"] = dict(extra_body)
                out = await client.chat.completions.create(**kwargs)
                return out.choices[0].message.content or ""
            except Exception as exc:
                if attempt == retries - 1:
                    print(f"generation failed: {type(exc).__name__}: {exc}",
                          flush=True)
                    return ""
                await asyncio.sleep(2 * (attempt + 1))
    return ""


def merge_system_into_user(messages):
    """Gemma's chat template rejects a system role, so the official LaMP-QA
    formatter folds the system prompt into the user turn instead."""
    if len(messages) >= 2 and messages[0]["role"] == "system":
        merged = messages[0]["content"] + "\n\n" + messages[1]["content"]
        return [{"role": "user", "content": merged}] + list(messages[2:])
    return messages


async def _track(i, coro):
    return i, await coro


async def generate_all(args, records):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(args.concurrency)
    extra_body = _extra_body(args)

    prompts = []
    for rec in records:
        messages = build_prompt(rec, args.num_contexts)
        if args.merge_system:
            messages = merge_system_into_user(messages)
        prompts.append(messages)

    async def run_batch(idxs, temperature):
        tasks = [
            _one(client, sem, args.model, prompts[i], temperature, args.top_p,
                 args.max_tokens, extra_body=extra_body)
            for i in idxs
        ]
        done = 0
        got = {}
        for coro in asyncio.as_completed([_track(k, t) for k, t in enumerate(tasks)]):
            k, text = await coro
            got[idxs[k]] = text
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)
        return got

    outputs = [None] * len(prompts)
    pending = list(range(len(prompts)))
    temperature = args.temperature
    # Re-generate any response whose JSON fails to parse, raising the temperature
    # each round, and only blank it after max_retries. Without this, malformed
    # outputs fall back to the raw JSON blob -- longer than the answer it wraps,
    # so it scores higher.
    retries = 0
    while pending:
        got = await run_batch(pending, temperature)
        for i, text in got.items():
            outputs[i] = text
        still = [i for i in pending if _extract_output(outputs[i]) is None]
        if not still:
            break
        retries += 1
        if retries > args.max_retries:
            for i in still:
                outputs[i] = ""  # official behaviour: give up, score as empty
            print(f"  gave up on {len(still)} unparseable responses after "
                  f"{args.max_retries} retries", flush=True)
            break
        print(f"  retry round {retries}: {len(still)} responses failed to parse",
              flush=True)
        pending = still
        if temperature < 1.0:
            temperature = min(1.0, temperature + 0.1)
    return outputs


def str_to_json(input_str):
    """Port of the official ``utils/json_utils.py:str_to_json``.

    The critical detail is ``.replace("\\n", "")``. Models routinely emit raw
    newlines inside the JSON string value, which is invalid JSON; stripping them
    before parsing is what keeps the official pipeline's failure rate low.
    """
    import json5

    if input_str.startswith("json"):
        input_str = input_str[len("json"):]
    if input_str.endswith("json"):
        input_str = input_str[: -len("json")]
    input_str = (
        input_str.strip().replace("\n", "").replace("```json", "").replace("```", "")
    )
    try:
        return json.loads(input_str, strict=False)
    except Exception:
        pass
    return json5.loads(input_str)


def _fenced_json_blocks(text):
    """Candidate JSON payloads inside a chatty response, best guess first.

    ``str_to_json`` deletes the ``` fences and parses whatever is left, so it
    only works when the JSON object is the *entire* response. Models routinely
    open with a paragraph of prose before the fence, and the official parser
    throws all of them away. Pulling the fenced block out first recovers them.

    Strictly additive: it only runs after ``str_to_json`` has already failed, so
    any response the official parser accepts is parsed identically.
    """
    out = []
    out += re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        out.append(text[first: last + 1])
    return out


def _parse_personalized_answer(text):
    """Pull ``personalized_answer`` out of the official RAG JSON block."""
    text = text or ""
    for candidate in [text] + _fenced_json_blocks(text):
        try:
            obj = str_to_json(candidate)
        except Exception:
            continue
        if isinstance(obj, dict) and "personalized_answer" in obj:
            return str(obj["personalized_answer"])
    return None


_THINK_RE = re.compile(r"<think\s*>.*?</think\s*>", re.S | re.I)


def _strip_reasoning(text):
    """Remove ``<think>...</think>`` blocks emitted by reasoning models.

    An unterminated block (the model hit the token limit mid-thought) leaves
    nothing scoreable, so everything from the dangling open tag onward is
    dropped too.
    """
    if not text:
        return text
    out = _THINK_RE.sub("", text)
    lone = re.search(r"<think\s*>", out, re.I)
    if lone:
        out = out[:lone.start()]
    tail = re.search(r"</think\s*>", out, re.I)
    if tail:
        out = out[tail.end():]
    return out.strip()


def _extract_output(text):
    """The scoreable answer, or None when the JSON contract broke."""
    return _parse_personalized_answer(_strip_reasoning(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="official_rag", choices=("official_rag",),
                    help="only the official RAG protocol is supported")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model",
                    default=os.environ.get("POLICY_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--base_url", default="http://127.0.0.1:8200/v1")
    ap.add_argument("--num_contexts", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0, help="0 disables")
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--presence_penalty", type=float, default=0.0)
    ap.add_argument("--repetition_penalty", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--merge_system", action="store_true",
                    help="fold the system prompt into the user turn (needed for Gemma)")
    ap.add_argument("--max_retries", type=int, default=20,
                    help="re-generate unparseable JSON this many times "
                         "(temperature +0.1 each round) before blanking it")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_thinking", action="store_true",
                    help="switch off the chat template reasoning branch "
                         "(Qwen3); no-op where the template ignores it")
    args = ap.parse_args()

    records = load_jsonl(args.dataset)
    if args.limit:
        records = records[: args.limit]

    print(f"[{args.mode}] generating for {len(records)} records", flush=True)
    texts = asyncio.run(generate_all(args, records))

    responses = {}
    empty = 0
    for rec, text in zip(records, texts):
        answer = _extract_output(text)
        if not answer:
            empty += 1
            answer = ""
        responses[rec["id"]] = [{"output": answer}]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False)
    print(f"  wrote {len(responses)} responses -> {args.output} "
          f"(empty/unparseable: {empty})", flush=True)


if __name__ == "__main__":
    main()
