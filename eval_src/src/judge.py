"""LaMP-QA rubric judge (Qwen2.5-32B-Instruct) behind a vLLM OpenAI server.

Implements mu(x, y_hat, E) exactly as the official LaMP-QA evaluator does:
each (response, aspect) pair is scored 0/1/2 by the judge at temperature 0.0,
malformed JSON is retried with the temperature ramped up by 0.1 each round, and
the per-question score is sum(scores) / (2 * |E|), i.e. rescaled to [0, 1].

Requests are issued with a large concurrency so the judge GPUs stay saturated;
this is the dominant GPU consumer during both training and evaluation.
"""

import asyncio
import json
import os

import json5

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


def format_aspect(aspect):
    return (
        f'-aspect: {aspect["aspect"]}\n'
        f'    -reason: {aspect["reason"]}\n'
        f'    -evidence: {aspect["evidence"]}'
    )


class RubricJudge:
    """Async client for the rubric judge.

    `score_batch` takes aligned lists and returns one score in [0, 1] per item.
    """

    def __init__(
        self,
        base_url=None,
        model=None,
        concurrency=256,
        max_retries=8,
        max_tokens=256,
        timeout=600.0,
    ):
        # JUDGE_URL may list several replicas, comma separated; requests are
        # spread over them round-robin so every judge GPU stays saturated.
        raw = base_url or os.environ.get("JUDGE_URL", "http://127.0.0.1:8100/v1")
        self.base_urls = [u.strip() for u in raw.split(",") if u.strip()]
        self.base_url = self.base_urls[0]
        self.model = model or os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._clients = None
        self._rr = 0
        # Number of (response, aspect) pairs that exhausted `max_retries` and
        # were scored 0.0. A dead judge server is indistinguishable from a
        # genuinely unaddressed aspect in the returned scores, so callers that
        # persist those scores MUST check this — see `assert_healthy`.
        self.failures = 0
        self.calls = 0

    def _get_client(self):
        # Created lazily so the object stays picklable / fork-safe.
        if self._clients is None:
            from openai import AsyncOpenAI

            self._clients = [
                AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=self.timeout, max_retries=0)
                for u in self.base_urls
            ]
        client = self._clients[self._rr % len(self._clients)]
        self._rr += 1
        return client

    async def _score_one(self, sem, question, details, response, aspect):
        client = self._get_client()
        messages = [
            {"role": "system", "content": _EVAL_PROMPT_SYSTEM},
            {
                "role": "user",
                "content": _EVAL_PROMPT_USER.format(
                    question=question,
                    details=details,
                    response=response,
                    aspects=format_aspect(aspect),
                ),
            },
        ]
        temperature = 0.0
        async with sem:
            self.calls += 1
            for _ in range(self.max_retries):
                try:
                    out = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        top_p=0.95,
                        max_tokens=self.max_tokens,
                    )
                    score = float(parse_json(out.choices[0].message.content)["match_score"])
                    # Guard against the judge inventing out-of-range scores.
                    return max(0.0, min(2.0, score))
                except Exception:
                    temperature = min(1.0, temperature + 0.1)
            self.failures += 1
            return 0.0

    async def _score_all(self, questions, details, responses, aspects_list):
        sem = asyncio.Semaphore(self.concurrency)
        tasks, owners = [], []
        for i, (q, d, r, asps) in enumerate(zip(questions, details, responses, aspects_list)):
            # An empty response (e.g. broken tag schema) scores 0 without
            # burning judge capacity on it.
            if not (r or "").strip():
                continue
            for asp in asps:
                tasks.append(self._score_one(sem, q, d, r, asp))
                owners.append(i)

        results = await asyncio.gather(*tasks) if tasks else []

        totals = [0.0] * len(questions)
        for owner, score in zip(owners, results):
            totals[owner] += score
        return [
            totals[i] / (2.0 * len(asps)) if asps else 0.0
            for i, asps in enumerate(aspects_list)
        ]

    def score_batch(self, questions, details, responses, aspects_list):
        """Blocking wrapper: list of mu(x, y, E) in [0, 1]."""
        return asyncio.run(self._score_all(questions, details, responses, aspects_list))

    async def score_batch_async(self, questions, details, responses, aspects_list):
        return await self._score_all(questions, details, responses, aspects_list)

    def assert_healthy(self, max_failure_rate=0.002):
        """Raise if too many judge calls exhausted their retries.

        A judge server that dies mid-batch returns 0.0 for every remaining call,
        which is indistinguishable from "the response addressed no aspect". That
        is harmless for a one-off eval (the score is visibly wrong) but silently
        corrupts anything that *persists* the scores — it once wrote 2,447 of
        6,000 SFT targets as an arbitrary candidate instead of the best-of-5.
        """
        if self.calls and self.failures > max_failure_rate * self.calls:
            raise RuntimeError(
                f"judge failed on {self.failures}/{self.calls} calls "
                f"({100.0 * self.failures / self.calls:.1f}%) after {self.max_retries} "
                f"retries each — the server at {','.join(self.base_urls)} is probably "
                f"down. Refusing to treat those as zero scores."
            )
        if self.failures:
            print(f"judge: {self.failures}/{self.calls} calls scored 0 after "
                  f"{self.max_retries} retries (within tolerance)", flush=True)


def wait_for_server(base_url=None, model=None, timeout=3600):
    """Block until every listed judge replica answers, so callers can start eagerly."""
    import time

    import requests

    raw = base_url or os.environ.get("JUDGE_URL", "http://127.0.0.1:8100/v1")
    for url in [u.strip() for u in raw.split(",") if u.strip()]:
        deadline = time.time() + timeout
        while True:
            try:
                if requests.get(f"{url}/models", timeout=10, proxies={"http": None, "https": None}).status_code == 200:
                    break
            except Exception:
                pass
            if time.time() > deadline:
                raise RuntimeError(f"server at {url} did not come up within {timeout}s")
            time.sleep(5)
    return True
