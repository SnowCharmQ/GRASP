"""Prompts and helpers for the official LaMP-QA RAG evaluation.

Transcribed from the LaMP-QA benchmark's own ``data/formetters.py``. Only the
official RAG protocol is kept -- that is the single mode GRASP is evaluated
under. ``build_official_messages`` emits the exact system+user turn the
benchmark scores against; ``OFFICIAL_RAG_USER`` carries the specific leading and
trailing newlines the official formatter produces, so it must not be
re-derived by hand.
"""


OFFICIAL_RAG_SYSTEM = "You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions and detailed descriptions of these questions.\n# Your input:\n    - The user's current question from a post.\n    - The user's past post questions and detailed descriptions of these questions.\n# Your task: Answer the user's current question in a personalized way by considering this user's past post questions and detailed descriptions of these questions, to learn about the user's preferences.\n# Your output: You should generate personalized answer to the user's current question by considering this user's past post questions and detailed descriptions of these questions to learn about user's preferences. Your output should be a valid json object in ```json ``` block that contains the following fields:\n    - personalized_answer: contains the personalized answer to the user's current question considering the this user's past post questions and detailed descriptions of these questions to learn about user's preferences.\n"

OFFICIAL_RAG_USER = '\n# Past post questions and detailed descriptions of these questions:\n{profile}\n# Current post question:\n{question}\n'


def format_profile(profile, num_contexts=10):
    """Serialize the top-`num_contexts` ranked profile entries (LaMP-QA style)."""
    return "\n\n".join(p["text"] for p in profile[:num_contexts])


def build_official_messages(system_prompt, question, profile, num_contexts=10):
    """The official LaMP-QA RAG turn: system prompt + personalized context."""
    user = OFFICIAL_RAG_USER.format(
        profile=format_profile(profile, num_contexts), question=question)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
