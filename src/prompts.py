"""LaMP-QA prompts and teacher-only privileged information (PI).

Keep official templates verbatim. Teacher messages extend the student
system/user messages; assert_template_invariant checks this contract."""

from __future__ import annotations

# Official LaMP-QA protocol (verbatim -- do not edit)

OFFICIAL_RAG_SYSTEM = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions and detailed descriptions of these questions.
# Your input:
    - The user's current question from a post.
    - The user's past post questions and detailed descriptions of these questions.
# Your task: Answer the user's current question in a personalized way by considering this user's past post questions and detailed descriptions of these questions, to learn about the user's preferences.
# Your output: You should generate personalized answer to the user's current question by considering this user's past post questions and detailed descriptions of these questions to learn about user's preferences. Your output should be a valid json object in ```json ``` block that contains the following fields:
    - personalized_answer: contains the personalized answer to the user's current question considering the this user's past post questions and detailed descriptions of these questions to learn about user's preferences.
"""

OFFICIAL_RAG_USER = """
# Past post questions and detailed descriptions of these questions:
{profile}
# Current post question:
{question}
"""


def format_profile(profile, num_contexts: int = 10) -> str:
    """Serialize the top-``num_contexts`` ranked profile entries (LaMP-QA style)."""
    return "\n\n".join(p["text"] for p in profile[:num_contexts])


# Teacher-only PI extends the student prompt without changing its shared text.

_PI_HEADER = "# What this user is actually looking for (private editorial notes):"

_PI_NARRATIVE_HEADER = "## The user's own elaboration of the question:"

_PI_ASPECTS_HEADER = "## The specific aspects this user expects the answer to address:"

# Instruction appended to the *system* prompt for the teacher only. It tells the
# teacher what to do with the PI without changing the output contract.
_PI_SYSTEM_CLAUSE = """
# Additional private context (this turn only):
You are additionally given private editorial notes describing what this specific user is actually looking for: their own elaboration of the question, and the concrete aspects they expect the answer to address. Use these notes to decide what to cover and how to prioritise it. Never mention, quote, restate or allude to the notes themselves -- the reader must not be able to tell that you had them. Your output format is unchanged.
"""

VALID_PI_MODES = ("none", "narrative", "aspects", "all")


class PILeakInStudentContext(Exception):
    """The retrieved student context contains privileged information; drop the row."""


def format_aspects(aspects) -> str:
    """Render the rubric aspects for the teacher."""
    lines = []
    for a in aspects:
        lines.append(f"- {a['aspect']}")
        reason = a.get("reason")
        if reason:
            lines.append(f"    - why it matters to this user: {reason}")
        evidence = a.get("evidence")
        if evidence:
            lines.append(f"    - in the user's own words: {evidence}")
    return "\n".join(lines)


def build_pi_block(details, aspects, pi_mode: str) -> str:
    """Build teacher-only narrative, aspects, or both; mode none returns no block.
    Missing requested content also produces an empty block."""
    if pi_mode == "none":
        return ""

    if pi_mode not in VALID_PI_MODES:
        raise ValueError(f"unknown pi_mode {pi_mode!r}; expected one of {'|'.join(VALID_PI_MODES)}")

    sections = []
    if pi_mode in {"narrative", "all"} and details:
        sections.append(f"{_PI_NARRATIVE_HEADER}\n{details.strip()}")
    if pi_mode in {"aspects", "all"} and aspects:
        sections.append(f"{_PI_ASPECTS_HEADER}\n{format_aspects(aspects)}")

    if not sections:
        # e.g.
        return ""

    body = "\n\n".join(sections)
    return f"\n{_PI_HEADER}\n{body}\n"


# Message builders


def build_student_messages(row, num_contexts: int = 10):
    """The student's context: question + top-k retrieved profile entries."""
    user = OFFICIAL_RAG_USER.format(
        profile=format_profile(row.get("profile") or [], num_contexts),
        question=row["question"],
    )
    return [
        {"role": "system", "content": OFFICIAL_RAG_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_teacher_messages(row, num_contexts: int = 10, pi_mode: str = "all"):
    """The teacher's context: the student's context **plus** the PI block."""
    student = build_student_messages(row, num_contexts=num_contexts)
    pi_block = build_pi_block(row.get("details"), row.get("aspects") or [], pi_mode)

    if not pi_block:
        return student

    return [
        {"role": "system", "content": OFFICIAL_RAG_SYSTEM + _PI_SYSTEM_CLAUSE},
        {"role": "user", "content": student[1]["content"] + pi_block},
    ]


# The invariant check


def assert_template_invariant(row, num_contexts: int = 10, pi_mode: str = "all"):
    """Check roles, official student text, and teacher-only prompt extensions.

    Reject leaked narrative/details and annotator reasons. Aspect titles and
    evidence may legitimately overlap the question or retrieved profile."""
    s = build_student_messages(row, num_contexts=num_contexts)
    t = build_teacher_messages(row, num_contexts=num_contexts, pi_mode=pi_mode)

    assert len(s) == len(t) == 2, f"expected 2 messages, got {len(s)}/{len(t)}"
    assert [m["role"] for m in s] == [m["role"] for m in t] == ["system", "user"], (
        "role sequence diverged"
    )

    pi_block = build_pi_block(row.get("details"), row.get("aspects") or [], pi_mode)

    if not pi_block:
        assert s == t, "with an empty PI block the teacher prompt must equal the student's"
        return True

    # (2) system prompt
    assert t[0]["content"].startswith(s[0]["content"]), (
        "teacher system prompt does not start with the student's -- the shared "
        "boilerplate diverged, so the KL would be measuring format not behaviour"
    )
    sys_extra = t[0]["content"][len(s[0]["content"]) :]
    assert sys_extra == _PI_SYSTEM_CLAUSE, (
        f"unexpected extra system text ({len(sys_extra)} chars): {sys_extra!r}"
    )

    # (3) user message
    assert t[1]["content"].startswith(s[1]["content"]), (
        "teacher user message does not start with the student's -- the profile "
        "or question region was modified rather than appended to"
    )
    user_extra = t[1]["content"][len(s[1]["content"]) :]
    assert user_extra == pi_block, (
        f"unexpected extra user text ({len(user_extra)} chars): {user_extra!r}"
    )

    # Reconstruct the official student prompt to detect extra PI bytes.
    expected_user = OFFICIAL_RAG_USER.format(
        profile=format_profile(row.get("profile") or [], num_contexts),
        question=row["question"],
    )
    assert s[1]["content"] == expected_user, (
        "student user message is not byte-identical to the official RAG "
        "template instantiated with (profile, question); something extra was "
        "injected into the student's context"
    )
    assert s[0]["content"] == OFFICIAL_RAG_SYSTEM, (
        "student system prompt drifted from the official formatter"
    )

    # Check narrative and reasons for leakage; aspect/evidence overlap is allowed.
    details = (row.get("details") or "").strip()
    if details and details in s[1]["content"]:
        raise PILeakInStudentContext(
            f"row {row.get('id')}: the question narrative appears verbatim in the "
            f"student prompt (the retrieved profile contains the asker's own "
            f"current post), so the privileged block is not privileged here"
        )
    for a in row.get("aspects") or []:
        reason = (a.get("reason") or "").strip()
        if reason and reason in s[1]["content"]:
            raise PILeakInStudentContext(
                f"row {row.get('id')}: aspect rationale {reason[:60]!r} appears "
                f"in the student prompt"
            )

    return True
