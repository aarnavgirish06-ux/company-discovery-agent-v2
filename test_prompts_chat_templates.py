"""
test_prompts_chat_templates.py

Unit tests confirming DISCOVERY_PROMPT / EVIDENCE_PROMPT format correctly,
in particular that literal curly braces (the JSON example in each system
prompt's OUTPUT FORMAT section, and potentially arbitrary braces in
scraped webpage text substituted into the human turn) are never mistaken
for template variables.

Run with: python3 test_prompts_chat_templates.py
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from prompts import (
    DISCOVERY_PROMPT,
    EVIDENCE_PROMPT,
    EVIDENCE_SYSTEM_PROMPT,
    INTENT_PROMPT,
    INTENT_SYSTEM_PROMPT,
    QA_PROMPT,
    QA_SYSTEM_PROMPT,
    RESPONSE_SYNTHESIS_PROMPT,
    RESPONSE_SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


discovery_messages = DISCOVERY_PROMPT.format_messages(user_prompt="Find pharma companies in Powai")

check("DISCOVERY_PROMPT produces exactly 2 messages", len(discovery_messages) == 2)
check("DISCOVERY_PROMPT's first message is a SystemMessage", isinstance(discovery_messages[0], SystemMessage))
check(
    "DISCOVERY_PROMPT's system message content is SYSTEM_PROMPT verbatim, braces included",
    discovery_messages[0].content == SYSTEM_PROMPT,
)
check("DISCOVERY_PROMPT's second message is a HumanMessage", isinstance(discovery_messages[1], HumanMessage))
check(
    "DISCOVERY_PROMPT's human message content is exactly the supplied user_prompt",
    discovery_messages[1].content == "Find pharma companies in Powai",
)

brace_heavy_query = 'Find companies similar to "{Acme}" with turnover {>100cr}'
discovery_messages_with_braces = DISCOVERY_PROMPT.format_messages(user_prompt=brace_heavy_query)
check(
    "A user_prompt containing literal braces passes through DISCOVERY_PROMPT unchanged",
    discovery_messages_with_braces[1].content == brace_heavy_query,
)

evidence_messages = EVIDENCE_PROMPT.format_messages(user_prompt="Document text with a { curly brace } in it")

check("EVIDENCE_PROMPT produces exactly 2 messages", len(evidence_messages) == 2)
check(
    "EVIDENCE_PROMPT's system message content is EVIDENCE_SYSTEM_PROMPT verbatim",
    evidence_messages[0].content == EVIDENCE_SYSTEM_PROMPT,
)
check(
    "EVIDENCE_PROMPT's human message passes through literal braces unchanged",
    evidence_messages[1].content == "Document text with a { curly brace } in it",
)

intent_messages = INTENT_PROMPT.format_messages(
    user_prompt='Conversation so far:\nNo companies have been discussed yet in this conversation.\n\nNo company is currently in focus.\n\nNewest user message: "test"'
)

check("INTENT_PROMPT produces exactly 2 messages", len(intent_messages) == 2)
check("INTENT_PROMPT's first message is a SystemMessage", isinstance(intent_messages[0], SystemMessage))
check(
    "INTENT_PROMPT's system message content is INTENT_SYSTEM_PROMPT verbatim",
    intent_messages[0].content == INTENT_SYSTEM_PROMPT,
)
check("INTENT_PROMPT's second message is a HumanMessage", isinstance(intent_messages[1], HumanMessage))

response_synthesis_messages = RESPONSE_SYNTHESIS_PROMPT.format_messages(
    user_prompt='Known facts:\n(none)\n\nUser message: "test"'
)

check("RESPONSE_SYNTHESIS_PROMPT produces exactly 2 messages", len(response_synthesis_messages) == 2)
check(
    "RESPONSE_SYNTHESIS_PROMPT's system message content is RESPONSE_SYNTHESIS_SYSTEM_PROMPT verbatim",
    response_synthesis_messages[0].content == RESPONSE_SYNTHESIS_SYSTEM_PROMPT,
)

check("INTENT_SYSTEM_PROMPT mentions the new COMPANY_QUESTION intent", "COMPANY_QUESTION" in INTENT_SYSTEM_PROMPT)

qa_messages = QA_PROMPT.format_messages(user_prompt='Question: "test"\n\n(no evidence)')

check("QA_PROMPT produces exactly 2 messages", len(qa_messages) == 2)
check("QA_PROMPT's first message is a SystemMessage", isinstance(qa_messages[0], SystemMessage))
check(
    "QA_PROMPT's system message content is QA_SYSTEM_PROMPT verbatim",
    qa_messages[0].content == QA_SYSTEM_PROMPT,
)
check("QA_PROMPT's second message is a HumanMessage", isinstance(qa_messages[1], HumanMessage))

print()
print("=" * 70)
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S):")
    for f in _FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
else:
    print("ALL TESTS PASSED")
