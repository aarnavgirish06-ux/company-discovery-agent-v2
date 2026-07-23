"""
entity_matching.py

Deterministic legal-entity matching. This is the single verification layer
that every candidate identifier (GST, CIN, ...) must pass before being
accepted -- it answers "is this page genuinely about the exact legal
entity that was requested," not "is this page about something similar."

WHY THIS IS SEPARATE FROM identifier_lookup.py:

Company verification and identifier extraction are different concerns.
identifier_lookup.py decides *where* in a page to look for an identifier
once a page has been confirmed to be about the right company; this module
decides whether that confirmation is warranted at all. Keeping them apart
means the matching rules can be reasoned about (and tested) independently
of pattern/checksum/corroboration logic.

DESIGN: NORMALIZE LEGAL FORM, DON'T DELETE IT

A prior version of this matching logic stripped legal-form words ("Pvt",
"Ltd", "Private", "Limited", ...) out of both the query and the page text
before comparing them, then used fuzzy substring similarity on what was
left. That has two failure modes:

1. Deleting legal-form words means legal form is never actually verified
   -- "Shree Samarth Packaging Pvt Ltd" and "Shree Samarth Packaging LLP"
   become indistinguishable once "Pvt Ltd" and "LLP" are both erased.

2. Fuzzy substring similarity (e.g. rapidfuzz's partial_ratio) measures
   whether the shorter string's characters align well *somewhere* inside
   the longer one -- it does not penalize the longer string for having
   extra substantive words. That's why "Infosys Limited" fuzzily matched
   "Infosys Consulting India Limited", and "Shree Samarth Packaging Pvt
   Ltd" fuzzily matched "Shree Swami Samarth Packaging Pvt Ltd": both
   extra words (`Consulting India`, `Swami`) get absorbed into a still-high
   similarity score.

This module instead:

- Canonicalizes legal-form tokens ("Pvt" <-> "Private", "Ltd" <->
  "Limited", ...) via a lookup table, so equivalent legal forms compare as
  identical -- WITHOUT deleting them.
- Splits a company name into `substantive_tokens` (the actual name) and
  `legal_form` (a canonicalized tuple of trailing legal-form tokens, or
  None if the query didn't specify one).
- Requires an EXACT, contiguous match of `substantive_tokens` in the page
  text -- not a fuzzy score -- with symmetric boundary checks so that a
  match embedded inside a longer name (on either side) is rejected. This
  is what actually distinguishes "Infosys" from "Infosys Consulting
  India" and "Shree Samarth Packaging" from "Shree Swami Samarth
  Packaging": neither is a fuzzy-similarity problem, they're an exact
  contiguous-run-with-clean-boundaries problem.
- If the query specified a legal form, requires the page's trailing legal
  form (when the page states one at all) to canonicalize to the same
  form. A page mentioning the company without stating any legal form in
  that particular sentence is treated as unconstrained, not as a mismatch
  -- only an explicit, different legal form is a rejection.

PAN-CATEGORY CROSS-CHECK:

A GSTIN embeds a PAN, whose 4th character encodes the holder's category
(Company, Individual, Firm, Trust, ...). When the requested legal form
maps unambiguously to a PAN category, a GST candidate whose embedded PAN
category doesn't match is rejected -- e.g. a "Private Limited" request
should never accept a GSTIN whose PAN says Individual.

This is intentionally only enforced where the mapping is unambiguous
(currently: Company-type forms -> "C", Proprietorship -> "P"). Forms like
LLP are deliberately NOT mapped: sources disagree on whether LLPs are
filed under the Firm ("F") category or their own convention, so guessing
would trade one silent false-accept bug for a silent false-reject bug.
Skipping validation for an unmapped form is a deliberate no-op, not an
oversight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Legal-form canonicalization (Indian entity types only -- GST/CIN are
# India-specific, so there's no reason to cover foreign forms like "GmbH"
# or "Pte Ltd" here).
#
# Maps a raw, lowercased token -> its canonical form. Both abbreviated
# and spelled-out variants map to the same canonical value, which is what
# makes "Pvt Ltd" and "Private Limited" compare as identical without
# either being deleted.
LEGAL_FORM_SYNONYMS: Dict[str, str] = {
    "pvt": "private",
    "private": "private",
    "ltd": "limited",
    "limited": "limited",
    "llp": "llp",
    "co": "company",
    "company": "company",
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "opc": "opc",
    "prop": "proprietorship",
    "proprietorship": "proprietorship",
    "partnership": "partnership",
    "firm": "firm",
    "trust": "trust",
    "society": "society",
}

# Maps a canonicalized, ordered legal-form tuple -> the PAN category
# character (GSTIN characters 2-12 are the PAN; PAN character index 3 is
# the category code) it unambiguously implies. Only forms with a single,
# undisputed category are listed here -- see module docstring for why
# LLP, partnership, trust, society, and firm are deliberately absent.
_LEGAL_FORM_TO_PAN_CATEGORY: Dict[Tuple[str, ...], str] = {
    ("private", "limited"): "C",  # Company
    ("limited",): "C",            # Company
    ("opc",): "C",                # One Person Company -> still Company
    ("proprietorship",): "P",     # Individual
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# English "closed-class" function words: articles, prepositions,
# conjunctions, pronouns, and common copular/auxiliary verb forms. This is
# a small, linguistically closed (i.e. not open to new entries the way
# nouns/verbs/proper nouns are) vocabulary, used ONLY to recognize a clean
# grammatical boundary immediately after/before a name match -- e.g.
# "Acme Traders IS listed..." ends the name at "Traders" because "is"
# cannot plausibly be part of a company name, whereas "Acme Traders
# CONSULTING is listed..." does NOT end at "Traders" because "consulting"
# is an open-class word that could plausibly continue the name. This
# plays the same role for ordinary sentence text that the legal-form
# table plays for "Pvt"/"Ltd": both are small, closed, enumerable
# vocabularies, not fuzzy heuristics or per-case rules.
_BOUNDARY_WORDS = frozenset(
    {
        # Articles / determiners
        "a", "an", "the", "this", "that", "these", "those",
        # Prepositions
        "of", "in", "on", "at", "for", "to", "by", "from", "with", "as",
        "into", "onto", "near", "under", "over", "about",
        # Conjunctions
        "and", "or", "but", "nor",
        # Pronouns / relative pronouns
        "it", "its", "which", "who", "whose",
        # Copular / auxiliary verb forms
        "is", "was", "are", "were", "be", "being", "been",
        "has", "have", "had", "will", "would", "can", "could",
    }
)


@dataclass(frozen=True)
class ParsedEntity:
    """
    A company name split into its substantive part and its (optional)
    legal form.

    `legal_form` is None when the name has no recognized trailing
    legal-form tokens at all -- e.g. a bare "Infosys" query -- which is
    the signal used both by `find_entity_mentions()` (to leave legal form
    unconstrained) and by `pan_category_matches()` (to skip PAN-category
    validation entirely, since the entity type can't be inferred).
    """
    substantive_tokens: Tuple[str, ...]
    legal_form: Optional[Tuple[str, ...]]


def _tokenize(text: str) -> List[str]:
    """Lowercases and splits text into alphanumeric tokens, discarding all punctuation as separators."""
    return _TOKEN_PATTERN.findall(text.lower())


def _is_legal_form_token(token: str) -> bool:
    return token in LEGAL_FORM_SYNONYMS


def parse_entity_name(name: str) -> ParsedEntity:
    """
    Splits `name` into substantive tokens and a canonicalized legal-form
    tuple, by peeling the maximal contiguous run of recognized legal-form
    tokens off the END of the token list. Company names put legal form at
    the end ("... Private Limited"), never the start, so only the
    trailing run is considered.

    If every token happens to be a recognized legal-form token (a
    degenerate case), the split is not applied -- the whole name is
    treated as substantive rather than reducing it to nothing.
    """
    tokens = _tokenize(name)

    split_idx = len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        if not _is_legal_form_token(tokens[i]):
            break
        split_idx = i

    if split_idx == 0:
        split_idx = len(tokens)

    substantive = tuple(tokens[:split_idx])
    legal_form_tokens = tokens[split_idx:]
    legal_form = tuple(LEGAL_FORM_SYNONYMS[t] for t in legal_form_tokens) if legal_form_tokens else None

    return ParsedEntity(substantive_tokens=substantive, legal_form=legal_form)


def _boundary_is_valid(
    tokens: List[str],
    start: int,
    end: int,
    requested_legal_form: Optional[Tuple[str, ...]],
) -> bool:
    """
    Given a candidate match of the substantive tokens at tokens[start:end],
    decides whether it's a genuine full-name mention rather than a
    fragment of a longer name.

    Symmetric boundary check:
    - If a token immediately precedes the match and it is NOT a
      recognized legal-form token AND NOT a closed-class boundary word
      (see `_BOUNDARY_WORDS`), the match is a suffix of a longer name
      (e.g. requested "Samarth Packaging" found inside "Shree Samarth
      Packaging") -> reject.
    - If a token immediately follows the match and it is NOT a recognized
      legal-form token AND NOT a closed-class boundary word, the match is
      a prefix of a longer name (e.g. requested "Infosys" found inside
      "Infosys Consulting India") -> reject.

    Closed-class function words (articles, prepositions, conjunctions,
    pronouns, common copular/auxiliary verbs) are treated the same as a
    sentence boundary here because they are grammatically incapable of
    being part of a company's proper name -- "Acme Traders IS listed"
    cleanly ends the name at "Traders", the same as if a period sat
    there, while "Acme Traders CONSULTING is listed" does not, since
    "consulting" is an ordinary open-class word that could genuinely be
    part of a longer registered name.

    If both boundaries are clean, and the query specified a legal form,
    any legal-form tokens immediately following the match must
    canonicalize to exactly that form. A page that mentions the company
    without stating a legal form in this particular sentence is treated
    as unconstrained (not a mismatch) -- only a stated, different legal
    form is a rejection.
    """
    def _is_clean_boundary_token(token: str) -> bool:
        return _is_legal_form_token(token) or token in _BOUNDARY_WORDS

    if start > 0 and not _is_clean_boundary_token(tokens[start - 1]):
        return False

    if end < len(tokens) and not _is_clean_boundary_token(tokens[end]):
        return False

    if requested_legal_form is None:
        return True

    trailing_legal_form: List[str] = []
    j = end
    while j < len(tokens) and _is_legal_form_token(tokens[j]):
        trailing_legal_form.append(LEGAL_FORM_SYNONYMS[tokens[j]])
        j += 1

    if not trailing_legal_form:
        return True  # No legal form stated here -- unconstrained, not a mismatch.

    return tuple(trailing_legal_form) == requested_legal_form


def find_entity_mentions(page_text: str, requested: ParsedEntity) -> List[int]:
    """
    Finds every sentence-like segment of `page_text` that contains a
    genuine, full mention of `requested` -- an exact, contiguous run of
    its substantive tokens with clean boundaries on both sides (see
    `_boundary_is_valid`) -- and returns the character offset of each
    such segment.

    Returns [] (meaning "this document does not verify as being about the
    requested entity") if `requested.substantive_tokens` is empty, or if
    no segment contains a valid mention. This preserves the existing
    existence-gating behavior in identifier_lookup.py: a document with no
    verified mentions is skipped entirely.
    """
    substantive = requested.substantive_tokens
    n = len(substantive)
    if n == 0:
        return []

    mention_positions: List[int] = []

    for match in re.finditer(r"[^.\n]+", page_text):
        segment = match.group(0)
        tokens = _tokenize(segment)
        if len(tokens) < n:
            continue

        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i:i + n]) != substantive:
                continue
            if _boundary_is_valid(tokens, i, i + n, requested.legal_form):
                mention_positions.append(match.start())
                break  # One valid mention is enough to count this segment.

    return mention_positions


def pan_category_matches(gstin: str, requested: ParsedEntity) -> bool:
    """
    Cross-checks a GSTIN's embedded PAN category (PAN character index 3)
    against the entity type implied by `requested.legal_form`.

    Returns True (i.e. does not reject) whenever:
    - the query specified no legal form at all (entity type can't be
      inferred), or
    - the specified legal form isn't in `_LEGAL_FORM_TO_PAN_CATEGORY`
      (an intentionally unmapped, ambiguous form like LLP).

    Only returns False on a confident, unambiguous mismatch -- e.g. a
    "Private Limited" request whose candidate GSTIN's PAN category is "P"
    (Individual) rather than "C" (Company).
    """
    if requested.legal_form is None:
        return True

    expected_category = _LEGAL_FORM_TO_PAN_CATEGORY.get(requested.legal_form)
    if expected_category is None:
        return True

    pan = gstin[2:12]
    if len(pan) < 4:
        return True  # Malformed PAN slice; let format/checksum validation handle it elsewhere.

    return pan[3] == expected_category
