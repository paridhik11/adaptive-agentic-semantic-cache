"""Privacy-preserving descriptive repetition counts for conversation roots."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import unicodedata


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def summarize_first_turns(rows: list[dict], *, limit: int = 5000) -> dict:
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    roots = []
    assistant_reply_dates: dict[str, list[datetime]] = {}
    for row in rows:
        if (row.get("role") == "assistant" and row.get("parent_id") is not None
                and row.get("deleted") is not True and row.get("synthetic") is not True):
            tree = row.get("message_tree_id")
            timestamp = row.get("created_date")
            if isinstance(tree, str) and tree and isinstance(timestamp, str):
                try:
                    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError as error:
                    raise ValueError("invalid created_date on assistant reply") from error
                if parsed.tzinfo is None:
                    raise ValueError("assistant reply created_date must include a timezone")
                assistant_reply_dates.setdefault(tree, []).append(parsed)
    for row in rows:
        if row.get("role") != "prompter" or row.get("parent_id") is not None:
            continue
        if row.get("deleted") is True or row.get("synthetic") is True:
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        tree = row.get("message_tree_id")
        user = row.get("user_id")
        timestamp = row.get("created_date")
        message_id = row.get("message_id")
        if not all(isinstance(value, str) and value for value in (tree, timestamp, message_id)):
            continue
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("invalid created_date on eligible root message") from error
        if parsed.tzinfo is None:
            raise ValueError("created_date must include a timezone")
        roots.append((parsed, message_id, tree, user if isinstance(user, str) else None, text))

    roots.sort(key=lambda item: (item[0], item[1]))
    unique = []
    seen_trees = set()
    for root in roots:
        if root[2] in seen_trees:
            continue
        seen_trees.add(root[2])
        unique.append(root)
    if len(unique) < limit:
        raise ValueError(f"requested {limit} roots but only {len(unique)} eligible roots exist")
    sample = unique[:limit]

    seen_exact: set[str] = set()
    seen_canonical: set[str] = set()
    prior_exact_reply_dates: dict[str, datetime] = {}
    prior_canonical_reply_dates: dict[str, datetime] = {}
    seen_exact_by_user: dict[str, set[str]] = {}
    seen_canonical_by_user: dict[str, set[str]] = {}
    exact = canonical = exact_same_user = canonical_same_user = 0
    exact_prior_answer_available = canonical_prior_answer_available = 0
    same_user_denominator = 0
    by_language = Counter()

    # Map only aggregate language labels by message ID; text and IDs never enter the result.
    root_language = {
        row.get("message_id"): row.get("lang")
        for row in rows
        if row.get("role") == "prompter" and row.get("parent_id") is None
    }
    for root_time, message_id, tree, user, text in sample:
        exact_key = text
        canonical_key = _normalized(text)
        exact += exact_key in seen_exact
        canonical += canonical_key in seen_canonical
        exact_prior_answer_available += prior_exact_reply_dates.get(exact_key, root_time) < root_time
        canonical_prior_answer_available += prior_canonical_reply_dates.get(canonical_key, root_time) < root_time
        seen_exact.add(exact_key)
        seen_canonical.add(canonical_key)
        tree_replies = assistant_reply_dates.get(tree, [])
        if tree_replies:
            earliest_reply = min(tree_replies)
            prior_exact_reply_dates[exact_key] = min(prior_exact_reply_dates.get(exact_key, earliest_reply), earliest_reply)
            prior_canonical_reply_dates[canonical_key] = min(prior_canonical_reply_dates.get(canonical_key, earliest_reply), earliest_reply)
        if user is not None:
            same_user_denominator += 1
            own_exact = seen_exact_by_user.setdefault(user, set())
            own_canonical = seen_canonical_by_user.setdefault(user, set())
            exact_same_user += exact_key in own_exact
            canonical_same_user += canonical_key in own_canonical
            own_exact.add(exact_key)
            own_canonical.add(canonical_key)
        language = root_language.get(message_id)
        by_language[language if isinstance(language, str) and language else "unknown"] += 1

    return {
        "schema_version": "natural-repetition-summary-v1",
        "sample_method": "first eligible root prompts in timestamp order from one pinned dataset split",
        "sample_size": len(sample),
        "eligible_root_count_in_shard": len(unique),
        "sample_start_utc": sample[0][0].isoformat(),
        "sample_end_utc": sample[-1][0].isoformat(),
        "global_exact_repeat_count": exact,
        "global_exact_repeat_share": exact / len(sample),
        "global_nfkc_casefold_whitespace_repeat_count": canonical,
        "global_nfkc_casefold_whitespace_repeat_share": canonical / len(sample),
        "exact_repeats_with_prior_assistant_reply": exact_prior_answer_available,
        "canonical_repeats_with_prior_assistant_reply": canonical_prior_answer_available,
        "assistant_reply_availability_is_not_answer_review": True,
        "known_user_id_count_in_sample": len(seen_exact_by_user),
        "same_user_denominator": same_user_denominator,
        "same_user_exact_repeat_count": exact_same_user,
        "same_user_exact_repeat_share": exact_same_user / same_user_denominator if same_user_denominator else None,
        "same_user_canonical_repeat_count": canonical_same_user,
        "same_user_canonical_repeat_share": canonical_same_user / same_user_denominator if same_user_denominator else None,
        "language_counts": dict(sorted(by_language.items())),
        "raw_prompts_exported": False,
        "user_ids_exported": False,
        "limitations": [
            "This is a descriptive sample of crowdsourced OpenAssistant prompts, not production API traffic or a representative population sample.",
            "Repeat counts do not establish answer correctness, safe semantic reuse, or tenant compatibility.",
            "User ID is only a within-source-user grouping key and is not equivalent to a deployment tenant.",
            "The dataset contains human-submitted prompts but the corpus may include model exposure; pretraining contamination is unknown.",
        ],
    }
