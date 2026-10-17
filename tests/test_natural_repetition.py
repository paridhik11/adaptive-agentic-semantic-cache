import json

import pytest

from src.benchmark.natural_repetition import summarize_first_turns


def _row(message_id, when, text, *, user="u1", tree=None, role="prompter", parent=None,
         deleted=False, synthetic=False, lang="en"):
    return {"message_id": message_id, "created_date": when, "text": text, "user_id": user,
            "message_tree_id": tree or message_id, "role": role, "parent_id": parent,
            "deleted": deleted, "synthetic": synthetic, "lang": lang}


def test_chronological_exact_and_canonical_counts_respect_user_scope():
    rows = [
        _row("m1", "2023-01-01T00:00:00+00:00", "Hello world", user="u1"),
        _row("m2", "2023-01-02T00:00:00+00:00", "Hello world", user="u2"),
        _row("m3", "2023-01-03T00:00:00+00:00", " hello   WORLD ", user="u1"),
        _row("m4", "2023-01-01T12:00:00+00:00", "PRIVATE ANSWER", tree="m1", role="assistant", parent="m1"),
        _row("m5", "2023-01-05T00:00:00+00:00", "ignored deleted", deleted=True),
        _row("m6", "2023-01-06T00:00:00+00:00", "ignored synthetic", synthetic=True),
    ]

    result = summarize_first_turns(rows, limit=3)

    assert result["global_exact_repeat_count"] == 1
    assert result["global_nfkc_casefold_whitespace_repeat_count"] == 2
    assert result["exact_repeats_with_prior_assistant_reply"] == 1
    assert result["canonical_repeats_with_prior_assistant_reply"] == 2
    assert result["assistant_reply_availability_is_not_answer_review"] is True
    assert result["same_user_exact_repeat_count"] == 0
    assert result["same_user_canonical_repeat_count"] == 1
    assert result["sample_start_utc"] < result["sample_end_utc"]
    assert result["raw_prompts_exported"] is False
    assert result["user_ids_exported"] is False
    serialized = json.dumps(result)
    assert all(secret not in serialized for secret in ("Hello world", "u1", "m1", "PRIVATE ANSWER"))


def test_repeated_tree_is_counted_once():
    rows = [
        _row("m1", "2023-01-01T00:00:00+00:00", "same", tree="tree"),
        _row("m2", "2023-01-02T00:00:00+00:00", "same", tree="tree"),
    ]
    result = summarize_first_turns(rows, limit=1)
    assert result["eligible_root_count_in_shard"] == 1
    assert result["global_exact_repeat_count"] == 0


def test_requires_aware_dates_and_enough_rows():
    with pytest.raises(ValueError, match="timezone"):
        summarize_first_turns([_row("m1", "2023-01-01T00:00:00", "q")], limit=1)
    with pytest.raises(ValueError, match="only 1 eligible roots"):
        summarize_first_turns([_row("m1", "2023-01-01T00:00:00+00:00", "q")], limit=2)


def test_normalization_is_unicode_casefold_and_whitespace_only():
    rows = [
        _row("m1", "2023-01-01T00:00:00+00:00", "Straße  test"),
        _row("m2", "2023-01-02T00:00:00+00:00", "STRASSE test"),
        _row("m3", "2023-01-03T00:00:00+00:00", "STRASSE, test"),
    ]
    result = summarize_first_turns(rows, limit=3)
    assert result["global_exact_repeat_count"] == 0
    assert result["global_nfkc_casefold_whitespace_repeat_count"] == 1


def test_reply_after_repeat_does_not_count_as_prior_answer_availability():
    rows = [
        _row("m1", "2023-01-01T00:00:00+00:00", "same", tree="t1"),
        _row("m2", "2023-01-02T00:00:00+00:00", "same", tree="t2"),
        _row("a1", "2023-01-03T00:00:00+00:00", "PRIVATE ANSWER", tree="t1", role="assistant", parent="m1"),
    ]
    result = summarize_first_turns(rows, limit=2)
    assert result["global_exact_repeat_count"] == 1
    assert result["exact_repeats_with_prior_assistant_reply"] == 0
    assert result["canonical_repeats_with_prior_assistant_reply"] == 0


def test_later_duplicate_thread_can_supply_earlier_available_reply():
    rows = [
        _row("m1", "2023-01-01T00:00:00+00:00", "same", tree="t1"),
        _row("m2", "2023-01-02T00:00:00+00:00", "same", tree="t2"),
        _row("m3", "2023-01-03T00:00:00+00:00", "same", tree="t3"),
        _row("a1", "2023-01-05T00:00:00+00:00", "late", tree="t1", role="assistant", parent="m1"),
        _row("a2", "2023-01-02T12:00:00+00:00", "earlier", tree="t2", role="assistant", parent="m2"),
    ]
    result = summarize_first_turns(rows, limit=3)
    assert result["global_exact_repeat_count"] == 2
    assert result["exact_repeats_with_prior_assistant_reply"] == 1
