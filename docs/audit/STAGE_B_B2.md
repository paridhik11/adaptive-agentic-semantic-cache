# B2 rejection-only signature veto

Evidence: builder-authored synthetic dev diagnostics, not human ground truth.
Rules parse QUESTION TEXT ONLY: operation, input/output, direction, numeric
literals, units, versions, polarity and known requirements. Unknown domain or
unsupported constraint rejects. Rules cannot approve reuse, resolve ambiguity,
promote a runner-up or fall back after rejecting the classifier's top choice.

The lexical domain dictionary is builder-authored and aligned with synthetic
tasks; perfect fixture rejection is not natural-language generalization. On
64 designated groups: all 192 paraphrases survive the veto; 64 cases in each
other nonexact family are rejected. This includes entity changes that may
correctly match ANOTHER cached seed, which is handled separately in B3.
Veto-dev-summary.json records every family count; actual classifier-conditional
additional rejections and wrongly killed positives are measured in B3.

All 727 Stage A dev signature records were mechanically reviewed and
quarantined: no independently verified signature status and/or missing typed
fields for direction/version/quantity/units/polarity. Completed builder metadata
is not verified metadata. No Stage A labels or signatures were changed; none
is a Stage B policy input. The raw audit is legacy-signature-review.json.

Commands:

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python scripts/review_stage_b_veto.py
PYTHONPATH=.venv/lib/python3.12/site-packages python -m pytest -q tests/test_intent_stage_a.py tests/test_intent_stage_b.py
```

Focused engineering suite: 17 passed in 4.40s. No final read/evaluation, serving
change, provider call or natural-set claim.
