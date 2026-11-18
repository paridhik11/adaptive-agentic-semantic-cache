# Natural-wording review request — no responses supplied by Codex

Work on a COPY of `data/intent_stage_b/v2/natural-review-request.jsonl`: 50 blank slots,
balanced across the suggested families. Write BOTH questions yourself. Do not
copy generated dataset wording, model outputs, public pairs or this project's
formatter templates. Keep Python requests self-contained and state any material
input/output representation, direction, version, quantity or constraint. Use
ordinary wording rather than imitating a questionnaire template.

Declare reviewer identity or a stable pseudonym, whether the author/labeler is
human or model-assisted, and what policy/code/results the reviewer has seen.
Do not claim independence or blinding if those conditions do not hold. A model
review is model-assisted evidence, never independent human ground truth.

Relation is directional: compare **new question relative to cached question**.

| Label | Rule |
| --- | --- |
| same_intent | Same operation, direction and material requirements; only wording changes. Do not infer missing context. |
| new_more_specific | The new request retains every material cached requirement and adds at least one; a strict subset of acceptable solutions remains. |
| new_more_general | The new request removes a material requirement and permits a strict superset of solutions. |
| different_intent | Material operation, direction or requirements differ and neither request is a strict specialization of the other. |
| null | Ambiguous, missing antecedent/context, conflicting requirements, uncertain interpretation or unresolved label disagreement. Explain why. |

Input-domain restrictions can be comparable: do not automatically call every
input-kind change `different_intent`. Distinguish a narrower domain from an
incompatible representation. A numeric or version change may be incompatible,
a restriction or unclear; judge its meaning, not token inequality. If materiality
is uncertain, leave null. Exact copies do not test semantic paraphrase recognition.

These are QUESTION relation labels, not judgments that a stored answer is
correct, sufficient, current or supported. Answer quality requires its own
evidence and rubric. A safe cache must also enforce identical trusted scope.

Suggested families are prompts for diversity, not labels. A direction or entity
change might form a positive pair if the wording actually preserves meaning.
Write a rationale identifying what is preserved or changed. Ideally a second
reviewer labels independently; keep both labels, disagreements and adjudication
provenance. Never force a null case into a definite class to increase sample size.

Return a separately named completed JSONL without running the policy yourself.
Do not overwrite the frozen blank request or generated train/dev/calibration
files; completed reviews belong in a new versioned natural-review directory. The implementation
and thresholds must remain frozen for its first diagnostic evaluation; do not
tune on it and then call it fresh. Once used for development it is spent.

This small packet is a natural-language diagnostic. It cannot by itself satisfy
the earlier accepted-group final safety gate or establish 98% precision. It is
not the locked Stage A final split, which remains untouched and unevaluated.
