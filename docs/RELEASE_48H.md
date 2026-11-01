# Group 4 release status — 2026-11-02

Groups 1–3 are implemented with offline fixture evidence. Group 4 is partial.
Resume component metrics and limitations: `Resume_Ready_Project.md`.

## New completed installation check

A fresh Python virtual environment installed the minimal web runtime plus build
and test tools. The wheel was built in isolation, installed with `--no-deps`, and
loaded from that environment's site-packages in a subprocess outside the source
tree. Packaged HTML/JS/CSS served successfully. Real loopback HTTP verified:
three scripted calls for a source task, zero for an approved exact repeat,
fresh approval required, scoped memory listing and clearing. This is an offline
reference fixture, not live provider utility or semantic-answer validation.

174 focused software tests passed in 3.53s in this environment. One warning:
Starlette deprecated its current httpx TestClient adapter. Full ML dependency
installation and a full-package pip dependency check were not performed: this
checks the intentionally minimal gateway profile.

Receipts, build log, wheel SHA-256 and installed dependency versions:
`artifacts/documentation_rebuild/group4/clean-install-v1/`.

## Reproduce

From the repository root, using a new environment and new receipt path:

```bash
python -m venv /tmp/group4-release-env
/tmp/group4-release-env/bin/python -m pip install -r deploy/requirements-gateway.txt pytest==9.1.1 httpx==0.28.1 build
/tmp/group4-release-env/bin/python -m build --wheel --outdir /tmp/group4-wheel
/tmp/group4-release-env/bin/python -m pip install --no-deps /tmp/group4-wheel/adaptive_agentic_semantic_cache-0.1.0-py3-none-any.whl
/tmp/group4-release-env/bin/python scripts/check_document_install.py --output NEW_HTTP_RECEIPT.json
/tmp/group4-release-env/bin/python -m pytest -q tests/test_workbench.py tests/test_document_agent.py tests/test_before_agent_cache.py tests/test_document_orchestration_memory.py tests/test_durable_semantic_rebuild.py tests/test_response_cache.py tests/test_sqlite_response_cache.py
```

The installation smoke refuses to overwrite its receipt and verifies installed
module provenance. Build-only artifacts stay outside Git; hashes are recorded.

## Remaining failures and unknowns

- Browser visual/responsive QA has not passed. The previous Chromium installation
  attempt returned an invalid ZIP archive. The Playwright flow is prepared.
- Public deployment and the manual GitHub workflow are not verified.
- Approved MiniLM/Quora restoration and exposed diagnostic already ran. Dense
  similarity recommended an insufficient nested-conversion answer; the fixed
  pair-gated policy accepted none. General automatic semantic serving remains
  disabled/unqualified. Do not tune on these spent human-reviewed questions.
- Human review covers model-authored items, not independent human-authored data.
- Live costs, useful semantic savings and end-to-end LLM latency remain unmeasured.
- No final evaluation, new training, provider calls or serving promotion occurred.

## Publication

The user requested resuming GitHub publication, but automatic approval review
rejected the push as insufficiently explicit about recorded review data after
an earlier pause. No push or alternative transport followed that rejection.
Destination remains paridhik11/adaptive-agentic-semantic-cache, branch
integration/documentation-rebuild. Main is unchanged; no force or merge.
