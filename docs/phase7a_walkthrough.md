# Phase 7a — Secret Scan & Key Hygiene Walkthrough

**Date**: 2026-10-01  
**Branch**: main (no branch created, no source logic modified)  
**Commit at time of scan**: `57f4060` (HEAD, Phase 6 blind evaluation results)

---

## 1 — Toolchain Setup

Neither `gitleaks` nor `trufflehog` was pre-installed. Gitleaks was downloaded directly:

```powershell
# Download
Invoke-WebRequest `
  -Uri "https://github.com/gitleaks/gitleaks/releases/download/v8.18.4/gitleaks_8.18.4_windows_x64.zip" `
  -OutFile "$env:TEMP\gitleaks.zip" -UseBasicParsing

# Extract
Expand-Archive -Path "$env:TEMP\gitleaks.zip" `
               -DestinationPath "$env:TEMP\gitleaks_bin" -Force
```

**Tool version**: gitleaks v8.18.4 (windows/x64)

---

## 2 — Task 1: Full Git-History Scan

### Command run

```powershell
cd "c:\Users\parid\Downloads\Agentic AI\adaptive-agentic-semantic-cache"

& "$env:TEMP\gitleaks_bin\gitleaks.exe" detect `
    --source . `
    --log-opts "--all" `
    --report-path "docs/phase7a_secret_scan_raw.txt" `
    --report-format json `
    --no-banner
```

### Raw output (stdout/stderr)

```
1:56AM INF 23 commits scanned.
1:56AM INF scan completed in 597ms
1:56AM INF no leaks found
```

### JSON report

```json
[]
```

**Exit code: 0** — no secrets detected in any of the 24 commits.

### Cross-checks performed

```powershell
# Was .env ever tracked by git?
git log --all --full-history -- ".env"
# Result: (empty — never committed)

# Is the actual key anywhere in history?
git log --all -p | Select-String "26f85cb82471fbe0beda9727..."
# Result: zero matches — actual secret NEVER committed

# Is the key prefix (sk-or-v1-) anywhere in history?
git log --all -p | Select-String "sk-or-v1-"
# Result: exactly 1 match — the placeholder help string in run_openrouter_eval.py line 47:
#   print("Please add OPENROUTER_API_KEY=sk-or-v1-... to .env or your environment.")
# This is a help-text comment, not a credential.
```

**Verdict: NO secrets in git history.**

---

## 3 — Task 2: Working-Tree Grep

All patterns were searched in `*.py *.md *.json *.txt`, excluding `.venv/` and `.git/`.

### 3.1 Pattern: `sk-`

| File | Lines | Nature |
|------|-------|--------|
| `data/raw/phase6_blind_eval_dataset.json` | 388, 1717 | `"disk-space"` — false positive |
| `data/phase6_telemetry.json` | 1013, 4586 | `"disk-space"` — false positive |
| `scripts/build_phase6_sealed_dataset.py` | 185 | `["disk-space", "df"]` — false positive |
| `scripts/run_openrouter_eval.py` | 47 | Help-text placeholder `sk-or-v1-...` — not a real key |

**All hits are false positives. No real keys.**

### 3.2 Pattern: `OPENROUTER`

Matches in: `scripts/run_openrouter_eval.py`, `scripts/run_phase4_calibration.py`,
`src/decision/decision_step.py`, `src/decision/judge_call.py`,
`src/evaluation/decision_evaluator.py`, `tests/test_*.py`

All hits reference the **variable name** `OPENROUTER_API_KEY` in code or tests — no literal key values.

### 3.3 Pattern: `api_key` (case-insensitive)

Matches across `src/decision/judge_call.py`, `tests/`, `scripts/`, `docs/phase3_judge_call_walkthrough.md:100`.

All are Python parameter names, docstring references, or mock argument names. No literal values.

### 3.4 Pattern: `Bearer `

| File | Line | Nature |
|------|------|--------|
| `scratch/check_all_generations.py` | 27 | `f"Bearer {API_KEY}"` — uses env var |
| `scripts/run_load_test.py` | 79 | `f"Bearer {api_key}"` — uses env var |
| `scripts/run_new_dataset_load_test.py` | 207 | `f"Bearer {api_key}"` — uses env var |

All construct the header dynamically from an env var. No hardcoded tokens.

### 3.5 Pattern: `.env`

Matches in: `scratch/check_all_generations.py:15`, `scripts/label_synthetic_feedback.py:35,37`,
`scripts/run_openrouter_eval.py:28,47`, `src/decision/judge_call.py:197,202,206,210,212,213,273,274,281,283`,
`README.md:145`.

All are references to the `.env` **filename** in comments, docstrings, or instructions. No secret values.

### 3.6 High-Entropy (30+ char alphanumeric) Strings

| File | Lines | Nature |
|------|-------|--------|
| `data/raw/phase6_blind_eval_dataset.json` | 71, 512, 527, 1437, 1667 | Stack Overflow / Math.SE URLs |
| `docs/phase2_reproducibility_fix.md` | 23, 34, 48, 156, 166, 258, 264 | Git commit hash `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `docs/phase2_walkthrough.md` | 124 | URL fragment |
| `docs/phase3_final_decision.md` | 88 | URL |
| `docs/phase3_implementation.md` | 11 | URL |
| `src/cache/embedding.py` | 25, 63 | HuggingFace model name/path |
| `tests/test_model_integration.py` | 8, 156 | HuggingFace model name/path |

**All hits are URLs, commit hashes, or model identifiers. No credential material.**

---

## 4 — Task 3: .gitignore Audit

### Coverage confirmed ✅

| Pattern | Covered by .gitignore | Line |
|---------|-----------------------|------|
| `.env` | ✅ `.env` exact match | 38 |
| `.env.*` | ✅ `.env.*` wildcard (excepts `.env.example`) | 39–40 |
| `*.key` | ✅ Added in this phase | new |
| `*.pem` | ✅ Added in this phase | new |
| `*.token` | ✅ Added in this phase | new |
| HAR/request cache | ✅ `*.har`, `requests_cache/`, `http_cache/` added | new |
| Log files | ✅ `*.log`, `logs/` added | new |
| Python cache | ✅ `__pycache__/`, `.pytest_cache/` | existing |

### Key-loading logic audit

The runtime reads the key **exclusively from the environment**, never from hardcoded values:

- **`src/decision/judge_call.py` → `_resolve_api_key()`** (lines 196–233):
  - Checks `os.environ.get("OPENROUTER_API_KEY")` first.
  - Falls back to reading `.env` file on disk (this file is gitignored).
  - Returns `None` if not found.
- **Key-missing behaviour** (lines 302, 395, 403, 627, 635):
  - `if not self.api_key: return None` (client property returns `None`)
  - Judge falls back to `BYPASS` with rationale: `"Judge unavailable: missing OpenRouter API key or client; safe fallback to BYPASS"`
- **`scripts/run_openrouter_eval.py`** (lines 45–48):
  - Explicit guard: `if not judge.api_key: print("ERROR: OPENROUTER_API_KEY is not set."); sys.exit(1)`

**Verdict: Code correctly reads key from environment, fails with clear error if unset. ✅**

---

## 5 — Task 4: .env.example

File: [`.env.example`](file:///c:/Users/parid/Downloads/Agentic%20AI/adaptive-agentic-semantic-cache/.env.example)

Updated with:
- Placeholder values only (`sk-or-v1-REPLACE_WITH_YOUR_OPENROUTER_KEY`, etc.)
- Format hints for each key (where to get it, expected format)
- Comments explaining the fail-closed fallback behaviour

---

## 6 — Files Changed in This Phase

| File | Action |
|------|--------|
| `docs/phase7a_secret_scan_raw.txt` | **Created** — raw gitleaks output + metadata |
| `docs/phase7a_walkthrough.md` | **Created** — this document |
| `.gitignore` | **Updated** — added `*.key`, `*.pem`, `*.p12`, `*.pfx`, `*.token`, `*.secret`, `*.har`, `*.log`, `logs/`, `telemetry_local/`, `cache_dump/`, `requests_cache/`, `http_cache/` |
| `.env.example` | **Updated** — placeholder values, format hints, source instructions |

---

## 7 — Reproducibility

To re-run the scan yourself:

```powershell
# 1. Download gitleaks (one-time)
Invoke-WebRequest `
  -Uri "https://github.com/gitleaks/gitleaks/releases/download/v8.18.4/gitleaks_8.18.4_windows_x64.zip" `
  -OutFile "$env:TEMP\gitleaks.zip" -UseBasicParsing
Expand-Archive "$env:TEMP\gitleaks.zip" "$env:TEMP\gitleaks_bin" -Force

# 2. Run full-history scan
cd "c:\Users\parid\Downloads\Agentic AI\adaptive-agentic-semantic-cache"
& "$env:TEMP\gitleaks_bin\gitleaks.exe" detect `
    --source . `
    --log-opts "--all" `
    --report-path "docs/phase7a_secret_scan_raw.txt" `
    --report-format json `
    --no-banner

# 3. Confirm .env was never committed
git log --all --full-history -- ".env"

# 4. Re-run working-tree grep
$exts = "*.py","*.md","*.json","*.txt"
foreach ($pat in @("sk-","OPENROUTER","api_key","Bearer ",".env")) {
    Write-Host "=== $pat ===" -ForegroundColor Cyan
    Get-ChildItem -Recurse -Include $exts |
        Where-Object { $_.FullName -notmatch '\\.venv\\' -and $_.FullName -notmatch '\\.git\\' } |
        Select-String -Pattern $pat |
        Select-Object Path, LineNumber |
        Format-Table -AutoSize
}
```

### Scan Commit SHA

```
HEAD at time of scan: 57f4060  (Phase 6: Blind evaluation results and walkthrough)
```

---

## 8 — Final Verdict

| Check | Result |
|-------|--------|
| Gitleaks history scan (24 commits, all branches) | ✅ No findings |
| `.env` ever committed to git | ✅ Never |
| Actual key value in any commit | ✅ Not present |
| Working-tree `sk-` hits | ✅ All false positives (disk-space, help placeholder) |
| Working-tree `Bearer ` hits | ✅ All use env-var interpolation, no literals |
| Working-tree high-entropy hits | ✅ URLs, commit hashes, model names only |
| `.gitignore` covers `.env`, `*.key`, telemetry | ✅ Confirmed / extended |
| Code reads key from env var, fails cleanly if unset | ✅ Confirmed |
| `.env.example` has placeholder values only | ✅ Updated |

> **⚠️ ACTION REQUIRED FOR YOU**: Your `.env` file on disk contains a real `OPENROUTER_API_KEY`. While it was never committed (safe from git exposure), you should rotate/revoke this key at https://openrouter.ai/keys as a precaution, since the key was visible on your local filesystem and this session read it. This is standard practice after a security audit.
