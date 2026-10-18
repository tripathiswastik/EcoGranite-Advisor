## Summary of Changes
<!-- Brief bullet points explaining what changes were made and why -->
- 

## Security & Privacy Checklist (RUFlow / Antigravity Rules)
- [ ] **Secret & Key Hygiene**: No raw API keys, private tokens, or credentials committed.
- [ ] **Data Isolation**: No proprietary PII or production customer data included; synthetic fixtures used.
- [ ] **Least Privilege Tool Use**: Verified typed parameter invocation; no arbitrary shell injection vectors.
- [ ] **Prompt Guard & Output Filtering**: External ingestion screening and Colang/guardrail compliance verified (if LLM-facing).

## Verification Evidence
<!-- Provide test run results, coverage diffs, and sandbox verification logs -->
- [ ] Unit tests passed locally (Flakiness Filter: 5 consecutive clean runs)
- [ ] AST / Linting check clean (`ruff` / `black` / `eslint`)
- [ ] Test coverage delta: +0.0% or greater

```bash
# Verification command executed
```

## Breaking Changes Warning
- [ ] **No breaking changes**: Backwards-compatible modification.
- [ ] **Breaking changes**: Explicit flag for API, data schema, or infrastructure contract alteration (requires HITL sign-off).
