# Security Policy

This repository follows a security-first workflow. The goal is to prevent secret leakage, reduce vulnerable code, and handle incidents quickly and safely.

Note: Security scanners and CI wiring are introduced in dedicated tasks (CI in Task 13, scanners in Task 14). This document sets expectations now and will be enforced automatically once those tasks land.

## Reporting a Vulnerability or Security Concern

- Preferred contact: security@bayes-group.example (placeholder)
- Alternative: open a private security advisory or contact the internal "#security" channel (placeholder)
- Target response: within 2 business days

Please include impacted files, commit SHAs, reproduction steps, and any logs that do not contain sensitive data.

## Never Commit Secrets

- Do not commit API keys, tokens, passwords, private keys, cloud credentials, or dataset access URLs.
- Use environment variables and secret managers; keep `.env` and similar files out of version control.
- Treat logs, stack traces, and screenshots as potentially sensitive.

### If a Secret Leaks

1. Revoke/rotate the credential at the provider immediately.
2. Remove the secret from code and history (e.g., `git filter-repo` or BFG). Do not rely on a revert alone.
3. Add replacement via environment variables or secret manager; never reintroduce secrets into VCS.
4. Notify security owners at security@bayes-group.example with the commit SHA and scope.
5. Review access logs for misuse and take follow-up actions (e.g., forced token/session invalidation).

## Security Tooling (to be configured)

These tools will run locally via pre-commit and in CI once Tasks 13–14 are completed:

- gitleaks: secret scanning.
- semgrep: static application security testing (SAST).
- bandit: Python security linter for common issues.
- Dependency audit: checks known CVEs (e.g., pip-audit or safety).

All findings that meet defined severity thresholds will fail CI. Contributors should run the hooks locally before pushing.

## Environments and Least Privilege

- Use a project-local virtual environment: `.venv` (avoid global interpreters).
- Do not run dev or CI jobs as root when avoidable.
- Grant the minimum cloud/service permissions required; prefer short-lived credentials.
- Store per-environment credentials separately; never reuse production secrets in development or tests.

## Data Handling

- Do not commit datasets or any PII/PHI. Keep sensitive data out of logs and error messages.
- Use anonymized or synthetic data for examples and tests.
- If handling regulated data, ensure encryption at rest/in transit and follow applicable policies.

## Dependencies and Updates

- Pin direct dependencies where practical and update regularly to address CVEs.
- Resolve critical/high vulnerabilities promptly; document temporary risk acceptances.

## Developer Checklist (pre-push)

- No secrets in code, configs, or examples.
- Scanners and linters pass locally once configured (pre-commit/CI).
- Logs and outputs do not contain sensitive data.

## Verification

To verify this policy file exists:

```bash
test -f docs/SECURITY.md && echo "OK"
```
