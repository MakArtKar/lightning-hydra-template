# Security Policy

This repository follows a security-first workflow. The goal is to prevent secret leakage, reduce vulnerable code, and handle incidents quickly and safely.

## Reporting a Vulnerability or Security Concern

- Preferred: Open a private security advisory on GitHub
- Alternative: Contact repository maintainers directly via private channels
- Target response: within 2 business days

Please include impacted files, commit SHAs, reproduction steps, and any logs that do not contain sensitive data.

## Never Commit Secrets

- Do not commit API keys, tokens, passwords, private keys, cloud credentials, or dataset access URLs.
- Use environment variables and secret managers; keep `.env` and similar files out of version control.
- Treat logs, stack traces, and screenshots as potentially sensitive.

### If a Secret Leaks

1. **Revoke immediately**: Rotate the credential at the provider right away.
2. **Remove from history**: Use `git filter-repo` or BFG to purge the secret from git history. A simple revert is insufficient.
3. **Replace securely**: Add the replacement via environment variables or a secret manager; never commit secrets to VCS.
4. **Notify maintainers**: Contact repository maintainers with the commit SHA, affected files, and scope of exposure.
5. **Audit access**: Review provider access logs for misuse and take appropriate follow-up actions (e.g., forced session invalidation).

## Security Tooling

These tools run locally via pre-commit and in CI:

- gitleaks: secret scanning.
- semgrep: static application security testing (SAST).
- bandit: Python security linter for common issues.
- Dependency audit: checks known CVEs (e.g., pip-audit or safety).

All findings that meet defined severity thresholds will fail CI. Contributors should run pre-commit hooks locally before pushing:

```bash
make format  # runs pre-commit on all files
```

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

- [ ] No secrets in code, configs, or examples
- [ ] Pre-commit hooks pass locally (`make format`)
- [ ] Tests pass (`make test`)
- [ ] Logs and outputs do not contain sensitive data

## Quick Verification

Verify security configuration:

```bash
# From repository root
test -f docs/SECURITY.md && echo "Security policy: OK"
test -f gitleaks.toml && echo "Gitleaks config: OK"
test -f bandit.yaml && echo "Bandit config: OK"
test -f .semgrep.yml && echo "Semgrep config: OK"
```
