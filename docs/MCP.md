## MCP usage guide (Cursor)

This repository supports using Model Context Protocol (MCP) tools from Cursor (e.g., BrowserMCP, Chrome DevTools MCP) to aid runtime inspection, visual debugging, and safe sandboxed I/O.

### What MCP is and when to use it

- **MCP**: A standardized way for tools to expose capabilities to AI assistants in a controlled manner.
- **Use MCP when**:
  - **Runtime inspection** is needed (observe logs, network calls, DOM state).
  - **Visual debugging** helps (layout/DOM/CSS via DevTools MCP).
  - **Sandboxed I/O** is required (fetch public pages or capture artifacts without broad system access).
- **Avoid MCP** when local static analysis suffices or when it would expose secrets or internal-only endpoints.

### Setup in Cursor (minimal permissions)

1. Open Cursor > Settings > MCP.
2. Enable only the MCPs you need (e.g., BrowserMCP, Chrome DevTools MCP).
3. Grant the **least privileges**:
   - Restrict domains to public targets when browsing.
   - Disable file write unless necessary; prefer read-only captures.
   - Avoid pasting tokens, cookies, or credentials.
4. Confirm logs/artifacts do not include secrets before saving or sharing.

Link back to the agent execution loop in `AGENTS.md` for step-by-step workflow alignment.

### Safe usage guidelines

- **No secrets**: Never include API keys, cookies, private URLs, or proprietary data in MCP inputs, logs, or artifacts.
- **Scope approvals**: When prompted, approve only necessary actions, limited to the smallest scope and time.
- **Sanitize outputs**: Before attaching artifacts, redact PII and sensitive headers (e.g., Authorization, Set-Cookie).
- **Reproducibility**: Prefer deterministic capture steps and document versions and URLs.
- **Local mirroring**: Where possible, replicate issues with local logs or test fixtures rather than remote targets.

### Attaching MCP artifacts/logs to PRs

- Store artifacts under `docs/artifacts/mcp/` or `logs/mcp/`:
  - Screenshots: `docs/artifacts/mcp/<ticket-id>/screenshot.png`
  - HAR/network logs: `logs/mcp/<ticket-id>/network.har`
  - HTML snapshots: `docs/artifacts/mcp/<ticket-id>/snapshot.html`
- Add a short README in the artifact folder describing:
  - Tool used and version, URL or target, timestamp, steps to reproduce.
  - Any redactions performed.
- Reference artifacts in the PR description with concise bullet points and links.

### Workflow integration

- Follow `AGENTS.md` execution loop (venv, deps, format, tests) and use MCP only where it adds value.
- For functional checks, prefer local runs (e.g., `python ml_core/train.py debug=fdr`) and attach MCP captures only when they provide additional insight.

### Quick verification

```bash
cd /Users/Artem.Makoian/work/bayes_group/lightning-hydra-template
test -f docs/MCP.md && echo "OK"
```
