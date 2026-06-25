---
name: finalize-pr
description: "Pre-merge finalization checklist for a change. Use when: wrapping up a feature or fix before opening or updating a PR, preparing to hand work back to the user for commit, or when /finalize-pr is invoked. Bumps the package version when warranted, refreshes README benchmark stats and prose to match behavior changes, runs an editor-lens pass over the diff to trim comment debt, runs the full test suite and flake8, and surfaces a final summary. Does NOT commit, push, or open PRs — those remain the user's responsibility."
argument-hint: "Optionally name the change being finalized (e.g. 'substep solver') and any version bump intent (patch/minor/major)"
---

# Finalize PR

Run the repeatable housekeeping that should happen before a change is handed
back to the user for commit: version bump, README/docs refresh, tests, and
lint. This skill catches the things that are easy to forget when the code
itself is done.

## When to Use

- A feature or fix is functionally complete and you are about to wrap up.
- Behavior, performance, or public API changed in a way the README documents.
- Before telling the user the work is ready to commit.

## When NOT to Use

- Mid-task, when code is still in flux.
- For pure exploration or research with no code change.

## Hard Rules

- **Never commit, push, amend, tag, or open/modify a PR.** All git operations
  are the user's responsibility. This skill prepares the tree; the user lands
  it.
- **Always work inside the project venv** (`source .env314/bin/activate`).
  Never run `pip`, `pytest`, `python`, or `flake8` outside it.
- **Do not invent benchmark numbers.** Every stat written to the README must
  come from a benchmark run performed in this session.

## Procedure

Track progress with a todo list so no step is silently skipped.

### 1. Establish What Changed

Summarize the change set in one or two sentences. Determine:

- Did runtime **behavior** change (physics, solver defaults, scene output)?
- Did **performance** change (the benchmark numbers will move)?
- Did the **public API** or CLI flags change?
- Did any **prose** in the README or `copilot-instructions.md` become stale?

This drives which of the steps below apply. Skip steps that are clearly N/A,
but state that you are skipping them and why.

### 2. Bump the Version (if warranted)

The version lives in [pyproject.toml](pyproject.toml) (`version = "..."`).

- **Patch** — bug fix, internal refactor, no API or behavior change visible to
  callers.
- **Minor** — new feature, new flag, changed defaults that stay backward
  compatible.
- **Major** — breaking API change.

Propose the new version to the user and wait for confirmation before editing,
unless the user already stated the bump intent. If the change is purely
internal scaffolding with no user-visible effect, ask whether a bump is wanted
at all rather than assuming one.

### 3. Refresh the README and Docs

If behavior or performance changed:

1. **Re-run the benchmark** that the README quotes, with the same parameters
   shown in the README (currently `--shapes 80 --frames 300 --runs 5`). Use the
   numbers from this run — never stale or guessed values.
2. **Update the stats table and its summary line** to match.
3. **Update prose, diagrams, and feature bullets** that describe the changed
   behavior (e.g. the per-frame step section and any mermaid diagram).
4. **Grep for stale claims** across `README.md`, `.github/copilot-instructions.md`,
   and benchmark docstrings. A change in one place often leaves a contradicting
   note in another.

Keep the README's existing tone: numbers are a trend, not a contract.

### 4. Run the Full Test Suite

```bash
source .env314/bin/activate
pytest -q
```

- All tests must pass. The golden-master determinism test
  (`test_golden_master_state_is_reproducible`) is the key oracle: if it fails
  after a refactor that was meant to be behavior-preserving, stop and
  investigate — do not update the golden values to make it pass unless the
  behavior change was intentional and approved.
- If new behavior is untested, add tests before finalizing (follow
  `testing-with-boc`).

### 5. Lint

```bash
flake8 src test bench
```

Must exit 0. Fix every finding; do not suppress with `noqa` unless the project
already does so for that rule and it is justified.

### 6. Editor-Lens Pass Over the Diff

Trim comment debt accumulated during the change. Scope this to the **files
touched by this change** (not the whole repo) using the `editor-lens` agent.

1. Build the list of changed in-scope source files (Python only):

   ```bash
   git diff --name-only --diff-filter=d main...HEAD \
     | grep -E '^(src/bocphysics|bench|test|scripts)/.*\.py$'
   ```

2. Invoke `review-loop` with `editor-lens` as the reviewer and that file
   list as the target, applying the Keep / Rewrite / Cut policy in
   [.github/agents/editor-lens.agent.md](../../agents/editor-lens.agent.md).
   Iterate until the lens reports no new findings.

3. Separately run the editor-lens **cryptic-reference sweep** over *every*
   changed text file — including `README.md`, `PLAN.md`, and `docs/**` — to
   catch finding IDs, remediation slugs, and internal plan/review backrefs
   that leaked into user-facing prose. For user-facing files, propose the
   rewrite and get user approval before editing; do not silently rewrite.

The editor-lens full-prose scope explicitly excludes `docs/**`, `README.md`,
`PLAN.md`, and `.github/**` — only the cryptic-reference sweep touches those.
Re-run `flake8` after this pass since collapsing comments can shift lines.

### 7. Final Summary

Report back to the user:

- The version (old → new, or "unchanged" with the reason).
- Which README/doc sections were updated, with the headline stat delta.
- The editor-lens result: counts of cuts / rewrites / kept, multi-line
  comments collapsed, and any cryptic references found in user-facing docs.
- Anything the editor-lens flagged as ambiguous or as a finding (e.g. a
  classical-synchronization rationale) that needs the user's decision.
- Test result (`N passed`) and lint result (`rc=0`).
- Any deferred items or follow-ups noticed but intentionally not done.
- An explicit reminder that the tree is ready for **the user** to commit —
  this skill does not commit.

## Guidelines

- **Move slow to go fast.** Run the steps in order; verify each before moving
  on. A failing test or lint error stops the line.
- **Surface, don't bury.** If you find a stale doc claim or an untested code
  path you cannot fix in scope, list it in the final summary rather than
  silently leaving it.
- **One source of truth for numbers.** The benchmark run in this session is the
  only acceptable source for README stats.
