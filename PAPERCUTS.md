# PAPERCUTS

Small, non-blocking frictions encountered by agents while working. Review this file periodically and sand them down.

## 2026-07-26T00:35:47.743Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `docs`

Playwright skill requires screenshots under output/playwright but the wrapper does not create that directory, so the first screenshot fails with ENOENT. The workflow docs should include mkdir -p output/playwright before capture.

## 2026-07-26T00:52:59.334Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `tooling`

While locating Q1 RL orchestration identifiers in dijktrans-pufferlib, the fff MCP multi_grep call failed immediately with 'Transport closed'; fell back to local rg.

## 2026-07-26T04:49:35.982Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `tooling`

While deploying Q1 sources to server-pc, macOS rsync rejected the GNU-only --info=stats2 option; no files copied, switched to portable --stats.

## 2026-07-26T04:53:09.520Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `infra`

During Q1 server-pc setup, uv used its default ~/.cache/uv despite the deployment being scoped to ~/q1-runs; caught and killed the job, then set UV_CACHE_DIR explicitly inside scratch. External cache entries cannot be removed under the approved scope.

## 2026-07-26T04:58:14.662Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `tooling`

While checking the upstream PufferLib libc10 loader issue, the GitHub codebase search failed at repo import due to a Vercel security checkpoint; continued with installed binary inspection and runtime tests.

## 2026-07-26T05:04:10.191Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `tooling`

For a detached Q1 pytest gate, fleet spawn forwarded its documented --cwd option into the remote pytest argv, causing 'unrecognized arguments: --cwd'; retried with explicit cd inside the command.

## 2026-07-26T05:06:53.863Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `tooling`

During remote Q1 suite preflight, scripts/q1_suite.py lacked the assumed --verify-only/--suite-dir CLI; command failed before reading files, switched to authoritative manifest SHA-256 comparison.

## 2026-07-26T05:08:22.299Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `infra`

The first two full Q1 remote jobs failed before Python because CachyOS server-pc has no /usr/bin/time; logs captured, switched the one allowed retry to bash epoch timestamps.

## 2026-07-26T06:05:20.834Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `fleet-cli`

While checking detached Q1 jobs on server-pc, fleet jobs list and fleet jobs --help both treated list/--help as host selectors; the actual syntax is fleet jobs [<sel>] and bare fleet prints usage.

## 2026-07-26T06:18:46.715Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `python-venv`

In dijktrans-pufferlib's rebuilt uv venv, .venv/bin/pytest tests/ -q cannot import the repo package, while the documented .venv/bin/python -m pytest tests/ -q correctly places cwd on sys.path and passes all 16 tests.

## 2026-07-26T07:29:28.963Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `fleet-cli`

A continuous local loop around fleet jobs wait worked through the first 55-second timeout, then fleet briefly returned failed to inspect job and the loop spun without backoff; had to Ctrl-C the local poll while detached remote jobs remained healthy.

## 2026-07-26T07:30:52.078Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `fleet-cli`

After repeated fleet jobs wait inspection failures on server-pc, fleet jobs reported two Q1 jobs as dead even though each outer run wrapper, command shell, and Python child remained alive in ps --forest; tracker status was false-negative while jobs continued normally.

## 2026-07-26T11:58:36.166Z — codex — gpt-5

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `validation`

While validating Q1 supervised artifacts, I assumed captures was a step-keyed dict, but run_q1_supervised emits it as a list of capture records. A small documented schema or validation helper would prevent ad-hoc validator failures.

## 2026-07-26T15:34:07.289Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `missing-tool`

The Q1 verification command assumed Ruff lived at .venv/bin/ruff, but this project venv does not install the linter even though a global Ruff is available. Document the lint runner or add the dev dependency so the verification command is reproducible.

## 2026-07-26T17:03:45.519Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `import-path`

Q1 aggregation fails when invoked as `.venv/bin/python scripts/merge_q1_evaluations.py` because the script imports `scripts.evaluate_q1_rl` while Python puts only scripts/ on sys.path. Document/use `python -m scripts.merge_q1_evaluations` or make scripts a proper package-safe entrypoint.

## 2026-07-26T17:40:17.608Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib/paper`
- **Tags:** `misleading-error`

While rendering an 8-page PDF for visual QA, pdftoppm emitted unpadded page-1.png names but the contact-sheet command assumed page-01.png. Check actual output naming or glob numerically before montage.

## 2026-07-26T17:40:32.080Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib/paper`
- **Tags:** `misleading-error`

ImageMagick montage failed with 'unable to read font' even though the contact sheet requested no labels. Use resize plus +append/-append composition to avoid montage's implicit font dependency.

