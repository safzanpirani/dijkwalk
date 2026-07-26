# PAPERCUTS

Small, non-blocking frictions encountered by agents while working. Review this file periodically and sand them down.

## 2026-07-26T00:35:47.743Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/dijktrans-pufferlib`
- **Tags:** `docs`

Playwright skill requires screenshots under output/playwright but the wrapper does not create that directory, so the first screenshot fails with ENOENT. The workflow docs should include mkdir -p output/playwright before capture.

