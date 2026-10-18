# Rules for this fork

- Upstream baseline: tag `upstream-838bea5`. `master` mirrors upstream, work happens on `main`.
- Never change model behaviour without a passing parity test (phd_work/parity.py, once it exists).
- Edits to upstream files: minimal, one logical change per commit, prefix [port]/[fix]/[cfg].
- All new code goes into `phd_work/` (prefix [phd]). Do not refactor upstream code.
- Log every encountered error and its fix in phd_work/port-log.md.
- Python env: uv, Python 3.12, versions pinned in uv.lock.
