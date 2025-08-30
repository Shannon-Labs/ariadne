# Repository Guidelines

## Project Structure & Module Organization
- `ariadne/`: Core Python package and CLI (`ariadne`).
- `ariadne_mac/`: macOS/Apple‑silicon optimized flows and CLI (`ariadne-mac`).
- `tests/`, `tests_mac/`: Pytest suites (default test paths).
- `examples/`: Runnable usage demos (see Make targets below).
- `configs/`: Configuration files for examples and experiments.
- `docs/`, `reports/`, `bench/`: Documentation, generated reports, and benchmarks.

## Build, Test, and Development Commands
- `make setup`: Install/upgrades `pip`, then install project in editable mode.
- `make dev`: Editable install with visualization extras (`[viz]`).
- `make lint` / `make format`: Lint and auto‑format via Ruff.
- `make typecheck`: Static type checks with MyPy.
- `make test`: Run the full pytest suite (quiet mode).
- `make examples` / `make examples-mac`: Run curated examples.
- CLI: `ariadne --help`, `ariadne-mac tune-router`, `ariadne-mac summarize`.

## Coding Style & Naming Conventions
- Python ≥ 3.11. Format and lint with Ruff (line length 100).
- Type hints required; MyPy runs in strict mode for packages.
- Naming: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE`.
- Keep public APIs typed and documented; prefer small, composable functions.

## Testing Guidelines
- Framework: Pytest. Default paths: `tests`, `tests_mac` (configured in `pyproject.toml`).
- Markers: `@pytest.mark.slow`, `@pytest.mark.azure` for long/infra tests.
- Run: `make test` or `pytest -q`. Selective: `pytest tests/test_simple.py -k pattern`.
- Include tests with new features and bug fixes; prefer fast unit tests.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject (≤72 chars), meaningful body when needed.
- Before opening a PR, ensure `make lint typecheck test` passes locally.
- PRs must include: clear description, linked issues, rationale, and test coverage.
- If changing CLIs or outputs, include example commands and snippets/screenshots.
- Update docs/examples when behavior changes; keep `examples/` runnable.

## Security & Configuration Tips
- Do not commit secrets. Use environment variables or local config files in `configs/`.
- Azure‑dependent tests are marked `azure`; run them only with a configured workspace.
