PYTHON := python3
PKGS := ariadne ariadne_mac

.PHONY: setup dev lint typecheck test examples bench format

setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e .[viz]

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy $(PKGS)

test:
	pytest -q

examples:
	$(PYTHON) examples/01_qasm3_roundtrip.py
	$(PYTHON) examples/02_mitigation_autopilot.py
	$(PYTHON) examples/03_router_showdown.py
	$(PYTHON) examples/04_qualtran_to_resources.py || true

.PHONY: examples-mac
examples-mac:
	$(PYTHON) examples/01_sv_limits.py
	$(PYTHON) examples/02_stim_qec.py
	$(PYTHON) examples/03_tn_qaoa.py
	$(PYTHON) examples/04_qasm3_qcec.py
	$(PYTHON) examples/05_qualtran_resources.py || true

.PHONY: tune-router summarize
tune-router:
	ariadne-mac tune-router

summarize:
	ariadne-mac summarize

bench:
	@echo "Benchmarks TBD"
