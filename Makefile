# POSIX make. On Windows (no make), run the underlying commands directly:
#   python -m unittest discover -s tests -t .
#   python -m qedrf run | audit | bench --out-dir paper
PY ?= python3

.PHONY: help test run audit bench paper clean

help:
	@echo "make test    - substrate-free test suite (skips torch/substrate tests cleanly)"
	@echo "make run     - full pipeline: train + audit + diagnostics (needs torch + substrate on cache miss)"
	@echo "make audit   - quick Markdown summary from cached runs"
	@echo "make bench   - regenerate tables/macros/figures from cached runs -> paper/"
	@echo "make paper   - bench + pdflatex -> paper/main.pdf"
	@echo "make clean   - remove __pycache__ and LaTeX build junk"

test:
	PYTHONPATH=. $(PY) -m unittest discover -s tests -t .

run:
	PYTHONPATH=. $(PY) -m qedrf run

audit:
	PYTHONPATH=. $(PY) -m qedrf audit

bench:
	PYTHONPATH=. $(PY) -m qedrf bench --out-dir paper

paper: bench
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null && echo "built paper/main.pdf"

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.synctex.gz
