.PHONY: install install-full test compile clean

install:
	pip install -e ".[dev]"

install-full:
	pip install -e ".[corpus,ragas,vision,dev]"

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

compile:
	PYTHONPATH=src python -m compileall -q src tests scripts

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
