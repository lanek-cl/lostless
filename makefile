.PHONY: lint format check

lint:
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r .
	isort .
	black .

format:
	isort .
	black .

check:
	isort --check-only .
	black --check .
	flake8 .