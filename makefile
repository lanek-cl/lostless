.PHONY: lint format check clean security-check dead-code

lint:
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r .
	isort .
	ruff check . --fix
	black .

format:
	ruff check .  --fix
	black .

check:
	isort --check-only .
	ruff check . 
	black --check .
	flake8 .

clean:
	isort .
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r .

security-check:
	bandit -r .

dead-code:
	vulture .
