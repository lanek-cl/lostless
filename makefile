.PHONY: lint

lint:
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r .
	isort .
	black .