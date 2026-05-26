.PHONY: test lint format format-check typecheck build

all: test lint format typecheck

test:
	$(info ****************** running tests ******************)
	uv run --all-groups --all-extras pytest tests

lint:
	$(info ****************** running pre-commit ******************)
	uv run --all-groups --all-extras prek run -a

format:
	$(info ****************** formatting ******************)
	uv run --all-groups --all-extras ruff format

format-check:
	$(info ****************** checking formatting ******************)
	uv run --all-groups --all-extras ruff format --check

typecheck:
	$(info ****************** type checking ******************)
	uv run --all-groups --all-extras ty check src/

build:
	$(info ****************** building ******************)
	uv build
