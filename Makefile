PYTHON := "/Users/admin/Handwritten Digit Recognition /.venv/bin/python"

.PHONY: install train evaluate test api demo format lint typecheck

help:
	@echo "Available targets: install train evaluate test api demo format lint typecheck"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

test:
	$(PYTHON) -m pytest tests -q

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

demo:
	$(PYTHON) -m streamlit run app/streamlit_app.py

format:
	$(PYTHON) -m black .

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src api
