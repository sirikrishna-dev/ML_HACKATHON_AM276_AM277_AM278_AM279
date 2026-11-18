PYTHON ?= python

.PHONY: eda candidates train tune decode api ui test

eda:
	$(PYTHON) -m notebooks.01_eda

candidates:
	$(PYTHON) scripts/build_candidates.py --split train

train:
	$(PYTHON) scripts/train_models.py

tune:
	$(PYTHON) scripts/tune.py

decode:
	$(PYTHON) scripts/decode_eval.py

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/app.py

test:
	$(PYTHON) -m pytest -q
