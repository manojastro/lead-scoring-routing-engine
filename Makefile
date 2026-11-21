# File: Makefile
.PHONY: help install data train promote-staging api mlflow docker-up docker-down \
        batch-dryrun batch test lint clean

PYTHON ?= python

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install pinned dependencies
	$(PYTHON) -m pip install -r requirements.txt

data:  ## Generate the seeded synthetic lead dataset
	$(PYTHON) -m training.generate_data --rows 20000 --out data/leads.csv --seed 42

train:  ## Train, log to MLflow, register + transition to Staging
	$(PYTHON) -m training.train --data data/leads.csv \
		--tracking-uri $${MLFLOW_TRACKING_URI:-http://localhost:5000} \
		--registered-model-name $${MODEL_NAME:-lead-quality-classifier}

promote-staging:  ## Promote latest Staging version to Production (V=<version>)
	$(PYTHON) -c "from mlflow.tracking import MlflowClient; import os; \
MlflowClient(os.environ.get('MLFLOW_TRACKING_URI','http://localhost:5000')).transition_model_version_stage(\
name=os.environ.get('MODEL_NAME','lead-quality-classifier'), version=os.environ['V'], stage='Production')"

api:  ## Run the FastAPI scoring service locally
	$(PYTHON) -m uvicorn serving.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:  ## Start a local MLflow tracking server + UI on :5000
	$(PYTHON) -m mlflow server --backend-store-uri ./mlruns-store \
		--artifacts-destination ./mlartifacts --host 0.0.0.0 --port 5000

docker-up:  ## Build and start api + mlflow services
	docker compose up --build -d && docker compose ps

docker-down:  ## Stop the local stack
	docker compose down

batch-dryrun:  ## Preview unscored-lead backfill without writing (LIMIT=100)
	$(PYTHON) -m batch.score_unscored --dry-run --limit $${LIMIT:-100} --chunk-size 50

batch:  ## Execute resumable backfill scoring loop (LIMIT=5000 CHUNK=500)
	$(PYTHON) -m batch.score_unscored --limit $${LIMIT:-5000} --chunk-size $${CHUNK:-500}

test:  ## Run unit tests (Salesforce mocked)
	$(PYTHON) -m pytest tests/ -v

lint:  ## Basic syntax sanity across packages
	$(PYTHON) -m compileall -q training serving batch tests

clean:  ## Remove caches and generated artifacts
	rm -rf .pytest_cache **/__pycache__ mlruns mlruns-store mlartifacts data/leads.csv
