USER_NAME = $(shell whoami)

recreate-env: environment.yml
	@echo "Recreating conda environment..."; \
	conda env create -f environment.yml; \
	conda activate graph_sm; \
	echo "Initializing DVC..."; \
	dvc init; \
	echo "Full initialization complete!"

create-user-notebook:
	@if [ ! -d "Notebooks" ]; then \
		mkdir Notebooks; \
	fi
	@if [ ! -f "Notebooks/$(USER_NAME).ipynb" ]; then \
		echo "Creating user notebook for $(USER_NAME)..."; \
		touch Notebooks/$(USER_NAME).ipynb; \
	else \
		echo "User notebook for $(USER_NAME) already exists."; \
	fi

setup-dvc-preprocess-pipeline:
	@echo "Setting up DVC preprocess pipeline..."; \
	dvc stage add -n instafake-preprocess \
	  -d Src/preprocess_instafake.py \
	  -d Dataset/InstaFake/raw/faked-v1.0/fakeAccountData.json \
	  -d Dataset/InstaFake/raw/faked-v1.0/realAccountData.json \
	  -o Dataset/InstaFake/interim/combined_account_data.csv \
	  python Src/preprocess_instafake.py; \
	echo "DVC preprocess pipeline setup complete!"

run-preprocess-instafake:
	@echo "Running preprocessing script..."; \
	dvc repro instafake-preprocess; \

# This target assumes that the training script is named `train_model.py` and is located in the `Src` directory.
train:
	@echo "Training model..."; \
	python Src/train_model.py; \
	echo "Model training complete!"



