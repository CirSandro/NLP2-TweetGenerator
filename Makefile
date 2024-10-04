# Directorie
DATA_DIR = data

# Commands
.PHONY: all setup install download_data

all: setup install download_data

# Install dependencies
install: 
	@echo "Installing dependencies..."
	pip install streamlit
	pip install gdown
	pip install 'transformers>=4.11'
	pip install accelerate

# Download data if it doesn't exist
download_data:
	@echo "Checking for data directory..."
	@if [ ! -d "$(DATA_DIR)" ]; then \
		mkdir $(DATA_DIR); \
	fi
	@echo "Downloading dataset..."
	gdown --id 1Ue-2rahSQhDXrQweXz5WVe73CIvdcddR -O $(DATA_DIR)/tweet_train.csv

