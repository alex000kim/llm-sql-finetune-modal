# LLM Fine-tuning with Modal

This project demonstrates how to fine-tune a large language model (LLM) for SQL query generation using Modal, a cloud platform for running distributed Python applications.

## Overview
The application fine-tunes the Microsoft Phi-3-mini-4k-instruct model on a SQL dataset to improve its ability to generate SQL queries from natural language questions. It uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters for efficient training.

## Features
- Dataset preprocessing and caching
- Distributed training on GPU
- Model evaluation
- Easy deployment and scaling with Modal
## Prerequisites

- A Modal account (https://modal.com)
- Python 3.10 or later
- The `modal` Python package installed
- 
## Setup
- Install the Modal CLI:
`pip install modal`
- Set up Modal token:
`modal seup`
- Create a secret named `my-huggingface-secret` in your Modal workspace with your Hugging Face API token (`HF_TOKEN`).

## Usage
To run the entire pipeline:
`modal run llm-sql-finetune-modal.py`
This will:
1. Set up the environment and preprocess the dataset
2. ine-tune the model
3. Evaluate the fine-tuned model
4. 
## Code Structure
- `setup_and_preprocess()`: Prepares the dataset and caches it in a Modal Volume
- `train_model()`: Fine-tunes the model using PEFT and LoRA
- `evaluate_model()`: Evaluates the fine-tuned model on a test set
- `main()`: Orchestrates the entire pipeline

