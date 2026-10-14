
# Project 1: English-to-French Neural Machine Translation

## 1. Project Objective

Implement an English-to-French neural machine translation system using PyTorch's high-level recurrent neural network APIs.

The project focuses on understanding sequential data preparation, encoder–decoder architecture, recurrent neural networks, model training, and autoregressive inference.

Two independent implementations will be developed: a GRU-based encoder–decoder and an LSTM-based encoder–decoder. The purpose is to learn how to implement and use each architecture, not to compare which performs better.

## 2. Learning Objectives

By completing this project, I should be able to:

1. Clean and prepare English–French sentence pairs for sequential modeling.
2. Understand tokenization, vocabulary construction, special tokens, padding, truncation, and sequence lengths.
3. Convert text into numerical sequences and PyTorch tensors.
4. Implement an encoder–decoder architecture using `nn.GRU`.
5. Implement an encoder–decoder architecture using `nn.LSTM`.
6. Understand the different hidden-state requirements of GRU and LSTM.
7. Implement teacher forcing and next-token prediction during training.
8. Train and evaluate each model independently.
9. Implement autoregressive inference using greedy decoding and beam search.
10. Organize reusable preprocessing, training, and evaluation functions.
11. Document the implementations clearly in a GitHub repository.

## 3. Project Scope

This is a learning-focused deep learning project, not a production translation system or an architecture benchmarking exercise.

The project will use one English–French sentence-pair dataset and a shared preprocessing pipeline.

The GRU and LSTM implementations will be developed in separate notebooks. Each model will have its own architecture implementation, training process, and evaluation examples.

The emphasis is on understanding how each architecture works and how to implement it correctly using PyTorch's high-level APIs.

Extensive hyperparameter searches, unnecessary exploratory data analysis, and production deployment are outside the scope of this project.

## 4. Notebook Structure

### Notebook 1: Text Cleaning and Data Preparation

- Load and inspect the sentence-pair dataset.
- Clean and tokenize the English and French text.
- Build separate source and target vocabularies.
- Define special tokens.
- Convert sentences into numerical sequences.
- Apply padding and sequence-length handling.
- Split the dataset into training and validation sets.
- Implement the PyTorch Dataset and DataLoader.
- Verify the prepared data and tensor shapes.

**Deliverable:** A reusable, verified dataset pipeline.

### Notebook 2: GRU Encoder–Decoder

- Implement the encoder using `nn.GRU`.
- Implement the decoder using `nn.GRU`.
- Connect both components into a Seq2Seq model.
- Implement teacher forcing and the training loop.
- Train the model and monitor training and validation loss.
- Implement greedy decoding and beam search.
- Evaluate generated translations and inspect model behavior.

**Deliverable:** A working GRU-based translation system.

### Notebook 3: LSTM Encoder–Decoder

- Implement the encoder using `nn.LSTM`.
- Implement the decoder using `nn.LSTM`.
- Understand and transfer both hidden and cell states.
- Implement the training loop, reusing appropriate utility functions.
- Train the model and monitor training and validation loss.
- Implement greedy decoding and beam search.
- Evaluate generated translations and inspect model behavior.

**Deliverable:** A working LSTM-based translation system.

## 5. Reusable Functions

Reusable functions will be introduced where they reduce duplication and improve clarity.

Examples include:

- Text preprocessing and vocabulary utilities.
- Dataset preparation and batching.
- Training and validation loops.
- Loss calculation.
- Greedy decoding and beam search.
- Translation evaluation and output inspection.

The GRU and LSTM models will retain their own architecture implementations. Shared utilities will support both without forcing them into a single model implementation.

## 6. GitHub Repository Structure

```text
english-french-machine-translation/
├── notebooks/
│   ├── 01_text_preprocessing.ipynb
│   ├── 02_gru_translation.ipynb
│   └── 03_lstm_translation.ipynb
├── src/
│   └── utils.py
├── README.md
└── .gitignore
```

The `requirements.txt` file is removed. We can document the main dependencies and environment setup in the README if needed.

Large datasets and model checkpoints will not be committed to GitHub unnecessarily.

## 7. Expected Outcome

By the end of the project, I should understand how to prepare text for sequential modeling, build and train a GRU-based encoder–decoder, build and train an LSTM-based encoder–decoder, and generate translations using autoregressive decoding.

The success of the project will be measured by my understanding and correct implementation of the concepts, rather than by determining which architecture performs better.
