# English-to-French Neural Machine Translation

A learning-focused neural machine translation project implementing GRU- and LSTM-based encoder–decoder models using PyTorch's high-level recurrent neural network APIs.

The project explores how text is prepared for sequential modeling, how recurrent encoder–decoder architectures work, how sequence-to-sequence models are trained, and how autoregressive decoding generates translations.

## 1. Project Overview

Neural machine translation converts text from one natural language into another using a trained neural network.

In this project, English sentences serve as the source sequences, while their corresponding French translations serve as the target sequences. The system uses an encoder–decoder architecture to process the English input and generate a French translation one token at a time.

Two independent implementations are developed:

- **GRU encoder–decoder:** Uses PyTorch's `nn.GRU`.
- **LSTM encoder–decoder:** Uses PyTorch's `nn.LSTM`.

The primary objective is to understand and correctly implement both architectures. The project is not intended to establish which architecture performs better.

## 2. Learning Objectives

The project covers the following concepts:

- Text cleaning and preparation for sequential modeling.
- Tokenization and vocabulary construction.
- Special tokens, padding, truncation, and sequence lengths.
- Conversion of text into numerical sequences and PyTorch tensors.
- Encoder–decoder architecture for sequence-to-sequence learning.
- GRU and LSTM implementations using high-level PyTorch APIs.
- Hidden-state handling in GRUs and hidden-state and cell-state handling in LSTMs.
- Teacher forcing and next-token prediction during training.
- Training and validation workflows.
- Autoregressive inference using greedy decoding and beam search.
- BLEU-based translation evaluation.
- Reusable preprocessing, training, and evaluation utilities.
- Organizing and documenting a deep learning project on GitHub.

## 3. Dataset and Preprocessing

The project uses the English–French sentence-pair dataset distributed with the *Dive into Deep Learning* (D2L) materials.

The preprocessing pipeline prepares the sentence pairs for neural sequence modeling.

### Preprocessing workflow

1. Load and inspect the English–French sentence pairs.
2. Tokenize the English and French sentences separately.
3. Build separate source and target vocabularies.
4. Define special tokens for unknown tokens, padding, beginning of sequence, and end of sequence.
5. Convert sentences into token IDs.
6. Apply fixed-length padding and truncation.
7. Calculate sequence lengths for encoder processing.
8. Split the dataset into training, validation, and test sets.
9. Create PyTorch datasets and data loaders.
10. Verify tensor shapes and the handling of special tokens.

The English and French vocabularies are separate because the two languages have different token distributions.

The prepared dataset is reused by both model implementations to maintain a consistent data-processing pipeline.

## 4. Model Architecture

Both implementations use an encoder–decoder architecture.

### Encoder

The encoder processes the source sentence and produces a representation of the input sequence. Its final recurrent state is passed to the decoder to initialize target-sequence generation.

### Decoder

The decoder generates the target sentence autoregressively. During training, it uses teacher forcing with the target sequence. During inference, it starts with the beginning-of-sequence token and feeds previously generated tokens back into the decoder.

### GRU implementation

The GRU model uses `nn.GRU` for both the encoder and decoder. Its recurrent state is represented by the hidden state.

### LSTM implementation

The LSTM model uses `nn.LSTM` for both the encoder and decoder. Its recurrent state consists of two components:

- Hidden state
- Cell state

Both components are transferred from the encoder to initialize the decoder.

## 5. Project Structure

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

### Notebook 1: Text Cleaning and Data Preparation

Contains the dataset inspection, tokenization, vocabulary construction, sequence preparation, dataset splitting, and batching workflow.

### Notebook 2: GRU Encoder–Decoder

Implements the GRU encoder, decoder, sequence-to-sequence model, training workflow, and autoregressive inference.

### Notebook 3: LSTM Encoder–Decoder

Implements the corresponding LSTM architecture, including the transfer of hidden and cell states, training, inference, and translation evaluation.

### Reusable utilities

The `src/utils.py` module contains reusable functions that support common operations such as model training, validation, plotting training history, and evaluation.

The GRU and LSTM architecture implementations remain separate to make their differences easier to understand.

## 6. Training Approach

Both models are trained independently using the prepared dataset.

The training workflow includes:

- Mini-batch training with PyTorch data loaders.
- Teacher forcing for decoder inputs.
- Cross-entropy loss for next-token prediction.
- Exclusion of padding tokens from the loss calculation.
- Backpropagation and gradient clipping.
- Validation loss and token-level accuracy monitoring.
- Learning-rate scheduling.
- Saving trained model checkpoints.

The experiments cover a baseline configuration, a deeper configuration, and a configuration with a larger embedding dimension. These experiments support learning how model configurations are implemented and trained; they are not intended as an extensive hyperparameter search or a definitive architecture comparison.

## 7. Autoregressive Inference

After training, the decoder generates translations without access to the correct target tokens.

The inference process starts with the French beginning-of-sequence token and repeatedly predicts the next token. The generated token is then fed back into the decoder until an end-of-sequence token is generated or the maximum output length is reached.

Two decoding strategies are implemented.

### Greedy decoding

Greedy decoding selects the highest-probability next token at each generation step.

It is straightforward to implement, but an earlier token selection cannot be reconsidered.

### Beam search

Beam search maintains multiple candidate sequences during generation. At each step, the candidates are expanded, scored using cumulative log probabilities, and reduced to the highest-scoring candidates.

The LSTM implementation uses a beam width of 3 in the documented evaluation experiment.

Beam search explores more candidate sequences than greedy decoding, but it does not guarantee a better translation in every case.

## 8. Translation Evaluation

The LSTM E2 checkpoint was evaluated using greedy decoding and beam search. Both methods used the same trained model and the same test examples.

SacreBLEU was used to calculate corpus-level BLEU scores against the corresponding French reference translations.

### Preliminary evaluation results

The following results were obtained from a sample of 100 test examples.

| Decoding strategy | BLEU score |
|---|---:|
| Greedy decoding | 38.67 |
| Beam search (width 3) | 46.35 |
| Difference | +7.68 |

Beam search achieved a BLEU score 7.68 points higher than greedy decoding on this sample.

These results are preliminary and should not be interpreted as the final performance of the model on the complete test set. BLEU measures textual overlap with reference translations and does not directly establish semantic correctness or grammatical quality.

A full test-set evaluation and manual inspection of generated translations would provide a broader assessment.

## 9. Technology Stack

- **Python** — implementation language.
- **PyTorch** — recurrent neural network architectures, training, and inference.
- **Hugging Face Tokenizers** — text tokenization and vocabulary handling.
- **NumPy and pandas** — data handling and numerical operations.
- **Matplotlib** — training-history visualization.
- **SacreBLEU** — automatic machine translation evaluation.
- **Jupyter Notebook / Google Colab** — experimentation and GPU-backed execution.
- **Git and GitHub** — version control and project documentation.

## 10. Environment Setup

The project was developed using Python and PyTorch in a notebook-based environment.

To reproduce the work:

1. Clone the repository.
2. Install the required libraries listed in the notebook imports.
3. Open the notebooks in Jupyter or Google Colab.
4. Ensure the dataset is available at the path expected by the preprocessing notebook.
5. Run the preprocessing notebook before the model notebooks if the prepared data is not already available.
6. Run the GRU or LSTM notebook independently, following its documented workflow.

The project does not include a `requirements.txt` file. Dependencies should be installed according to the imports and environment setup documented in the notebooks.

Google Colab can be used for GPU-backed execution. Inference does not require retraining if a compatible saved checkpoint and the necessary tokenizer artifacts are available.

## 11. Scope and Limitations

This is an educational implementation focused on understanding recurrent sequence-to-sequence modeling.

The project does not aim to provide a production-ready translation service.

The current implementations use recurrent encoder–decoder architectures without attention. Consequently, the models have limitations when representing longer input sequences and generating complex translations.

Other limitations include:

- Evaluation results currently documented for the LSTM decoding comparison come from only 100 test examples.
- BLEU alone cannot establish translation quality.
- Beam search uses cumulative log probabilities without a length penalty in the current implementation.
- No extensive hyperparameter search or production deployment is included.

These limitations are documented to distinguish the learning objectives from the requirements of a production translation system.

## 12. Expected Outcomes

By completing this project, I aim to understand how to prepare paired text data, implement GRU and LSTM encoder–decoder architectures, train sequence-to-sequence models, and generate translations using autoregressive decoding.

The main outcome is practical understanding of recurrent neural networks and sequence-to-sequence learning through working implementations, rather than identifying a universally superior architecture.

## 13. References

- [Dive into Deep Learning (D2L)](https://d2l.ai/) — textbook and English–French dataset materials.
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — neural network modules, training, and tensor operations.
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) — tokenization and vocabulary utilities.
- [SacreBLEU](https://github.com/mjpost/sacrebleu) — standardized BLEU evaluation.

---

**Project:** English-to-French Neural Machine Translation

**Focus:** Recurrent neural networks, sequence-to-sequence learning, and autoregressive inference.

**Primary objective:** Learn to implement and understand GRU and LSTM encoder–decoder models using PyTorch.
