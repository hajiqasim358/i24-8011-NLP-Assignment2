# DS 5007: Natural Language Processing Assignment 2

**Student ID:** i24-8011  
**Course:** DS 5007, Natural Language Processing  
**University:** National University of Computer and Emerging Sciences, Islamabad Campus

This repository contains the implementation of Assignment 2. The work continues the BBC Urdu NLP pipeline and covers word representations, Urdu sequence labeling, and topic classification using a Transformer encoder implemented in PyTorch.

## Repository

GitHub: https://github.com/hajiqasim358/i24-8011-NLP-Assignment2

## Project Overview

The assignment is divided into three main parts.

| Part | Task | Main methods |
|---|---|---|
| 1 | Word embeddings | TF IDF, PPMI, t SNE, Skip Gram Word2Vec |
| 2 | Sequence labeling | Urdu POS tagging, NER, BiLSTM, CRF and Viterbi decoding |
| 3 | Topic classification | Custom Transformer encoder with multi head self attention |

The notebook contains the implementation, experiments, evaluation, visualizations, and saved model checkpoints.

## Dataset

The project uses BBC Urdu news articles.

The main input files are:

| File | Description |
|---|---|
| `data/raw.txt` | Raw Urdu news corpus |
| `data/cleaned.txt` | Cleaned Urdu news corpus used for the experiments |
| `data/Metadata.json` | Article metadata, including article identifiers, titles, URLs, and topic information |

The Transformer topic classification section uses 250 articles. All 250 articles were matched with the cleaned corpus and assigned one of five topic categories.

### Topic categories

| Category | Articles | Percentage |
|---|---:|---:|
| Politics | 25 | 10.0% |
| Sports | 73 | 29.2% |
| Economy | 13 | 5.2% |
| International | 57 | 22.8% |
| Health & Society | 82 | 32.8% |
| **Total** | **250** | **100%** |

## Part 1: Word Embeddings

This part builds and evaluates different word representations from the Urdu corpus.

### 1. TF IDF

A term document matrix is constructed from the cleaned corpus. The vocabulary is limited to the 10,000 most frequent tokens, with other tokens mapped to `<UNK>`.

The TF IDF matrix is saved as:

`embeddings/tfidf_matrix.npy`

The notebook also identifies discriminative words for the topic categories.

### 2. PPMI

A word co occurrence matrix is constructed using a symmetric context window of five tokens. Positive Pointwise Mutual Information weighting is then applied.

The resulting matrix is saved as:

`embeddings/ppmi_matrix.npy`

The notebook includes a t SNE visualization and nearest neighbour analysis.

### 3. Skip Gram Word2Vec

A Skip Gram Word2Vec model is implemented in PyTorch. It uses separate centre and context embedding matrices and negative sampling.

The main experiment settings include:

| Parameter | Value |
|---|---:|
| Embedding dimension | 100 |
| Context window | 5 |
| Negative samples | 10 |
| Learning rate | 0.001 |
| Optimizer | Adam |

The averaged centre and context embeddings are saved in the `embeddings` directory.

### 4. Four condition comparison

The notebook compares four representations using Mean Reciprocal Rank (MRR).

| ID | Condition | MRR |
|---|---|---:|
| C1 | PPMI baseline | 0.096005 |
| C2 | Skip Gram trained on raw corpus, 100D | 0.051791 |
| C3 | Skip Gram trained on cleaned corpus, 100D | 0.051009 |
| C4 | Skip Gram trained on cleaned corpus, 200D | 0.028986 |

In this experiment, the PPMI representation achieved the highest MRR. Increasing the Skip Gram embedding dimension from 100 to 200 did not improve MRR on the manually labelled word pairs.

The notebook also includes nearest neighbour results and analogy experiments. The analogy section checks vocabulary coverage and evaluates the top three predicted candidates. Vocabulary verification passing does not mean that the analogy prediction itself is correct.

### Part 1 output files

Examples of generated files include:

- `embeddings/tfidf_matrix.npy`
- `embeddings/ppmi_matrix.npy`
- `embeddings/embeddings_w2v.npy`
- `embeddings/embeddings_w2v_raw_100d.npy`
- `embeddings/embeddings_w2v_cleaned_200d.npy`
- `embeddings/word2idx.json`
- `results/four_condition_mrr_comparison.csv`
- `results/four_condition_mrr_comparison.json`
- `results/four_condition_detailed_mrr.json`
- `results/four_condition_nearest_neighbours.json`
- `results/ppmi_tsne_visualization.png`
- Skip Gram training loss plots and nearest neighbour results in `results/`

## Part 2: POS Tagging and Named Entity Recognition

This part uses 500 selected and annotated Urdu sentences. The dataset is split into training, validation, and test sets. POS tagging and NER are treated as token level sequence labeling tasks.

### POS tagging

A two layer bidirectional LSTM is trained using the Skip Gram embeddings from Part 1. Two embedding settings are evaluated:

1. Frozen embeddings
2. Fine tuned embeddings

The POS model uses a linear classification layer and cross entropy loss.

The notebook reports the following test results for the fine tuned POS model:

| Metric | Score |
|---|---:|
| Token accuracy | 95.11% |
| Macro precision | 93.86% |
| Macro recall | 95.43% |
| Macro F1 | 94.53% |

The best validation macro F1 values recorded in the embedding comparison were:

| Embedding mode | Best validation macro F1 |
|---|---:|
| Frozen | 86.79% |
| Fine tuned | 95.74% |

The notebook also generates a POS confusion matrix, class distribution information, and an embedding comparison.

### Named Entity Recognition

NER uses BIO labels for person, location, organisation, miscellaneous entities, and outside tokens. A BiLSTM with a Conditional Random Field output layer is trained. The CRF uses Viterbi decoding to find the best label sequence.

Test results with the CRF layer:

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| Person (PER) | 95.00% | 84.44% | 89.41% |
| Location (LOC) | 90.91% | 83.33% | 86.96% |
| Organisation (ORG) | 93.94% | 86.11% | 89.86% |
| Miscellaneous (MISC) | 0.00% | 0.00% | 0.00% |
| **Overall** | **93.40%** | **84.62%** | **88.79%** |

There were no MISC entities in the test support, so the MISC score should not be interpreted as performance on actual MISC examples.

The notebook also evaluates a Softmax model without CRF and includes ablation experiments for sequence direction, dropout, and embedding initialization. The recorded test entity F1 for the Softmax model without CRF was 0. The ablation results are saved in `results/ner_ablation_test_results.csv`.

### Part 2 output files

Examples of generated files include:

- `data/annotated_dataset.json`
- `data/selected_500_sentences.json`
- `data/train_dataset.json`
- `data/validation_dataset.json`
- `data/test_dataset.json`
- `data/pos_vocabulary_inventory.csv`
- `models/pos_bilstm_frozen.pt`
- `models/pos_bilstm_finetuned.pt`
- `models/ner_bilstm_crf.pt`
- `models/ner_bilstm_softmax.pt`
- `models/ner_crf.pt`
- `models/ner_ablation_a1_unidirectional.pt`
- POS and NER evaluation tables, confusion matrices, training curves, and error analysis files in `results/`

## Part 3: Transformer Encoder for Topic Classification

A Transformer encoder is implemented from individual PyTorch modules. PyTorch built in Transformer encoder and multi head attention modules are not used.

### Dataset preparation

The 250 articles are assigned to five categories. Each article is represented as a sequence of token IDs with a maximum length of 256 tokens.

The stratified dataset split is:

| Split | Articles |
|---|---:|
| Training | 175 |
| Validation | 37 |
| Test | 38 |
| **Total** | **250** |

The Transformer vocabulary contains 7,495 entries, including the special tokens `[PAD]`, `[UNK]`, and `[CLS]`.

The vocabulary is saved at:

`data/transformer_vocabulary.json`

### Model configuration

| Component | Configuration |
|---|---|
| Model dimension | 128 |
| Attention heads | 4 |
| Key dimension per head | 32 |
| Value dimension per head | 32 |
| Feed forward dimension | 512 |
| Encoder blocks | 4 |
| Maximum sequence length | 256 |
| Dropout | 0.1 |
| Classification classes | 5 |
| Classification head | 128 → 64 → 5 |
| Positional encoding | Fixed sinusoidal |
| Normalization | Pre Layer Normalization |

The model includes scaled dot product attention with padding masks, multi head self attention, a position wise feed forward network, sinusoidal positional encoding, residual connections, and a learned CLS token.

### Training configuration

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Weight decay | 0.01 |
| Epochs | 20 |
| Warmup steps | 50 |
| Scheduler | Cosine learning rate schedule |
| Loss | Cross entropy |

### Transformer test results

| Metric | Score |
|---|---:|
| Test accuracy | 78.95% |
| Macro precision | 60.83% |
| Macro recall | 62.91% |
| Macro F1 | 61.65% |

The test set contains 38 articles. The notebook also produces a five class confusion matrix, per class evaluation scores, training curves, and attention heatmaps for correctly classified articles.

The saved model checkpoint is:

`models/transformer_topic_classifier.pt`

### 8.2 BiLSTM vs. Transformer Comparison

The BiLSTM and Transformer architectures use different approaches to process sequential text.

The BiLSTM processes text in both forward and backward directions, capturing contextual information from both sides of a token. The Transformer uses self attention to capture relationships between tokens across an input sequence.

The Transformer achieved a test accuracy of 78.95% and a macro F1 score of 61.65% on the five category topic classification task.

The comparison considers five aspects:

1. Classification accuracy and the difference between the models.
2. Number of epochs required for convergence.
3. Average training time per epoch and computational differences.
4. Attention heatmaps showing which tokens receive higher attention weights.
5. Suitability of BiLSTM and Transformer architectures for datasets containing only 200 to 300 articles.

The BiLSTM provides bidirectional contextual representations through recurrent processing, whereas the Transformer uses multiple attention heads to learn relationships between tokens. The attention heatmaps provide an additional way to inspect the contextual information captured by the Transformer.

For small datasets, model suitability depends on dataset size, model complexity, training time, and generalisation performance.

## Project Structure

The following is the main project structure. Some generated result files may be added as experiments are updated.

```text
i24-8011-NLP-Assignment2/
│
├── data/
│   ├── annotated_dataset.json
│   ├── article_topics.json
│   ├── cleaned.txt
│   ├── lemmatized.txt
│   ├── Metadata.json
│   ├── pos_vocabulary_inventory.csv
│   ├── raw.txt
│   ├── selected_500_sentences.json
│   ├── test_dataset.json
│   ├── train_dataset.json
│   ├── transformer_vocabulary.json
│   ├── urdu_lemma_dictionary.csv
│   └── validation_dataset.json
│
├── embeddings/
│   ├── embeddings_w2v.npy
│   ├── embeddings_w2v_cleaned_200d.npy
│   ├── embeddings_w2v_raw_100d.npy
│   ├── ppmi_matrix.npy
│   ├── tfidf_matrix.npy
│   └── word2idx.json
│
├── models/
│   ├── ner_ablation_a1_unidirectional.pt
│   ├── ner_bilstm_crf.pt
│   ├── ner_bilstm_softmax.pt
│   ├── ner_crf.pt
│   ├── pos_bilstm_finetuned.pt
│   ├── pos_bilstm_frozen.pt
│   └── transformer_topic_classifier.pt
│
├── report/
│
├── results/
│   ├── four_condition_detailed_mrr.json
│   ├── four_condition_mrr_comparison.csv
│   ├── four_condition_mrr_comparison.json
│   ├── four_condition_nearest_neighbours.json
│   ├── ner_ablation_test_results.csv
│   ├── ner_crf_test_results.csv
│   ├── ner_crf_vs_softmax.csv
│   ├── ner_error_analysis.csv
│   ├── pos_confusion_matrix.csv
│   ├── pos_embedding_comparison.csv
│   └── ... additional plots and evaluation files
│
├── .gitattributes
├── Assignment_2.ipynb
├── README.md
└── requirements.txt
```

Large NumPy arrays and model checkpoints are tracked using Git LFS.

## Requirements

The project environment used:

| Software or library | Version |
|---|---|
| Python | 3.12.7 |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| pandas | 3.1.0 |
| scikit learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| Seaborn | 0.13.2 |
| PyTorch | 2.12.1+cu130 |

The PyTorch build used in the experiments is the CUDA 13.0 build. A compatible NVIDIA driver and CUDA capable GPU are needed to run the notebook with GPU acceleration. The notebook can be changed to use CPU, but training will take longer.

## Setup and Reproduction

### 1. Clone the repository

```bash
git clone https://github.com/hajiqasim358/i24-8011-NLP-Assignment2.git
cd i24-8011-NLP-Assignment2
```

### 2. Set up Git LFS

Install Git LFS if it is not already installed, then run:

```bash
git lfs install
git lfs pull
```

This downloads the large files tracked by Git LFS.

### 3. Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install dependencies

Install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

If the CUDA PyTorch wheel is not available for the current platform, install a compatible PyTorch build from the official PyTorch installation selector and then install the remaining packages.

### 5. Open the notebook

Start Jupyter:

```bash
pip install notebook
jupyter notebook
```

Open `Assignment_2.ipynb`.

### 6. Run the notebook

Run the notebook cells in order, from top to bottom. The notebook reads the corpus and metadata from the `data/` directory and writes embeddings, checkpoints, and evaluation outputs to the relevant project directories.

The main input files must be present before running the notebook:

- `data/raw.txt`
- `data/cleaned.txt`
- `data/Metadata.json`

Some cells train models and may take time. GPU acceleration is recommended, especially for Skip Gram training and the Transformer model.

## Reproducibility Notes

- Run the notebook from the project root so that relative paths to `data/`, `embeddings/`, `models/`, and `results/` resolve correctly.
- Run cells in order because later sections use vocabularies, embeddings, datasets, and model checkpoints created earlier.
- The reported metrics in this README are taken from the saved notebook outputs.
- Training results can vary if random seeds, data splits, library versions, or hardware settings change.
- The notebook includes both training code and evaluation outputs. Re running training cells will update the corresponding checkpoints and result files.

## Version Control

The repository uses incremental commits to record progress across the assignment. Git LFS is configured for large model and array files through `.gitattributes`.

## Author

**Student ID:** i24-8011  
**Course:** DS 5007, Natural Language Processing  
**Institution:** National University of Computer and Emerging Sciences, Islamabad Campus
