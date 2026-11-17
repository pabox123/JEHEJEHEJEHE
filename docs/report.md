# Sentiment Analysis on GoEmotions: Project Status and Transformer Extension

## 1. Introduction and Objectives

This project studies binary sentiment classification of Reddit comments from the GoEmotions dataset. The planned comparison includes a dummy baseline, a dense neural network, a vanilla recurrent neural network (RNN), a Long Short-Term Memory (LSTM) network, and a Transformer-based model.

The repository currently contains a data-preparation plan and a working DistilBERT extension. The baseline, dense, RNN, and LSTM notebooks are still placeholders, so no experimental comparison among those models can be reported yet. The project also relates sequence processing and neural memory to Turing Machine concepts, especially state, memory, transitions, and computability.

## 2. Dataset and Experimental Configuration

The dataset is GoEmotions, containing Reddit comments annotated with fine-grained emotions. The implemented filtering step retains comments consistently associated with the positive or negative emotion groups and excludes ambiguous, neutral, and mixed comments.

Loading the three official TSV files produced 29,991 filtered comments: 9,214 negative and 20,777 positive. The balancing step randomly selected 500 comments from each class using seed 42, producing a 1,000-comment dataset saved as `data/processed/balanced_dataset.csv`. Raw TSV files and processed CSV files are excluded from version control.

The Transformer notebook uses the following configuration:

| Setting | Value |
| :--- | :--- |
| Model | `distilbert-base-uncased` |
| Tokenizer | `DistilBertTokenizerFast` |
| Random seed | 42 |
| Train/test split | 80/20, stratified |
| Training size | 800 |
| Test size | 200 |
| Maximum sequence length | 64 tokens |
| Batch size | 16 |
| Epochs | 5 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Loss | Cross-entropy |
| Gradient clipping | Maximum norm 1.0 |

The dataset class converts text to strings, applies truncation and fixed-length padding, and returns `input_ids`, `attention_mask`, and integer labels. The Transformer uses the first hidden-state token as the classification representation and applies a dropout/linear classification head.

## 3. Implemented Components

### 3.1 Planned but Not Yet Implemented

The files `notebooks/01_eda.ipynb`, `notebooks/02_baseline_model.ipynb`, and `notebooks/03_dense_rnn_lstm.ipynb` contain “To be implemented” placeholders. Therefore, the following items have no verified implementation or results in the current repository:

- Dummy Classifier evaluation
- Dense neural network
- Vanilla RNN
- LSTM
- Comparative training times and performance

The report does not assign metrics or an empirical ranking to these models.

### 3.2 Transformer Extension

The implemented extension fine-tunes `distilbert-base-uncased` for binary classification. It:

1. Loads the balanced dataset when it is available, or creates a clearly marked synthetic development sample otherwise.
2. Tokenizes comments with `DistilBertTokenizerFast`.
3. Creates stratified training and test partitions.
4. Trains with AdamW for five epochs.
5. Clips gradients to a maximum norm of 1.0.
6. Selects the best model according to validation accuracy.
7. Computes accuracy, precision, recall, F1-score, Cohen's kappa, a confusion matrix, and a classification report.

## 4. Available Transformer Result

The tracked file `outputs/metrics/transformer_metrics.json` reports the following result on the 200-comment test split:

| Model | Accuracy | Precision | Recall | F1-Score | Cohen's Kappa |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DistilBERT | 0.775 | 0.762 | 0.800 | 0.780 | 0.550 |

The confusion matrix is `[[75, 25], [20, 80]]`: 155 test comments were classified correctly and 45 were misclassified. Cohen's kappa is 0.550, indicating moderate agreement beyond chance for this test split. The result describes the balanced 1,000-comment subset and should not be generalized to the complete GoEmotions dataset.

## 5. Relation to Turing Machine Concepts

The theoretical comparison is an architectural interpretation, not an experimental result from the current repository.

### 5.1 Recurrent Models

An RNN updates a hidden state sequentially:

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h).
$$

The hidden state is a bounded numerical summary of the processed prefix. It can be compared with the state and working memory of a Turing Machine, while the recurrent update plays a role analogous to a transition rule. This analogy is useful, but a practical RNN with finite precision and a fixed hidden dimension is not literally an unrestricted Turing Machine.

An LSTM adds a cell state and gates that control which information is retained, written, or exposed. This provides a more effective mechanism for preserving information across a sequence and helps reduce the practical effects of vanishing gradients. It remains finite-dimensional and bounded in an actual implementation.

### 5.2 Transformer

The Transformer does not process a sequence through one recurrent hidden state. Self-attention computes interactions between token positions, allowing each position to use information from the complete input window. This supports parallel processing during training and provides direct access to other positions within the fixed context.

The implementation uses a maximum length of 64 tokens. Therefore, its memory is bounded by the input window, model parameters, and finite numerical precision. These practical constraints should be distinguished from theoretical statements about Turing completeness or unrestricted computability.


## 6. References

1. Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*. Proceedings of ACL 2020, 4040–4054. DOI: 10.18653/v1/2020.acl-main.372.
2. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
3. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv:1910.01108.
4. Siegelmann, H. T., & Sontag, E. D. (1992). *On the computational power of neural nets*. Proceedings of the 5th Annual Workshop on Computational Learning Theory, 282–290.
5. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems, 30.
