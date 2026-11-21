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

## 3. Neural Models: Architecture and Training Procedure

All three neural models share the same data pipeline: the 1,000-comment balanced
split (800 train / 200 test, stratified, seed 42) produced in Section 2. Text was
lowercased and stripped of punctuation and English stopwords during preprocessing
(`notebooks/01_eda.ipynb`), yielding the column `cleaned_text`. For the bag-of-words
(Dense) branch we used a `CountVectorizer` with `max_features=5000` and English
stopword removal, resulting in a 2,253-dimensional vocabulary. For the recurrent
branch we reused the same vocabulary, mapped tokens to indices, and represented each
comment as a sequence of at most 30 tokens (the approximate maximum length of
GoEmotions comments), with left-padding and index `0` reserved for padding/OOV
tokens. All models were trained in PyTorch 2.x on CPU with binary
cross-entropy loss, the Adam optimizer, a fixed batch size, and early stopping on
the validation loss. The final epoch count, learning rate, and patience per model
are given below.

### 3.1 Dense Neural Network (bag-of-words input)

The Dense model ignores word order and treats the vectorized comment as a single
feature vector. Its architecture is:

| Layer | Configuration |
| --- | --- |
| Input | BoW vector, 2,253 dimensions |
| Hidden 1 | Linear(2253 → 256) + ReLU + Dropout(0.3) |
| Hidden 2 | Linear(256 → 128) + ReLU + Dropout(0.3) |
| Output | Linear(128 → 1) + sigmoid |

Training hyperparameters: 20 epochs, learning rate 1e-3, batch size 64,
early-stopping patience 4. Dropout was included because the small dataset
(800 training comments) makes the high-dimensional input easy to overfit.

### 3.2 Vanilla Recurrent Neural Network

The vanilla RNN processes the token sequence sequentially through a recurrent
hidden state $h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)$ (cf. Section 5.1 of the
Theoretical analysis). We used a single-layer, **bidirectional** vanilla RNN so that
the final representation summarizes the comment from both directions:

| Layer | Configuration |
| --- | --- |
| Embedding | vocab 2,254 → 64 |
| Recurrent layer | 1 bidirectional vanilla RNN, hidden 128 per direction |
| Pooling | Concatenation of forward/backward final states (256) |
| Head | Dropout(0.3) + Linear(256 → 1) + sigmoid |

Training hyperparameters: 15 epochs, learning rate 1e-3, batch size 64,
early-stopping patience 3.

### 3.3 LSTM Network

The LSTM augments the recurrent update with a gated cell state, which preserves
information across positions and mitigates the vanishing-gradient problem that a
plain RNN suffers on sequential data (Hochreiter & Schmidhuber, 1997). The
implemented architecture is:

| Layer | Configuration |
| --- | --- |
| Embedding | vocab 2,254 → 64 |
| Recurrent layers | 2 stacked **bidirectional** LSTM layers |
| Pooling | Concatenation of forward/backward final states |
| Head | Dropout(0.3) + Linear → 1 + sigmoid |

Because the LSTM contains the largest number of parameters among the recurrent
models, we performed a manual grid search over three key hyperparameters
(hidden size ∈ {64, 128, 256}, learning rate ∈ {1e-3, 5e-4}, batch size ∈
{32, 64}), evaluating all 12 combinations for 10 epochs and ranking them by
F1-score (`notebooks/03_dense_rnn_lstm.ipynb`). The selected configuration
(hidden 64, lr 1e-3, batch 32) was retrained for up to 20 epochs with patience 4;
its final metrics are reported in Section 3.4 and saved to
`outputs/metrics/lstm_tuned_metrics.json`.

### 3.4 Performance Evaluation

All models were evaluated on the held-out 200-comment stratified test split.
Metrics (accuracy, precision, recall, macro-style F1 on the positive class, and
Cohen's kappa) were computed by `src/evaluate.py`; confusion matrices are
available in `outputs/figures/`.

| Model | Accuracy | Precision | Recall | F1-Score | Cohen's Kappa |
| --- | --- | --- | --- | --- | --- |
| Dense NN (BoW) | 0.745 | 0.738 | 0.760 | 0.749 | 0.490 |
| Vanilla RNN | 0.550 | 0.536 | 0.740 | 0.622 | 0.100 |
| LSTM (tuned) | 0.525 | 0.520 | 0.660 | 0.582 | 0.050 |
| DistilBERT (extension, Section 4) | 0.775 | 0.762 | 0.800 | 0.780 | 0.550 |

The corresponding confusion matrices (rows = true label, columns = predicted
label, order Negative/Positive) are:

- Dense NN: [[73, 27], [24, 76]]
- Vanilla RNN: [[36, 64], [26, 74]]
- LSTM (tuned): [[39, 61], [34, 66]]

Several observations follow directly from these numbers. First, the **Dense
bag-of-words network is the strongest recurrent-class competitor**: it outperforms
both sequential models on every metric and is the only classical model with a
substantially above-chance kappa (0.49). This is consistent with the very short
comments in GoEmotions (≈13 words): sentiment is largely carried by individual
keywords, and a parallel BoW representation avoids the information loss and
optimization difficulty that sequential processing introduces on such short,
sparse sequences. Second, the **vanilla RNN is heavily biased toward the positive
class** (recall 0.74 but accuracy 0.55, kappa 0.10): its confusion matrix shows
that most negative comments are misclassified as positive, a typical symptom of
underfitting on a small dataset and of gradient vanishing through time. Third, the
**tuned LSTM** still struggles to generalize (recall 0.66, kappa 0.05): even after
the grid search, the 800-comment training split is not enough for a two-layer
bidirectional LSTM to overcome the frequency bias of the data, and the early
stopping on the validation curve (patience 4) may have halted training
prematurely. The near-zero kappa indicates agreement barely above chance.
Finally, the ordering DistilBERT > Dense NN > Vanilla RNN > LSTM supports the
main theoretical discussion of this project (Section 5): architectures with
richer, content-based access to sequence information outperform single-state
recurrent memory in practice, even though all these devices remain bounded,
finite-memory approximations of a Turing machine rather than unrestricted ones.
As with Section 4, these results describe only the balanced 1,000-comment subset
and should not be generalized to the full GoEmotions corpus.

### 3.5 Transformer Extension

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
