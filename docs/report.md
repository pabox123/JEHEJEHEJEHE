# Sentiment Analysis on GoEmotions: Comparative Study of Neural Architectures and Turing Computability

**Course:** Computación y Estructuras Discretas III (CED-III-IT2) — Universidad Icesi (2026-2)  
**Authors:** Valeria Piza Saavedra (A00405037), Krystal Angely Giraldo Ramirez (A00407967), Miguel Jose Pabon Benavidez (A00406585)

---

## 1. Introduction and Objectives

This project investigates binary sentiment classification (**positive** vs. **negative**) on short, informal Reddit comments from the **GoEmotions** dataset (Demszky et al., 2020). Social media comments pose significant challenges for natural language processing due to brevity (~13 words on average), conversational slang, irony, and sparse lexical context.

The primary objectives of this work are:
1. **Data Engineering & Exploratory Analysis**: Construct a reproducible preprocessing pipeline that filters fine-grained emotion annotations into pure binary sentiment classes, balances the corpus, and characterizes vocabulary and length distributions.
2. **Neural Architecture Implementation & Tuning**: Design, train, and evaluate a no-skill baseline (**DummyClassifier**), three classical neural architectures (**Dense Neural Network**, **Bidirectional Vanilla RNN**, and **Stacked Bidirectional LSTM** with hyperparameter grid search), and a pre-trained Transformer extension (**DistilBERT**).
3. **Multi-Metric Comparative Evaluation**: Compare training and test behavior across all models using **Accuracy**, **Precision**, **Recall**, **F1-Score**, **Cohen's Kappa ($\kappa$)**, and confusion matrices.
4. **Formal Computability Analysis (OE3.3–OE3.6)**: Relate recurrent hidden states, gated memory cells, and Transformer self-attention to **Turing Machine** concepts, including state transitions ($\delta$), tape memory access, and bounded finite-state automata.

---

## 2. Dataset, Exploratory Data Analysis, and Preprocessing

### 2.1 Dataset Filtering and Balancing
GoEmotions comprises 58,009 English Reddit comments annotated across 27 fine-grained emotion categories plus a neutral label. Using the official sentiment grouping defined in `src/preprocessing.py` (12 positive emotions, 11 negative emotions, 4 ambiguous emotions, and neutral), we implemented a strict **pure-sentiment filter** (`classify_comment_sentiment`): a comment is retained if and only if **all** of its annotated emotion IDs map exclusively to either the positive group or the negative group. Mixed-sentiment, neutral, ambiguous, and duplicate comments are removed.

Applying this filter across the three official splits (`train.tsv`, `dev.tsv`, `test.tsv`) yielded **29,991 consistent binary comments**: **20,777 positive** (69.3%) and **9,214 negative** (30.7%). To eliminate majority-class bias and enable fair comparison on CPU hardware, we performed random undersampling with a fixed seed (`random_state = 42`), selecting **500 positive and 500 negative comments** (`data/processed/balanced_dataset.csv`). The balanced 1,000-comment corpus was partitioned via an **80/20 stratified split** (`seed = 42`) into **800 training** (`train.csv`) and **200 test** (`test.csv`) examples.

### 2.2 Exploratory Data Analysis and Text Cleaning (`notebooks/01_eda.ipynb`)
Exploratory inspection confirmed zero missing values and exact 50/50 class balance (`outputs/figures/class_distribution.png`). Raw comments have a mean length of **12.2 words** (median 12, maximum 29 words; `outputs/figures/text_length_distribution.png`). Our text-cleaning pipeline (`clean_text` and `tokenize_and_remove_stopwords`) converts text to lowercase, strips punctuation and special characters via regular expressions, and removes standard English NLTK stopwords as well as platform-specific Reddit noise (`[NAME]`, `[RELIGION]`, `reddit`, `edit`), producing the `cleaned_text` column (averaging ~7 cleaned tokens per comment). Word clouds (`outputs/figures/wordclouds.png`) and top-20 word frequency distributions (`outputs/figures/top_words.png`) reveal that sentiment polarity is heavily concentrated in high-salience lexical keywords (e.g., *love, good, happy, great, thank* vs. *hate, bad, sorry, worst, stupid*).

---

## 3. Model Architectures and Training Procedure

All models were trained and evaluated on the identical 800-train / 200-test stratified split (`seed = 42`).

### 3.1 Chance-Level Floor: Dummy Classifier Baseline (`notebooks/02_baseline_model.ipynb`)
To establish a minimum performance floor, we evaluated scikit-learn's `DummyClassifier` on the Bag-of-Words feature space using two strategies: `most_frequent` and `stratified` (`random_state = 42`). Both achieved **0.500 Accuracy** and a **Cohen's Kappa of 0.000** (`outputs/metrics/baseline_metrics.csv`, `outputs/figures/confusion_matrix_baseline.png`), mathematically confirming chance-level agreement on our balanced test split.

### 3.2 Classical Neural Models (`src/models.py`, `src/train.py`, `notebooks/03_dense_rnn_lstm.ipynb`)
For the classical neural branch, `cleaned_text` was vectorized using `CountVectorizer(max_features=5000, stop_words='english')`, yielding a **2,253-term vocabulary** fitted strictly on the 800 training samples. For sequential models (Vanilla RNN and LSTM), we mapped the same vocabulary to integer indices (`1..2253`, reserving `0` for padding/OOV) and left-padded or truncated each comment to **30 tokens**. All classical models were trained in PyTorch on CPU using binary cross-entropy with logits (`BCEWithLogitsLoss`), the `Adam` optimizer, and **Early Stopping** on validation loss, which restores the best checkpoint weights via `copy.deepcopy(model.state_dict())`.

| Architecture | Input Representation | Hidden Layers & Structure | Classification Head | Hyperparameters & Early Stopping |
| :--- | :--- | :--- | :--- | :--- |
| **Dense NN (BoW)** | 2,253-dim BoW vector (orderless) | `Linear(2253→256) + ReLU + Dropout(0.3)`<br>`Linear(256→128) + ReLU + Dropout(0.3)` | `Linear(128→1) + Sigmoid` | `lr=1e-3`, `batch=64`, max 20 epochs, `patience=4` (stopped at epoch 8; best val loss `0.5469` at epoch 4) |
| **Vanilla RNN** | 30-token sequence → `Embedding(2254, 64)` | 1-layer **Bidirectional** RNN (`hidden_dim=128` per direction) | Concat final states (`256`) + `Dropout(0.3) + Linear(256→1)` | `lr=1e-3`, `batch=64`, max 15 epochs, `patience=3` (stopped at epoch 12; best val loss `0.6720` at epoch 9) |
| **LSTM (Initial)** | 30-token sequence → `Embedding(2254, 64)` | 2 stacked **Bidirectional** LSTM layers (`hidden_dim=128`, `dropout=0.3`) | Concat final states (`256`) + `Dropout(0.3) + Linear(256→1)` | `lr=1e-3`, `batch=64`, max 15 epochs, `patience=3` (stopped at epoch 9; best val loss `0.6502` at epoch 6) |
| **LSTM (Tuned)** | 30-token sequence → `Embedding(2254, 64)` | 2 stacked **Bidirectional** LSTM layers (`hidden_dim=64`, `dropout=0.3`) | Concat final states (`128`) + `Dropout(0.3) + Linear(128→1)` | Grid-search best: `hidden=64`, `lr=1e-3`, `batch=32`, max 20 epochs, `patience=4` |

**LSTM Hyperparameter Grid Search**: Because the 2-layer bidirectional LSTM has the largest parameter footprint among the recurrent models, we executed an exhaustive 12-combination grid search over `hidden_dim ∈ {64, 128, 256}`, `lr ∈ {1e-3, 5e-4}`, and `batch_size ∈ {32, 64}` (10 epochs, `patience = 3`), ranking configurations by F1-score. The smallest capacity configuration (**`hidden_dim = 64, lr = 1e-3, batch_size = 32`**) ranked #1 (`Accuracy = 0.680, F1 = 0.6923, Kappa = 0.360`), whereas the largest capacity configuration (`hidden_dim = 256, lr = 5e-4, batch_size = 64`) collapsed to near-zero agreement (`Accuracy = 0.495, Kappa = -0.010`), confirming that high-capacity recurrent layers rapidly overfit our 800-comment training split.

### 3.3 Transformer Extension: DistilBERT (`notebooks/04_transformer_extension.ipynb`)
To evaluate contextual self-attention and transfer learning, we implemented `TransformerSentiment` (`src/models.py`), fine-tuning pre-trained **`distilbert-base-uncased`** (66M parameters, 6 layers, 12 attention heads). Raw comments are tokenized with `DistilBertTokenizerFast` using WordPiece subwords padded/truncated to **`max_length = 64`** (covering the 95th percentile of 33 subword tokens; `outputs/figures/04_sequence_length_distribution.png`). The model extracts the `[CLS]` token representation (`last_hidden_state[:, 0, :]`) and passes it through a two-layer head (`Dropout(0.3) → Linear(768→384) → ReLU → Dropout(0.3) → Linear(384→2)`). We trained for **5 epochs** (`batch_size = 16`, `AdamW` with `lr = 2e-5`, cross-entropy loss, gradient norm clipping at `1.0`), selecting the checkpoint with peak validation accuracy (achieved at Epoch 5 with `Train Acc = 0.9788`, `Val Acc = 0.7750`).

---

## 4. Experimental Evaluation and Comparative Analysis

All models were evaluated on the held-out 200-comment stratified test split (`src/evaluate.py`). Table 1 consolidates the test metrics across all configurations (`outputs/metrics/baseline_metrics.csv`, `outputs/metrics/model_comparison.csv`, `outputs/metrics/lstm_tuned_metrics.json`, and `outputs/metrics/transformer_metrics.json`).

**Table 1: Consolidated Test Set Performance (200 Samples: 100 Negative, 100 Positive)**

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Cohen's $\kappa$ | Confusion Matrix (`[[TN, FP], [FN, TP]]`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | `[[100, 0], [100, 0]]` |
| **Dummy (Stratified)** | 0.500 | 0.500 | 0.510 | 0.505 | 0.000 | `[[49, 51], [49, 51]]` |
| **Dense NN (BoW)** | **0.745** | **0.738** | **0.760** | **0.749** | **0.490** | `[[73, 27], [24, 76]]` |
| **Vanilla RNN (Bi)** | 0.550 | 0.536 | 0.740 | 0.622 | 0.100 | `[[36, 64], [26, 74]]` |
| **LSTM (Initial, 15 ep)** | 0.555 | 0.580 | 0.400 | 0.473 | 0.110 | `[[71, 29], [60, 40]]` |
| **LSTM (Tuned Final)** | 0.525 | 0.520 | 0.660 | 0.582 | 0.050 | `[[39, 61], [34, 66]]` |
| **DistilBERT (Extension)** | **0.775** | **0.762** | **0.800** | **0.780** | **0.550** | `[[75, 25], [20, 80]]` |

### 4.1 Training vs. Test Dynamics and Model Selection Justification
1. **Why DistilBERT is the Best Overall Model**: DistilBERT achieves the highest score on every evaluation criterion (**0.775 Accuracy, 0.780 F1-score, and $\kappa = 0.550$**, representing moderate-to-substantial agreement beyond chance), correctly classifying 155 of 200 test comments. Even though training accuracy reached 97.9% by Epoch 5 while validation loss began to rise (`0.5072` at Epoch 2 to `0.8308` at Epoch 5; `outputs/figures/04_transformer_training_curves.png`), its pre-trained subword representations and bidirectional self-attention generalize substantially better to unseen informal Reddit syntax than any model trained from scratch.
2. **Why Dense NN (BoW) Outperforms Recurrent Models (RNN & LSTM)**: While standard theory predicts `LSTM > Vanilla RNN > Dense NN`, our empirical ranking is **`Dense NN (κ = 0.490) >> Vanilla RNN (κ = 0.100) ≈ LSTM (κ = 0.110 initial / 0.050 retrained)`**. Inspection of the training curves (`outputs/figures/dense_training_curves.png` vs. `rnn_training_curves.png` and `lstm_training_curves.png`) explains why:
   - **Lexical locality in short texts**: Because GoEmotions comments average ~7 cleaned tokens, sentiment polarity is determined primarily by isolated keywords rather than long-range compositional dependencies. The Dense BoW network receives all 2,253 keyword indicators simultaneously in parallel, reaching its optimal validation loss (`0.5469`) by Epoch 4.
   - **Embedding cold-start and sample scarcity in recurrent models**: Both the Vanilla RNN and the 2-layer bidirectional LSTM must jointly learn a 2,254 × 64 embedding matrix and recurrent transition matrices from only 800 training examples. As a result, validation loss stagnates near `0.65–0.67`, and both models exhibit severe class-boundary instability — the Vanilla RNN and retrained LSTM collapse toward predicting the positive class (`FP = 64` and `FP = 61`, Recall `0.74` and `0.66`), whereas the initial LSTM run biased toward the negative class (`TN = 71, TP = 40`, F1 `0.473`). Furthermore, sensitivity to random weight initialization on 800 samples caused the 20-epoch LSTM retraining (`patience = 4`) to restore an early checkpoint with lower generalization ($\kappa = 0.050$) than the 10-epoch grid-search trial ($\kappa = 0.360$).

---

## 5. Relation to Turing Machine Concepts (OE3.3–OE3.6)

The architectural differences observed above correspond directly to formal models of computation:

### 5.1 Recurrent Networks as Sequential Tape Automata
Following **Siegelmann and Sontag (1992)**, a Recurrent Neural Network operates analogously to a classical **Turing Machine (TM)**:
- **Internal State and Working Tape ($Q, \Gamma$)**: At time step $t$, the hidden state vector $h_t \in \mathbb{R}^d$ acts as a compressed numerical encoding of both the control state and the tape contents summarizing the prefix $x_1, \dots, x_t$.
- **Transition Function ($\delta$)**: The recurrent update rule
  $$h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)$$
  corresponds to the state-transition function $\delta(q, \sigma)$, reading one input symbol per step in **$O(n)$ sequential operations**.
- **Gated Memory in LSTMs**: In a vanilla RNN, repeated application of $W_h$ over $n$ steps causes exponential gradient growth or decay, effectively erasing early tape symbols. The **LSTM** (Hochreiter & Schmidhuber, 1997) introduces an explicit memory register — the cell state $C_t$ — regulated by forget ($f_t$), input ($i_t$), and output ($o_t$) gates:
  $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t, \qquad h_t = o_t \odot \tanh(C_t).$$
  These gates emulate selective **read, write, and erase tape operations**, preserving state information across longer sequences.

### 5.2 Transformers, Parallel Memory Access, and Bounded Computability
- **Breaking Sequential Tape Traversal**: Unlike recurrent models, the **Transformer** (Vaswani et al., 2017) dispenses with sequential state recurrence. Scaled dot-product self-attention,
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V,$$
  grants every token direct content-addressable access to every other position in the input window in **$O(1)$ sequential path length** ($O(n^2)$ parallel operations), eliminating the sequential bottleneck of a moving tape head.
- **Theoretical Turing Completeness vs. Bounded Finite-State Automata**: Under idealized mathematical assumptions — unbounded rational precision and infinite context length — both RNNs (Siegelmann & Sontag, 1992) and Transformers with hard attention and positional encodings (Pérez et al., 2021; Yun et al., 2020) are **Turing-complete**. However, as shown by Hahn (2020), fixed-depth soft self-attention with bounded precision cannot recognize certain formal languages (such as unbounded PARITY or DYCK-2) as length scales. In our actual PyTorch implementations, every architecture operates under **32-bit IEEE-754 floating-point precision** (`float32`), fixed parameter dimensions, and strict sequence truncation (**30 tokens** for RNN/LSTM; **64 tokens** for DistilBERT). Consequently, all implemented models are formally **bounded Finite State Automata** rather than unrestricted Turing Machines.

---

## 6. Conclusions and Future Work

### 6.1 Conclusions
1. **Superiority of Pre-trained Self-Attention**: Fine-tuned **DistilBERT** achieved the best overall performance (**Accuracy: 0.775, F1: 0.780, $\kappa = 0.550$**), demonstrating that pre-trained subword representations combined with $O(1)$ parallel self-attention effectively handle informal, noisy Reddit comments.
2. **Bag-of-Words Efficiency on Short, Small-Scale Corpora**: On 800 short comments (~13 words), the **Dense NN (BoW)** (**Accuracy: 0.745, F1: 0.749, $\kappa = 0.490$**) decisively outperformed both **Vanilla RNN** ($\kappa = 0.100$) and **LSTM** ($\kappa = 0.050–0.110$). When sentiment is governed by localized keywords and training data is limited, direct parallel feature access avoids the optimization and overfitting pitfalls of training sequential transition gates from scratch.
3. **Practical Scope**: All reported metrics characterize the balanced 1,000-comment subset (`seed = 42`) and should not be extrapolated to the complete 29,991-comment filtered GoEmotions corpus without training at full scale.

### 6.2 Future Work
1. **Pre-trained Word Embeddings for Recurrent Models**: Initializing the Vanilla RNN and LSTM embedding layers with pre-trained **GloVe** or **FastText** vectors (and freezing them during early epochs) would decouple sequential recurrence from embedding cold-start on small data.
2. **Full-Corpus Training and Class-Weighted Loss**: Training all architectures on the complete 29,991-comment filtered GoEmotions dataset using class-weighted cross-entropy or focal loss would provide sufficient sample diversity for the LSTM's gated memory to learn compositional patterns (such as negation and sarcasm).
3. **Explainability via Attention Visualization**: Extracting DistilBERT attention weights and comparing them against Dense BoW feature importance (e.g., via Integrated Gradients or SHAP) would empirically verify which contextual tokens drive classification decisions.

---

## 7. References

1. Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020), 4040–4054. DOI: 10.18653/v1/2020.acl-main.372.
2. Hahn, M. (2020). *Theoretical Limitations of Self-Attention in Neural Sequence Models*. Transactions of the Association for Computational Linguistics (TACL), 8, 155–171.
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
4. Landis, J. R., & Koch, G. G. (1977). *The Measurement of Observer Agreement for Categorical Data*. Biometrics, 33(1), 159–174.
5. Pérez, J., Barceló, P., & Marinkovic, J. (2021). *Attention Is Turing Complete*. Journal of Machine Learning Research (JMLR), 22(75), 1–53.
6. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv:1910.01108.
7. Siegelmann, H. T., & Sontag, E. D. (1992). *On the computational power of neural nets*. Proceedings of the 5th Annual Workshop on Computational Learning Theory (COLT), 282–290.
8. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS), 30.
9. Yun, C., Bhojanapalli, S., Rawat, A. S., Reddi, S. J., & Kumar, S. (2020). *Are Transformers Universal Approximators of Sequence-to-Sequence Functions?* International Conference on Learning Representations (ICLR).
