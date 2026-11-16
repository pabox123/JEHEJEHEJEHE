---
marp: true
theme: default
paginate: true
---

# Sentiment Analysis on Reddit Comments
## A Comparative Study of Neural Architectures and Turing Computability

**CED-III-IT2 | Integrative Task 2**  
Miguel, Krystal, Valeria  
Universidad Icesi | November 2026

---

## Problem Statement and Objectives

**Challenge**: Classify sentiment (positive vs. negative) in short, noisy Reddit comments where language is informal, ironic, and emotionally charged.

**Objectives**:
- Implement and compare multiple neural architectures: Dense NN, RNN, LSTM, and Transformer
- Evaluate performance using accuracy, precision, recall, F1-score, and Cohen's kappa
- Connect architectures to Turing Machine concepts: memory, sequence processing, computability (OE3.3-OE3.6)

---

## Dataset: GoEmotions

**Source**: GoEmotions dataset (Demszky et al., ACL 2020)
- 58,009 Reddit comments with 27 fine-grained emotion labels
- Official mapping to 3 sentiment groups: positive, negative, ambiguous

**Our Subset**:
- Filtered to positive and negative only
- Balanced: 500 positive + 500 negative comments
- Random seed = 42 for reproducibility
- Average comment length: ~13 words

**Example comments**:
- Positive: *"A beautiful poem for a beautiful thing. Happy cake day btw!"*
- Negative: *"Just watching this happen upsets me ~_~"*

---

## Preprocessing Pipeline

1. **Filtering**: Applied official emotion-to-sentiment mapping
2. **Balancing**: Random sampling with fixed seed
3. **Text cleaning**: Lowercasing, punctuation removal, stopword removal
4. **Tokenization**:
   - Neural models: Word-level tokenization with embeddings
   - Transformer: `DistilBertTokenizerFast` (WordPiece)
5. **Padding/Truncation**: Fixed max length (64 tokens for neural, 128 for Transformer)

---

## Methodology: Model Architectures

**Baseline**: DummyClassifier (stratified) → minimum performance threshold

**Neural Models** (Planned):
- **Dense NN**: MLP with pre-computed embeddings, ReLU, dropout
- **Vanilla RNN**: Sequential hidden state updates: $h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)$
- **LSTM**: Gating mechanisms (forget, input, output) to retain long-term dependencies

**Extension** (Implemented):
- **DistilBERT**: Fine-tuned Transformer (66M parameters), self-attention, parallel processing

---

## Transformer Extension: Implementation

**Architecture**:
- Pre-trained `distilbert-base-uncased`
- Classification head: Dropout → Linear → ReLU → Dropout → Linear
- Uses [CLS] token representation for classification

**Training Configuration**:
| Parameter | Value |
| :--- | :--- |
| Epochs | 5 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Optimizer | AdamW |
| Loss | Cross-entropy |
| Gradient clipping | Max norm 1.0 |
| Train/test split | 80/20 stratified |

---

## Results: Transformer Performance

**Test Set Metrics** (120 samples):

| Metric | Score |
| :--- | :---: |
| Accuracy | 1.000 |
| Precision | 1.000 |
| Recall | 1.000 |
| F1-Score | 1.000 |
| Cohen's Kappa | 1.000 |

**Important Note**: Perfect score may reflect synthetic development dataset. Full evaluation pending with complete GoEmotions subset.

**Confusion Matrix**: Perfect classification on test set (no false positives or false negatives)

---

## Connection to Turing Machines (1/2)

**RNNs as Turing Machines** (Siegelmann & Sontag, 1992):
- Hidden state $h_t$ ≈ TM's **tape** (memory)
- Weight matrices ≈ **transition function** $\delta$
- Theoretically Turing-complete with infinite precision

**LSTM and Memory**:
- Cell state $C_t$ = "continuous tape" with explicit read/write/erase
- Gates control information flow → better approximation of unbounded TM memory
- Mitigates vanishing gradients (practical limitation of finite precision)

---

## Connection to Turing Machines (2/2)

**Transformers and Computability**:
- Break sequential state-transition constraint of RNNs
- Self-attention: $O(1)$ access to any position (vs. $O(n)$ for RNN tape head)
- Parallel processing during training
- **Practical limitation**: Bounded by fixed context window and finite precision
- Technically bounded Finite State Automata, but approach Turing completeness with infinite resources

**Key Insight**: Transformers overcome sequential computational bottlenecks of classical recurrent models → better generalization on complex, noisy language

---

## Comparative Analysis (Theoretical)

| Model | Memory | Sequence Processing | Interpretability |
| :--- | :--- | :--- | :--- |
| Dense NN | None (bag-of-words) | None | High |
| RNN | Hidden state $h_t$ | Sequential $O(n)$ | Low |
| LSTM | Cell state $C_t$ + gates | Sequential $O(n)$ | Low |
| Transformer | Attention over full context | Parallel $O(1)$ access | Medium (attention weights) |

**Performance Hierarchy** (Expected):
Transformer > LSTM > RNN > Dense NN > Dummy

**Trade-offs**: Training complexity, interpretability, and computational cost increase with model sophistication


---

## References

1. Demszky et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *ACL 2020*.
2. Hochreiter & Schmidhuber (1997). Long Short-Term Memory. *Neural Computation*.
3. Sanh et al. (2019). DistilBERT: smaller, faster, cheaper and lighter. *arXiv:1910.01108*.
4. Siegelmann & Sontag (1992). On the computational power of neural nets. *COLT*.
5. Vaswani et al. (2017). Attention Is All You Need. *NeurIPS*.

---
