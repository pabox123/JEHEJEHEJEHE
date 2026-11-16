# Sentiment Analysis on Social Media Text: A Comparative Study of Neural Architectures

## 1. Introduction
Sentiment analysis is a fundamental task in Natural Language Processing (NLP) that involves identifying and extracting subjective information from text. Social media platforms, such as Reddit, present unique challenges for sentiment analysis due to the informal nature of the language, the presence of slang, irony, sarcasm, and the fact that a single comment can express multiple, sometimes conflicting, emotions. 

The objective of this project is to build and compare supervised learning models for binary sentiment classification (positive vs. negative) on short, informal textual data. Specifically, we implement and evaluate a Dummy Classifier (baseline), a Dense Neural Network, a vanilla Recurrent Neural Network (RNN), and a Long Short-Term Memory (LSTM) network. Furthermore, we explore a Transformer-based extension (DistilBERT) to analyze how attention mechanisms differ from recurrent memory-based computation. 

Finally, this project connects these modern neural architectures to the theoretical foundations of computation, specifically relating their memory structures and sequence processing capabilities to the concepts of Turing Machines, computability, and formal languages (aligning with learning objectives OE3.3 to OE3.6).

## 2. Dataset Description
The dataset used in this project is **GoEmotions**, a corpus of English Reddit comments created by Google Research and introduced by Demszky et al. (2020) in the Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020). 

- **Source**: [Kaggle - GoEmotions](https://www.kaggle.com/datasets/debarshichanda/goemotions)
- **Original Size**: 58,009 comments labeled with 27 fine-grained emotion categories or Neutral.
- **Characteristics**: Comments are short (average ~13 words, max 30 tokens). Person names and religion terms are anonymized using the mask tokens `[NAME]` and `[RELIGION]`.
- **Filtered Version**: We utilize the authors' filtered version (54,263 comments), which includes only examples where at least two human raters agreed on the label.

The 27 emotions are mapped into three main sentiment groups: **Positive** (12 emotions, e.g., joy, admiration), **Negative** (11 emotions, e.g., anger, disappointment), and **Ambiguous** (4 emotions, e.g., confusion, surprise). For this binary classification task, we strictly isolate the pure positive and pure negative classes.

## 3. Preprocessing Steps
To ensure data quality and model performance, we applied a rigorous, multi-step preprocessing pipeline:

1. **Sentiment Mapping & Filtering**: Using the official GoEmotions mapping, we assigned a sentiment group to each comment. We retained *only* comments where all annotated emotions belonged to the same group (pure positive or pure negative), explicitly filtering out neutral, ambiguous, and mixed-sentiment comments.
2. **Deduplication**: Removed duplicate comments to prevent data leakage and overfitting.
3. **Balancing**: Constructed a perfectly balanced dataset by randomly sampling exactly 500 positive and 500 negative comments, using a fixed random seed (`random_state=42`) to ensure reproducibility.
4. **Text Cleaning**: 
   - Converted all text to lowercase.
   - Removed punctuation and special characters.
   - Removed standard English stopwords using the NLTK library.
   - **Custom Reddit Filtering**: Extended the stopword list to remove platform-specific noise (e.g., "u", "reddit", "edit", "thanks") and explicitly handled the `[NAME]` and `[RELIGION]` mask tokens to prevent them from skewing the vocabulary.
5. **Tokenization & Length Calculation**: Split the cleaned text into tokens and calculated the text length (in words) for Exploratory Data Analysis (EDA).
6. **Train/Test Split**: Performed a stratified 80/20 split (800 training samples, 200 testing samples) to maintain the 50/50 class distribution in both sets.

The final processed dataset is stored in the `data/processed/` directory, ensuring a clean separation between raw data and model-ready inputs.