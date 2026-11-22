# CED-III-IT2-JEHEJEHEJEHE

## Sentiment Analysis on GoEmotions Dataset

Academic machine learning project for sentiment analysis using a balanced subset of the GoEmotions dataset.

## Team Members

- Valeria Piza Saavedra - A00405037
- Krystal Angely Giraldo Ramirez - A00407967
- Miguel Jose Pabon Benavidez - A00406585

## Status: In Progress

The GoEmotions preprocessing pipeline and the DistilBERT implementation are complete. The filtered dataset contains 29,991 consistent positive or negative comments: 9,214 negative and 20,777 positive. From that dataset, 500 comments were sampled from each class using random seed 42, producing a 1,000-comment balanced dataset.

The DistilBERT run is recorded with accuracy 0.775, precision 0.762, recall 0.800, F1 0.780, and Cohen's kappa 0.550. The dummy, dense, RNN, and LSTM experiments remain pending.
