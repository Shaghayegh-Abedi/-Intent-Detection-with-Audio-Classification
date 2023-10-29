# Intent-Detection-with-Audio-Classification

Intent detection in natural language processing (NLP) is a classification task that involves identifying the intent behind a user’s input to determine the appropriate action to take in response. In this paper, a model has been trained on a labeled dataset to accurately categorize the user’s spoken words and establish their intended meaning. In the end, it should be possible for the model to forecast both the requested action and the item that it will impact. 

In order to predict audio intent using the collected features, this study seeks to develop and evaluate two classifiers: **SVM classifier and CNN classifier**.
 The suggested method can accurately categorize up to 58.8% of voices in a dataset of roughly **10k** samples. CNN yields 38% accuracy.

```SVM.py``` is Python implementation for SVM, including:
```
1. Preprocessing
  - Trimming silent areas
  - Adding length by padding
  - Turning into Mel-Frequency Cepstral Coefficients (MFCC)
  - Dimension reduction
  - Scaling the result
2. Model selection
3. Model Test
```
- [ ] Add ANN to test higher accuracy :tada:
