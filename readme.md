# Assignment 2: Contextualized Vectors, Parts of Speech, and Named Entities

## Overview
This project involves working with contextualized (sentence-dependent) vectors produced by sentence encoding models, specifically the RoBERTa model. The tasks include parts-of-speech (POS) tagging and named entity recognition (NER) using these vectors.

## Project Structure

### 0 Warmup
The HuggingFace Transformers library provides numerous sentence encoders. For this assignment, we use the “roberta-base” model, a masked language model. The initial tasks involve:

1. **Installing the HuggingFace Transformers library:**
   ```sh
   pip install transformers
   ```
2. **Encoding a sample sentence "I am so <mask>":**
   - Extract vectors for "am" and "<mask>".
   - Extract the top-5 word predictions for "am" and "<mask>" and their probabilities.
3. **Exploring Word Similarity:**
   - Find sentences with high cosine similarity for shared words.
   - Find sentences with low cosine similarity for shared words.
4. **Tokenization Analysis:**
   - Identify sentences where the number of tokens exceeds the number of words.

### 1 Part-of-Speech Tagging

#### Task
Use the training set to predict parts-of-speech (POS) for words in the test set without tuning parameters or training any model. Three approaches are explored:

1. **No Word Vectors:**
   - Utilize statistical analysis of word and tag frequencies in the training corpus.
   - Predict POS tags based on the most frequent tag for each word.
   - Handle unknown words by assigning the 'NN' (noun) tag.

2. **Static Word Vectors:**
   - Integrate Word2Vec embeddings to improve predictions, especially for words not in the training set.
   - Use similarity-based inference to assign tags based on the nearest neighbor's tag.

3. **Contextualized Word Vectors:**
   - Employ the RoBERTa-base model to provide contextually relevant predictions.
   - Use masked language modeling to infer tags for out-of-vocabulary (OOV) words.
   - Combine frequency data and static vectors for enhanced accuracy.

### 2 Named Entities Recognition (NER)

#### Task
Predict named entities using annotated data without training any classifier. The following approaches were used:

1. **Frequency-Based Approach:**
   - Predict entities based on the most frequent tag for each word in the training set.

2. **Incorporation of Word2Vec:**
   - Enhance predictions by finding similar words using Word2Vec embeddings and inferring tags from their nearest neighbors.

3. **Utilization of RoBERTa:**
   - Generate contextually relevant predictions using the RoBERTa model for OOV words.
   - Predict likely word replacements and determine the most common entity tag among these predictions.

## Results and Discussion

### Part-of-Speech Tagging

#### No Word Vectors
- **Accuracy:** 90.47%
- **Methodology:** Frequency analysis of word-tag pairs, defaulting to 'NN' for unknown words.

#### Static Word Vectors
- **Accuracy:** 90.90%
- **Methodology:** Integration of Word2Vec embeddings, leveraging semantic similarity for tag inference.

#### Contextualized Word Vectors
- **Accuracy:** 92.11%
- **Methodology:** Utilization of RoBERTa for contextually relevant predictions, combined with frequency data.

### Named Entities Recognition

#### Frequency-Based Approach
- **Accuracy:** 93.88%
- **Precision (All-types):** 71.66%
- **Recall (All-types):** 65.95%

#### Word2Vec
- **Accuracy:** 94.53%
- **Precision (All-types):** 71.33%
- **Recall (All-types):** 69.74%

#### RoBERTa
- **Accuracy:** 95.27%
- **Precision (All-types):** 72.27%
- **Recall (All-types):** 74.77%

## Submission Instructions
- **POS Predictions:** `POS_preds_X.txt` (X being 1, 2, or 3)
- **NER Predictions:** `NER_preds.txt`





## Conclusion
This assignment demonstrates the application of contextualized word vectors and various approaches to parts-of-speech tagging and named entity recognition, highlighting the effectiveness of advanced models like RoBERTa in capturing contextual nuances for improved accuracy.

## Directory Structure
The project directory contains the following files:


- **0.py:** Contains the warmup tasks using the RoBERTa model.
- **1.py:** Implements the POS tagging without using any word vectors.
- **2.py:** Implements the POS tagging with static word vectors and contextualized word vectors.
- **NER_preds.txt:** Contains the NER predictions.
- **POS_preds_1.txt:** Contains the POS predictions without word vectors.
- **POS_preds_2.txt:** Contains the POS predictions with static word vectors.
- **POS_preds_3.txt:** Contains the POS predictions with contextualized word vectors.
