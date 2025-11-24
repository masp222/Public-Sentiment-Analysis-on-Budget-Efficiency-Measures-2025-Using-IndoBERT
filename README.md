# 📘 Public Sentiment Analysis on Prabowonomics (2025)

### TF-IDF vs. IndoBERT Embeddings for Indonesian Social Media



---

## 📌 Overview

This repository contains the implementation of a **sentiment analysis system** evaluating public reactions toward **Prabowonomics budget efficiency policies** proposed for Indonesia in 2025. The project compares **TF-IDF** and **IndoBERT embeddings** combined with traditional machine learning models.

The study aims to understand how Indonesians perceive major budget cuts (Rp 306 trillion), using social media comments as real-time sentiment indicators.


---

## 🎯 Objectives

* Compare **traditional features (TF-IDF)** with **contextual embeddings (IndoBERT)**
* Evaluate multiple ML classifiers for Indonesian sentiment classification
* Identify which method performs best for policy-related social media text
* Provide insights for NLP practitioners & policymakers


---

## 📂 Dataset

Data was collected from **YouTube** and **Twitter** using targeted keywords related to:

* “kebijakan efisiensi”
* “kebijakan Prabowo”

Total samples: **3,167**

* 1,523 comments (YouTube Source 1)
* 1,298 comments (YouTube Source 2)
* 344 posts (Twitter)

(Table I — Page 3) 

---

## 🏷️ Sentiment Labels

Three labels were manually assigned:

* **Positive**
* **Negative**
* **Neutral**

Two annotators labeled the data and resolved mismatches through discussion.
Label distribution (Fig. 2 — Page 3):

* **Negative: 63%**
* **Positive: 22%**
* **Neutral: 15%**


---

## 🧹 Preprocessing Steps

The following pipeline cleans noisy Indonesian social media text:

1. Tokenization
2. Removal of mentions/links
3. Lowercasing
4. Slang & abbreviation normalization (custom dictionary)
5. Stopword removal (Sastrawi)
6. Stemming
   (Table II — Page 3)


---

## 🔤 Feature Extraction Methods

### **1. TF-IDF**

* Classical sparse vector representation
* Captures term importance using:
  `TF * log(N/DF)`
* High-dimensional


### **2. IndoBERT Embeddings**

* Contextual, bidirectional sentence representations
* Pretrained on the **Indo4B corpus** (news, social media, subtitles)
* Provides richer semantic understanding for Indonesian text


---

## ⚖️ Handling Class Imbalance

Since negative sentiment dominates the dataset (63%), the study applies:

### **SMOTE oversampling (after embedding)**

SMOTE is applied **on numerical features**, not raw text, to preserve semantic relationships.


---

## 🧠 Machine Learning Models Tested

The following five ML classifiers were tested using both TF-IDF and IndoBERT embeddings:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Gradient Boosting
* Decision Tree

Each model is evaluated across Accuracy, Precision, Recall, and F1-Score.


---

## 📊 Results

### **TF-IDF Results (Table IV — Page 5)**

Best models:

| Model               | Accuracy  | F1    |
| ------------------- | --------- | ----- |
| **GradientBoost**   | **65.4%** | 0.646 |
| Logistic Regression | 63.7%     | 0.646 |

---

### **IndoBERT Results (Table V — Page 5)**

Best models:

| Model                        | Accuracy  | F1        |
| ---------------------------- | --------- | --------- |
| **Random Forest (IndoBERT)** | **70.3%** | **0.672** |
| GradientBoost (IndoBERT)     | 65.8%     | 0.659     |

IndoBERT significantly boosts Random Forest performance (+9% over TF-IDF).


---

## 🎯 Key Insights

* **IndoBERT + Random Forest** achieved the highest accuracy (**70.3%**)
* Neutral sentiment remains the hardest class (recall < 55% across models)
* TF-IDF still performs competitively due to Indonesian’s simpler morphology
* IndoBERT provides richer semantic context that improves classification


---


## 📌 Future Directions

* Improve neutral sentiment detection with attention-based models
* Try fine-tuning IndoBERT instead of using frozen embeddings
* Explore hybrid embeddings (TF-IDF + BERT)
* Deploy model as an API (FastAPI or Flask)

---

## 🤝 Authors

* Mirekel Tjoa
* Mochammad Aqsa Sandhy Pradipta
* Alexander Agung Santoso Gunawan
* Rilo Chandra Pradana


