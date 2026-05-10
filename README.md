# Churn Prevention System

## Project Overview

This project develops a hybrid churn-aware recommender system for e-commerce customer retention.
The system combines:

- Customer churn prediction using LightGBM
- Customer lifetime modelling using BG/NBD
- Collaborative filtering retrieval using ALS
- Approximate nearest neighbour search with FAISS
- Deep & Wide ranking model implemented in PyTorch

The objective is to identify customers at risk of churn and recommend relevant products to improve engagement and retention.

---

## System Architecture

The recommendation pipeline consists of:

1. **Churn Prediction Layer**

   * LightGBM classifier predicts customer churn probability

2. **Customer Lifetime Value Layer**

   * BG/NBD model estimates:

     * probability customer is still active (`p_alive`)
     * expected future purchases

3. **Retrieval Layer**

   * ALS collaborative filtering retrieves candidate products
   * FAISS performs efficient similarity search

4. **Ranking Layer**

   * Deep & Wide model reranks candidate products using:

     * ALS score
     * FAISS similarity score
     * churn probability
     * behavioural features
     * customer lifetime features

5. **Recommendation API**

   * Returns top-N personalised product recommendations

---

## Dataset

The original dataset used in this project is the Online Retail II dataset from UCI, available on Kaggle:

[Online Retail II Dataset on Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci?utm_source=chatgpt.com)

Due to GitHub file size limitations, the full raw transactional dataset is not included in this repository.

A smaller sample dataset is provided for demonstration purposes.

### Dataset Features

* Customer transactions
* Product stock codes
* Invoice dates
* Quantity purchased
* Unit price
* Country information

---

## Methodology

Due to project scope and time constraints, the temporal evaluation was implemented using a train-test split rather than a full train-validation-test framework.

### 1. Churn Prediction

A LightGBM classifier was trained using customer behavioural features such as:

* Recency
* Frequency
* Monetary value
* Average basket size
* Purchase intervals

The model predicts the probability of customer churn.

---

### 2. Customer Lifetime Modelling

The BG/NBD model was used to estimate:

* Probability customer remains active (`p_alive`)
* Expected number of future purchases

These features were later incorporated into the recommendation ranking model.

---

### 3. Candidate Retrieval

An ALS (Alternating Least Squares) collaborative filtering model was trained on historical customer-item interactions.

The model generates top-K candidate products for each customer.

Evaluation metrics:

* Recall@K
* NDCG@K

---

### 4. Similarity Search with FAISS

FAISS was used to efficiently retrieve similar item embeddings from the ALS latent factor space.

The FAISS similarity score was used as an additional ranking feature.

---

### 5. Deep & Wide Ranking Model

A Deep & Wide neural network was trained to rerank candidate products.

Input features include:

* ALS retrieval score
* FAISS similarity score
* Churn probability
* BG/NBD outputs
* Customer behavioural features

The model outputs a final ranking score for each candidate item.

---

## Evaluation

### Retrieval Model

Metrics used:

* Recall@10
* Recall@50
* NDCG@10
* NDCG@50

### Ranking Model

The ranking model was evaluated using a held-out user split to assess recommendation quality on unseen customers.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* PyTorch
* Implicit ALS
* FAISS
* Lifetimes
* Jupyter Notebook

---
## How to Run

1. Clone the repository:

```bash
git clone https://github.com/PhuongAnhDuong/churn_prevention_system.git
cd churn_prevention_system
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the original dataset from Kaggle and place it in the `data/` folder:

https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

4. Open and run the main notebook:

```text
churn_prevention_system.ipynb
```

5. To run the API locally:

```bash
cd api
uvicorn app:app --reload
```
---

## Future Improvements

Potential future improvements include:

* Real-time recommendation serving
* Session-based recommendation models
* Transformer-based ranking models
* Online learning pipelines
* Explainable recommendation systems

---

## Author

Phuong Anh Duong Thi
