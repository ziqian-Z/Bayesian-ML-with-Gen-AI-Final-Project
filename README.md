# Bayesian Modeling for Uncertainty-Aware Credit Risk Decisions

## Application Area
Finance / Risk Management


# Project Overview

This project studies borrower default risk prediction using large-scale credit data. The objective is to estimate the probability that a borrower will default on a loan and use these probabilities to support lending decisions such as:

- Loan approval or rejection  
- Loan amount allocation  
- Interest-rate tiering  

Rather than treating credit risk as a deterministic classification problem, we frame it as an uncertain probabilistic inference problem, where each borrower is assigned a probability of default.

These probability estimates allow lenders to make risk-aware decisions under asymmetric costs, where approving a risky borrower may lead to significant losses while rejecting a good borrower leads to opportunity cost.


# Task Description

The primary modeling task is binary classification of loan outcomes, where the model estimates:

[
P(\text{default} = 1 \mid \text{borrower features})
]

The model outputs the probability of default rather than a hard decision boundary. This probabilistic approach allows downstream decision rules to be adjusted depending on the lender’s risk tolerance.

Key modeling goals include:

- Predict borrower default probability
- Evaluate model calibration and predictive performance
- Compare predictive models in terms of accuracy and uncertainty awareness



# Data Type

The project uses structured tabular data (.csv) containing borrower characteristics, loan attributes, and repayment outcomes.

Possible sources for credit-risk data include:

### Public Loan-Level Credit Datasets
- LendingClub Loan Data

### Credit Default Datasets with Repayment Behavior
- UCI Credit Card Default Dataset

### Synthetic Data
- Simulated repayment trajectories for validation and controlled experiments



# Dataset

This project primarily uses the Lending Club Loan Dataset, obtained from Kaggle.

Lending Club is a U.S.-based peer-to-peer lending platform that connects individual investors with borrowers. Investors provide capital while borrowers repay loans with interest over time.

The dataset contains:

- ~890,000 loan observations
- ~75 borrower and loan-level variables
- Historical loan performance data

Key feature groups include:

### Borrower Demographics
- income
- employment length
- home ownership status
- geographic location

### Credit Characteristics
- credit history length
- delinquency history
- credit inquiries
- credit utilization ratios

### Loan Attributes
- loan amount
- interest rate
- loan grade
- loan purpose

### Repayment Outcomes
Loan performance indicators such as:

- Fully Paid
- Current
- Late
- Charged Off
- Default

These repayment status indicators allow us to construct a binary default variable used for modeling.

The dataset reflects real-world credit performance and is widely used for research in credit risk modeling.



# Methodology

## Model

The primary predictive model used in this project is **TabPFN**, a transformer-based model designed for tabular classification tasks.

TabPFN (Tabular Prior-Data Fitted Network):

- Uses a transformer architecture
- Is pretrained on large numbers of synthetic tabular tasks
- Approximates Bayesian posterior predictions for tabular datasets

The model outputs probability estimates for binary outcomes, making it well suited for default risk prediction.

The pretrained model is obtained from an open-source repository:

TabPFN repository

https://huggingface.co/Prior-Labs/TabPFN-v2-clf

In this project, the pretrained TabPFN model is fine-tuned using the Lending Club dataset, allowing it to adapt to the structure and patterns present in real-world credit data.

This approach provides:

- strong predictive performance
- minimal feature engineering requirements
- probabilistic output suitable for credit risk analysis



## Data Sampling Strategy

The raw Lending Club dataset contains over 2 million loan records after data cleaning and preprocessing.

Training the TabPFN model on the full dataset was computationally expensive and required substantial GPU memory. To ensure efficient experimentation and reproducibility, we implemented a staged sampling strategy.

### Stage 1: Initial Sampling

We first randomly sampled:

50,000 observations

from the cleaned dataset to create a manageable working dataset.

### Stage 2: Reduced Training Sample

During model experimentation, we observed that training with 50,000 observations was still computationally slow for iterative experimentation and model tuning.

Therefore, we further reduced the training sample to:

20,000 observations

This smaller dataset allows faster training and model evaluation while still preserving sufficient data diversity to capture meaningful credit risk patterns.

Sampling was performed randomly while maintaining the original class distribution of default outcomes.



# Feature Engineering

Key preprocessing steps include:

- Removal of post-origination variables to prevent data leakage
- Handling missing values using domain-informed imputation strategies
- Aggregation of high-cardinality categorical variables
- One-hot encoding of categorical features for tabular model input
- Creation of derived features such as credit behavior indicators

These steps ensure that the model only uses information available at the time of loan underwriting.


# Modeling Pipeline

The modeling workflow follows these steps:

1. Data cleaning and preprocessing
2. Feature selection and engineering
3. Stratified sampling of the dataset
4. One-hot encoding of categorical variables
5. Train / validation / test split
6. TabPFN model training
7. Performance evaluation using predictive metrics


# Expected Outcomes

The final model produces:

- Estimated probability of borrower default
- Evaluation metrics such as ROC-AUC and classification accuracy
- Insights into borrower risk patterns

These results can be used to support uncertainty-aware credit decision-making in lending systems.



# Summary

This project applies modern tabular machine learning methods to the problem of credit risk prediction. By combining real-world Lending Club data with transformer-based tabular modeling, we aim to estimate borrower default probabilities and demonstrate how probabilistic predictions can support better financial decision-making under uncertainty.