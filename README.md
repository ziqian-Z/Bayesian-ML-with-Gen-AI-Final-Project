# Bayesian Modeling for Uncertainty-Aware Credit Risk Decisions

Wendy Zhao, Zimeng Yang, Yiqin He


# Project Overview

This project studies borrower default risk prediction using large-scale credit data. The objective is to estimate the probability that a borrower will default on a loan and use these probabilities to support lending decisions such as:

- Loan approval or rejection  
- Loan amount allocation  
- Interest-rate tiering  

Rather than treating credit risk as a deterministic classification problem, we frame it as an uncertain probabilistic inference problem, where each borrower is assigned a probability of default.

These probability estimates allow lenders to make risk-aware decisions under asymmetric costs, where approving a risky borrower may lead to significant losses while rejecting a good borrower leads to opportunity cost.


# Task Description

The primary modeling task is binary classification of loan outcomes, where the model estimates:


$$P(\text{default} = 1 \mid \text{borrower features})$$


This probabilistic approach allows downstream decision rules to be adjusted depending on the lender’s risk tolerance.


# Data 

## Source

We use the Lending Club Loan Dataset, obtained from Kaggle:
 
https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv/data

It originates from Lending Club, a U.S.-based peer-to-peer lending platform that connects individual investors with borrowers. Investors provide capital for loans, and borrowers repay the loan principal with interest over time.

## Dataset Description

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

These repayment status indicators allow us to construct a **binary default** variable used for modeling.

The dataset reflects real-world credit performance and is widely used for research in credit risk modeling.



# Methodology

## Data Pre-processing

- Columns with extremely high missing rates were removed.
- Post-origination variables that could cause data leakage (such as repayment outcomes or post-loan payment variables) were excluded.
- Joint loan applications were filtered out to simplify the modeling framework.
- Categorical variables were encoded using one-hot encoding.
- High-cardinality features such as ZIP codes were aggregated into broader geographic categories.
- Missing values in selected credit-history variables were handled using indicator variables and placeholder values.
- Other missing values were handled using domain-informed imputation strategies.

These steps ensure that the model only uses information available at the time of loan approval.

## Data Sampling

Because the original dataset contains more than two million observations, a random subset of the data was used for experimentation.
	•	50,000 observations were randomly sampled from the cleaned dataset.
	•	To reduce computational cost for TapPFN construction, 20,000 observations were used for training and model experimentation.

All sampling preserves the original class distribution of default and non-default observations.

## Model

We implement and compare two approaches for predicting borrower default risk: 

- a **Bayesian logistic regression** model and

- **TabPFN** (Tabular Prior-Data Fitted Network).

The Bayesian logistic regression serves as a baseline probabilistic model. It estimates the probability of borrower default using a logistic likelihood with Bayesian parameter estimation. By placing priors on the regression coefficients, the model provides interpretable relationships between borrower characteristics and default risk while allowing uncertainty in parameter estimates to be quantified through posterior distributions. 

In addition to the Bayesian model, we apply TabPFN, a pretrained transformer-based foundation model designed for tabular prediction tasks. TabPFN is trained on a large collection of synthetic tabular datasets and learns a general algorithm for solving classification problems through in-context learning. Instead of training a model from scratch on each dataset, TabPFN can directly infer relationships between features and outcomes in a single forward pass. 

By comparing these two models, we evaluate the trade-off between interpretability and uncertainty-aware inference offered by Bayesian logistic regression and the predictive power of modern transformer-based models for tabular data. This comparison allows us to assess whether advanced deep learning methods provide meaningful improvements over traditional statistical approaches in credit risk prediction.


# Result (Need to change)

## Model Performance Metrics

## Bayesian Logistic

## TapPFN

## Comparison

# Conclusion

# Limit and Future Work

