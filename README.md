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

### Model Description

We implemented a Bayesian logistic regression model as an interpretable baseline for credit risk prediction. Logistic regression is a natural choice for binary default prediction because it directly models the probability of default through a sigmoid transformation of a linear combination of borrower features. In the Bayesian framework, model parameters are treated as random variables with prior distributions, which allows us to quantify uncertainty in coefficient estimates and predicted probabilities. In this implementation, we used weakly informative Normal priors for the intercept and coefficients and performed posterior inference using variational inference (ADVI), which provides an efficient approximation to the posterior distribution.

To improve numerical stability and model performance, several preprocessing steps were applied. Continuous variables were standardized, categorical variables were encoded using one-hot encoding, and missing-value placeholders (e.g., 999) were converted to NaN before imputation. In addition, post-origination variables that would introduce target leakage were removed. The final feature set focuses on borrower characteristics and credit profile variables available at the time of loan issuance.

### Model Performance

![alt text](image-1.png)

The Bayesian logistic model achieved an AUC of 0.696, which indicates a moderate ability to rank higher-risk borrowers above lower-risk ones. The Brier score of 0.107 suggests that the predicted probabilities are reasonably well calibrated overall. The calibration curve further supports this observation: predicted probabilities generally follow the diagonal reference line, indicating that the model’s probability estimates are consistent with the observed default frequencies across probability bins.


However, the confusion matrix highlights an important challenge typical in credit risk modeling: class imbalance. The dataset contains substantially more non-default cases than default cases. As a result, using a standard threshold of 0.5 leads to very high accuracy (0.869) driven mainly by correct predictions of non-defaults, but extremely low recall for defaults (0.007). In other words, the model rarely predicts default at the 0.5 threshold, which limits its usefulness for operational risk detection. This does not necessarily indicate poor model quality; rather, it reflects the mismatch between the threshold and the base rate of default events.


## TapPFN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vJjomPtd3mIloP0Vk5gfwRBC8oGi9O__#scrollTo=bNK3shxdJuQh)

### ROC-AUC

![ROC-AUC Plot](Result/TabPFN%20result/roc-auc.png)

Test ROC-AUC: 0.7075

An AUC value of approximately 0.71 indicates that the model has a meaningful ability to distinguish between borrowers who default and those who do not. In probabilistic terms, this means that given a randomly selected pair of borrowers—one who defaulted and one who did not—the model assigns a higher predicted default probability to the defaulter about 71% of the time.

This level of performance suggests that the model captures relevant patterns in borrower credit characteristics and loan attributes, although the classification boundary between risky and safe borrowers is not perfectly separable.

### Prediction Distribution

![Prediction Distribution](Result/TabPFN%20result/prediction%20distribution.png)

The blue distribution represents non-default loans, while the orange distribution represents defaulted loans. Non-default borrowers are concentrated at lower predicted probabilities, typically below 0.15, while defaulted borrowers tend to have higher predicted probabilities and a longer right tail extending toward higher risk levels.

Although the two distributions overlap, there is a noticeable shift to the right for the default class, indicating that the model successfully assigns higher predicted risk to borrowers who ultimately defaulted.

### Calibration Analysis

![Calibration Curve](Result/TabPFN%20result/calibration.png)

The dashed diagonal line represents perfect calibration, where predicted probabilities match the actual default frequency. The model performs reasonably well in the lower probability region, where most borrowers are located. However, in the medium probability range the model tends to underestimate the true default rate, suggesting that higher-risk borrowers may have slightly greater default risk than the model predicts.

Despite this slight underestimation, the calibration pattern indicates that the model provides useful probabilistic risk estimates rather than purely ranking borrowers.

## Comparison

# Conclusion

# Limit and Future Work

