# Logistic Regression Analysis Summary

## Objectives
- Use Logistic Regression for classification
- Preprocess data for modeling
- Implement Logistic regression on real world data

## Scenario
A telecommunications firm is concerned about customers switching from land-line services to cable competitors. The goal is to predict which customers are most likely to leave (churn).

## Dataset
**Telco Churn Dataset** - Historical data with 200 customer records containing:
- Demographics (age, income, education)
- Service details (tenure, equipment, service usage)
- Target variable: churn (0 = stayed, 1 = churned)

## Workflow

### 1. Data Preprocessing
- Selected key features: `tenure`, `age`, `address`, `income`, `ed`, `employ`, `equip`
- Converted target variable `churn` to integer type
- Standardized features using `StandardScaler()` to bring all features to the same scale

### 2. Data Splitting
- Training set: 80% of data
- Testing set: 20% of data
- Used `train_test_split()` with random_state=4

### 3. Model Training
Built logistic regression model using `LogisticRegression().fit(X_train, y_train)`

**Model Parameters:**
- Intercept: -1.471
- Coefficients:
  - `tenure`: -0.846
  - `age`: -0.176
  - `address`: -0.124
  - `income`: -0.010
  - `ed`: 0.060
  - `employ`: -0.233
  - `equip`: 0.752

### 4. Coefficient Interpretation

**Negative coefficients (reduce churn likelihood):**
- Higher tenure, age, address, income, and employment reduce the probability of churning

**Positive coefficients (increase churn likelihood):**
- Higher education level and more equipment increase the probability of churning

**Intercept:**
- Negative intercept (-1.471) indicates baseline churn probability is low

### 5. Making Predictions

**Binary Predictions:**
- Used `predict()` to classify customers as 0 (stay) or 1 (churn)
- Threshold: 0.5

**Probability Predictions:**
- Used `predict_proba()` to get probability of each class
- Returns two columns: [P(stay), P(churn)]
- Example: [0.746, 0.254] means 74.6% likely to stay, 25.4% likely to churn

### 6. Performance Evaluation
- **Metric:** Log Loss
- **Result:** 0.626
- **Interpretation:** Reasonable prediction accuracy (lower log loss = better model)

## Key Insights

1. **Tenure** has the strongest negative impact on churn (-0.846) - longer customers are much less likely to leave
2. **Equipment** has the strongest positive impact (0.752) - customers with more equipment are more likely to churn
3. The model uses probabilities, not just binary predictions, providing more nuanced insights
4. Feature visualization shows relative importance of each predictor

## Conclusion
The logistic regression model successfully identifies customers at risk of churning and quantifies which factors influence churn decisions. This enables the company to implement targeted retention strategies for high-risk customers.
