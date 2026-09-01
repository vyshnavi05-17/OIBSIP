# 🚨 Fraud Detection Using Machine Learning

## 📌 Project Overview

This project focuses on building a **Machine Learning-based Fraud Detection System** to identify potentially fraudulent financial transactions.

The project uses a highly imbalanced financial transaction dataset containing **10,000 transactions**, where fraudulent transactions represent only a small percentage of the total data. Because of this severe class imbalance, the project focuses on meaningful fraud-detection metrics such as **Precision, Recall, F1-Score, and AUC-ROC** rather than relying on accuracy alone.

Two machine learning algorithms were developed and compared:

* **Logistic Regression**
* **Random Forest**

The project also uses **SMOTE (Synthetic Minority Over-sampling Technique)** to address the imbalance in the training data.

---

## 🎯 Project Objective

The main objective is to develop a machine learning pipeline capable of:

* Identifying fraudulent financial transactions
* Handling highly imbalanced transaction data
* Reducing the impact of class imbalance
* Comparing different classification algorithms
* Evaluating fraud detection using appropriate performance metrics
* Identifying the most influential fraud-related features
* Understanding the trade-off between detecting fraud and generating false alerts
* Discussing how the model could be scaled for high-volume transaction processing

---

## 📊 Dataset

The project uses the **`credit_card_fraud_10k.csv`** dataset.

### Dataset Statistics

| Metric                  |  Value |
| ----------------------- | -----: |
| Total Transactions      | 10,000 |
| Legitimate Transactions |  9,849 |
| Fraudulent Transactions |    151 |
| Legitimate Percentage   | 98.49% |
| Fraud Percentage        |  1.51% |

The dataset demonstrates a **severe class imbalance**, with legitimate transactions significantly outnumbering fraudulent transactions.

This makes fraud detection challenging because a model could achieve high accuracy simply by predicting most transactions as legitimate while failing to detect actual fraud.

---

## 🛠️ Technologies & Tools

### Programming Language

* Python

### Libraries

* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine learning and evaluation
* **Imbalanced-learn** – SMOTE for class balancing
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization

### Environment

* Jupyter Notebook

---

## 🔄 Project Workflow

The project follows the following machine learning workflow:

```text
Dataset
   ↓
Data Loading & Inspection
   ↓
Data Quality Checks
   ↓
Exploratory Data Analysis
   ↓
Class Imbalance Analysis
   ↓
Transaction Amount Analysis
   ↓
Time-of-Day Analysis
   ↓
Feature Preparation
   ↓
Train-Test Split
   ↓
SMOTE on Training Data
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Random Forest
   ↓
Model Evaluation
   ↓
Feature Importance Analysis
   ↓
Model Comparison
   ↓
Scalability Discussion
```

---

## 🔍 Exploratory Data Analysis

The project performs exploratory analysis to understand transaction behavior and fraud patterns.

### 1. Fraud vs Non-Fraud Distribution

The analysis shows a significant difference between legitimate and fraudulent transactions:

* **98.49%** of transactions are legitimate.
* **1.51%** of transactions are fraudulent.

This highlights the importance of handling class imbalance before training the models.

### 2. Transaction Amount Analysis

Transaction amounts were analyzed using:

* Box plots
* Histograms
* Fraud vs non-fraud comparisons

This helps investigate whether transaction amount patterns differ between fraudulent and legitimate transactions.

### 3. Fraud Rate by Transaction Hour

The project analyzes fraud activity based on the transaction hour.

An important observation from the analysis is that **early-morning transactions show higher fraud rates**.

* Hour 0 has the highest fraud rate at approximately **8.87%**.
* Hours 1–3 also show relatively high fraud rates.
* Fraud rates decrease significantly after the early-morning period.
* Hours 7, 12, 16, and 22 recorded **0% fraud** in this dataset.

This suggests that **transaction timing can be an important fraud-related feature**.

---

## ⚠️ Why Accuracy Is Not Enough

Accuracy can be misleading when working with highly imbalanced datasets.

With **98.49% legitimate transactions**, a model could theoretically predict every transaction as legitimate and achieve approximately **98.49% accuracy**.

However, it would detect **zero fraudulent transactions**.

For this reason, this project focuses on:

### Precision

Measures how many transactions predicted as fraud are actually fraudulent.

### Recall

Measures how many actual fraudulent transactions are successfully detected.

### F1-Score

Provides a balance between Precision and Recall.

### AUC-ROC

Measures how effectively the model separates fraudulent transactions from legitimate transactions.

---

## ⚖️ Handling Class Imbalance with SMOTE

The project uses **SMOTE (Synthetic Minority Over-sampling Technique)** to address the imbalance between fraud and non-fraud transactions.

SMOTE is applied **only to the training data**, while the test dataset remains unchanged.

This prevents synthetic samples from influencing the evaluation dataset and provides a more realistic assessment of model performance.

---

# 🤖 Machine Learning Models

## 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

The model was trained after:

* Removing the transaction ID
* Encoding the categorical `merchant_category` feature
* Splitting the data into training and testing sets
* Applying SMOTE to the training set
* Scaling the features

### Logistic Regression Performance

| Metric    | Performance |
| --------- | ----------: |
| Precision |        ~16% |
| Recall    |      93.33% |
| F1-Score  |      27.32% |
| AUC-ROC   |      98.30% |

### Interpretation

The Logistic Regression model achieved a very high **Recall of 93.33%**, meaning it successfully detected most fraudulent transactions.

However, the model achieved only around **16% Precision**, indicating that many legitimate transactions were incorrectly classified as fraudulent.

Therefore, while Logistic Regression is effective at detecting fraud, it generates a relatively high number of false-positive alerts.

---

# 🌲 2. Random Forest

A **Random Forest Classifier** was also trained using:

* 200 decision trees
* Random state of 42
* Parallel processing using `n_jobs=-1`

Unlike Logistic Regression, Random Forest was trained using the SMOTE-balanced training data without feature scaling.

### Random Forest Performance

| Metric    |                   Performance |
| --------- | ----------------------------: |
| Precision |                          ~46% |
| Recall    |                          ~73% |
| F1-Score  |                          ~56% |
| AUC-ROC   | As calculated in the notebook |

### Interpretation

Random Forest provides a better balance between **Precision and Recall** compared with Logistic Regression.

Although its Recall is lower than Logistic Regression, its Precision is substantially higher, meaning fewer legitimate transactions are incorrectly flagged as fraudulent.

The resulting F1-Score is also considerably higher.

---

# 📈 Model Comparison

| Model               | Precision | Recall | F1-Score |       AUC-ROC |
| ------------------- | --------: | -----: | -------: | ------------: |
| Logistic Regression |      ~16% | 93.33% |   27.32% |        98.30% |
| Random Forest       |      ~46% |   ~73% |     ~56% | As calculated |

### Model Selection

Based on the project results, **Random Forest is the preferred model among the two tested models**.

The reason is that it provides a stronger balance between:

* Detecting fraudulent transactions
* Reducing false-positive alerts
* Overall Precision and Recall balance

However, the final model used in a real-world fraud detection system should depend on the business cost of:

* Missing fraudulent transactions
* Incorrectly flagging legitimate transactions

---

# 🔎 Feature Importance Analysis

The Random Forest model was also used to identify the most influential features.

The most important features identified were:

| Feature                     | Approx. Importance |
| --------------------------- | -----------------: |
| `device_trust_score`        |             31.68% |
| `transaction_hour`          |             24.50% |
| `amount`                    |              8.74% |
| `velocity_last_24h`         |              7.69% |
| `merchant_category_Grocery` |              5.07% |

### Key Insight

`device_trust_score` was the most influential feature, followed by `transaction_hour`.

This indicates that **device-related characteristics and transaction timing play an important role in distinguishing fraudulent transactions within this dataset**.

Transaction amount and recent transaction velocity also contribute to fraud prediction.

---

# 📊 Visualizations

The notebook includes several visual analyses, including:

* Fraud vs Non-Fraud Transaction Distribution
* Transaction Amount Box Plot
* Transaction Amount Distribution
* Fraud Rate by Transaction Hour
* Class Distribution After SMOTE
* Logistic Regression Confusion Matrix
* Random Forest Confusion Matrix
* Logistic Regression ROC Curve
* Random Forest ROC Curve
* Random Forest Feature Importance

These visualizations help communicate both the dataset characteristics and model performance.

---

# 🧠 Business Insights

The project provides several important insights for fraud detection:

### 1. Fraud is a minority event

Only **1.51%** of transactions in the dataset are fraudulent, demonstrating why class imbalance must be addressed.

### 2. Early-morning transactions are higher risk

The fraud-rate analysis indicates that transactions occurring between **00:00 and 03:00** have significantly higher fraud rates in this dataset.

### 3. Device trust is highly influential

`device_trust_score` is the strongest feature identified by the Random Forest model.

### 4. Fraud detection requires a Precision-Recall balance

Maximizing Recall can help detect more fraudulent transactions, but it can also increase false-positive alerts.

### 5. Random Forest provides a better balance

Compared with Logistic Regression, Random Forest provides a stronger balance between fraud detection and false alerts in this project.

---

# 🚀 Scalability & Real-World Deployment

The notebook also considers how the fraud detection model could be deployed in a high-volume environment.

For example, processing **1 million transactions per hour** would require approximately **278 transactions per second**.

A production architecture could include:

```text
Incoming Transactions
        ↓
Streaming Platform / Message Queue
        ↓
Feature Processing
        ↓
Fraud Detection Model
        ↓
Prediction API
        ↓
Fraud / Non-Fraud Decision
        ↓
Monitoring & Alerting
```

For high-volume processing, the system could use:

* Real-time feature processing
* Batch processing
* Parallel model inference
* Model-serving APIs
* Caching
* Distributed processing
* Cloud-based auto-scaling
* Model performance monitoring
* Data-drift monitoring
* Continuous model retraining

The current notebook demonstrates fraud classification on a **10,000-transaction dataset**. A production system would require additional infrastructure, monitoring, security, optimization, and continuous model improvement.

---

# 🎯 Key Takeaways

* Fraud detection is a highly imbalanced classification problem.
* Accuracy alone is not a reliable metric for this dataset.
* SMOTE helps address class imbalance in the training data.
* Logistic Regression provides very high Recall but low Precision.
* Random Forest provides a better Precision-Recall balance.
* `device_trust_score` and `transaction_hour` are the most influential features.
* Early-morning transactions show higher fraud rates in this dataset.
* Random Forest is the preferred model among the two tested.
* Production deployment would require scalable model-serving infrastructure.

---

# 📁 Project Structure

```text
Fraud-Detection/
│
├── Fraud_Detection.ipynb
├── credit_card_fraud_10k.csv
├── fraud_dataset.csv
└── README.md
```

---

# 💡 Future Improvements

Potential improvements for a production-ready fraud detection system include:

* Hyperparameter tuning
* Cross-validation
* Threshold optimization
* Precision-Recall curve analysis
* Cost-sensitive learning
* Advanced ensemble models
* Real-time fraud scoring
* Model monitoring
* Data drift detection
* Automated model retraining
* API-based model deployment
* Cloud deployment

---

# 🏁 Conclusion

This project demonstrates an end-to-end **Machine Learning workflow for financial fraud detection**, starting from data exploration and class imbalance analysis through model development, evaluation, feature importance analysis, and scalability considerations.

Using a dataset of **10,000 financial transactions**, the project demonstrates the challenges associated with detecting a rare fraud class.

SMOTE was used to balance the training data, followed by the development of **Logistic Regression and Random Forest** models.

Logistic Regression achieved a **93.33% Recall**, making it effective at identifying fraudulent transactions, but its low Precision resulted in many false-positive alerts.

Random Forest achieved approximately **46% Precision, 73% Recall, and 56% F1-Score**, providing a better overall balance.

Therefore, **Random Forest was selected as the preferred model among the tested approaches**.

The project demonstrates how machine learning can be applied to fraud detection while highlighting the importance of class imbalance handling, appropriate evaluation metrics, feature analysis, and scalable deployment architecture.



# Retail Sales Exploratory Data Analysis (EDA)

## 📌 Project Overview

This project performs **Exploratory Data Analysis (EDA)** on a retail sales dataset using Python. The analysis focuses on understanding sales trends, customer demographics, product performance, transaction behaviour, and relationships between numerical variables.

The objective is to transform raw retail sales data into meaningful business insights that can support better decision-making in areas such as inventory planning, marketing, customer segmentation, and revenue optimization.

---

## 🎯 Objectives

The main objectives of this project are:

* Inspect and understand the retail sales dataset.
* Perform data cleaning and quality checks.
* Analyze descriptive statistics.
* Identify monthly and quarterly sales trends.
* Understand customer demographics.
* Analyze product-category performance.
* Study relationships between numerical variables using correlation analysis.
* Analyze transaction-value distribution.
* Generate actionable business recommendations.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical computations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **JupyterLab** – Development environment

---

## 📊 Analysis Performed

### 1. Data Inspection & Cleaning

The dataset was inspected to understand its structure, columns, data types, missing values, and duplicate records.

The analysis confirmed that **no duplicate records were found**, so duplicate-row removal was not required.

---

### 2. Descriptive Statistics

Descriptive statistics were used to understand the central tendency and variability of numerical variables such as:

* Age
* Quantity
* Price per Unit
* Total Amount

Mean, median, and summary statistics were calculated to understand the characteristics of the dataset.

---

### 3. Time Series Analysis

Monthly and quarterly sales trends were analyzed to understand how sales changed over time.

**Key findings:**

* May 2023 recorded the highest monthly sales of **53,150**.
* January 2024 recorded the lowest sales of **1,530**.
* Among the complete quarters, **Q4 2023** recorded the highest sales of **126,190**.
* Q1 2024 contains only January data, so it should not be directly compared with complete quarters.

These trends can help businesses improve inventory planning, sales targets, and promotional campaigns.

---

### 4. Customer Demographic Analysis

Customer demographics were analyzed using age groups and gender.

#### Age Group

The **46–55 age group** was the largest customer segment, representing **229 customers (22.9%)**.

Customers between **26 and 65 years** make up the majority of the customer base.

#### Gender

The gender distribution was relatively balanced:

* Female: **510 customers (51%)**
* Male: **490 customers (49%)**

This indicates that there is no strong gender imbalance in the dataset.

---

### 5. Product Category Analysis

Product performance was analyzed using sales quantity and revenue.

The analysis revealed an important difference between sales volume and revenue:

* **Clothing** had the highest quantity sold, with approximately **895 units**.
* **Electronics** generated the highest revenue despite having a lower sales quantity.
* **Beauty** had the lowest quantity sold and lowest total revenue among the three categories.

This demonstrates that product performance should not be evaluated using sales volume alone.

---

### 6. Correlation Analysis

A correlation heatmap was used to examine relationships between numerical variables.

Key findings:

* **Price per Unit and Total Amount:** strong positive correlation of **0.85**
* **Quantity and Total Amount:** moderate positive correlation of **0.37**
* **Age and Total Amount:** very weak correlation of **-0.06**

The results indicate that product pricing has a stronger relationship with transaction value than customer age.

---

### 7. Transaction Value Analysis

The distribution of transaction amounts was analyzed to understand customer spending behaviour.

The transaction values are **strongly right-skewed**, meaning most transactions are concentrated in the lower-value range, while a smaller number of transactions have substantially higher values.

This suggests opportunities to increase transaction value through:

* Product bundles
* Cross-selling
* Complementary product recommendations
* Minimum-purchase discounts
* Premium product offers

---

## 💡 Key Business Insights

1. **Electronics is an important revenue-generating category**, despite having fewer units sold than Clothing.
2. **Customers aged 26–65 form the majority of the customer base**, with the 46–55 segment being the largest.
3. **Sales fluctuate significantly over time**, with Q4 2023 showing the strongest complete-quarter performance.
4. **Product price has a strong relationship with transaction value**, with a correlation of 0.85 between Price per Unit and Total Amount.
5. **Most transactions are lower-value purchases**, creating an opportunity to increase average transaction value.
6. **Gender distribution is relatively balanced**, so customer segmentation should not rely heavily on gender alone.

---

## 📈 Business Recommendations

### 1. Focus on High-Value Electronics

Maintain sufficient inventory of high-performing Electronics products and consider targeted promotions to maximize revenue.

### 2. Target the 26–65 Customer Segment

Develop targeted marketing campaigns, loyalty programs, and personalized recommendations for the major customer segments.

### 3. Increase Low-Value Transaction Amounts

Use bundles, cross-selling, minimum-purchase discounts, and complementary product recommendations to increase transaction value.

### 4. Use Sales Trends for Planning

Use strong sales periods such as Q4 2023 as a reference for inventory planning, marketing campaigns, and sales targets.

### 5. Analyze Product Pricing

Since Price per Unit has a strong relationship with Total Amount, pricing strategy and high-value products should be considered when developing revenue-growth strategies.

---

## 📁 Project Structure

```text
OIBSIP/
│
└── DataAnalyst-Level1-Task1-RetailSalesEDA/
    │
    ├── Retail_Sales_EDA.ipynb
    └── README.md
```

---

## 🔍 Conclusion

The Retail Sales EDA project provides a detailed view of sales performance, customer demographics, product categories, and transaction behaviour.

The analysis shows that **sales trends, product pricing, customer segments, and transaction values** can provide valuable insights for business decision-making. In particular, Electronics contributes strongly to revenue, customers aged 26–65 represent the majority of the dataset, and product price has a strong positive relationship with transaction value.

Overall, the project demonstrates how **Python-based exploratory data analysis and visualization can be used to convert retail data into actionable business insights**.

