<p align="center">
  <img src="https://www.lgresearch.ai/data/upload/image/blog/LG_AI_Research_LG_Aimer_Hackathon_thumbnail_c3d603fc1.png" alt="LG" width="500" height="300">
</p>
Certainly! Here's the detailed `README.md` file in English and in Markdown format:

---

# LG-Aimers

## Sales Conversion Prediction Competition

### Overview

With the recent significant advancements in machine learning, many models with higher performance than traditional ones have emerged. These models are being adopted to predict the likelihood of customer conversion, allowing sales resources to be focused on high-potential leads.

In this competition, we aim to implement and compare models that determine whether a customer will convert based on various customer information.

### Data Description

| Field                       | Description                                                                          |
|-----------------------------|--------------------------------------------------------------------------------------|
| bant_submit                 | Ratio of filled values for Budget, Title, Needs, and Timeline in MQL components       |
| customer_country            | Customer's nationality (grouped by country, with those having frequency <= 20 as 'Others') |
| business_unit               | Business unit corresponding to the MQL requested product                             |
| com_reg_ver_win_rate        | Opportunity ratio calculated based on Vertical Level 1, business unit, and region    |
| customer_idx                | Customer company name                                                                |
| customer_type               | Customer type                                                                        |
| enterprise                  | Whether the company is a Global enterprise or a Small/Medium-sized enterprise        |
| historical_existing_cnt     | Number of previous conversions                                                        |
| id_strategic_ver            | Weight assigned to specific business unit and business area                          |
| it_strategic_ver            | Weight assigned to specific business unit and business area                          |
| idit_strategic_ver          | Value of 1 if either id_strategic_ver or it_strategic_ver is 1                       |
| customer_job                | Customer's job category                                                              |
| lead_desc_length            | Total length of the Lead Description text provided by the customer                   |
| inquiry_type                | Type of inquiry from the customer                                                    |
| product_category            | Category of the requested product                                                    |
| product_subcategory         | Subcategory of the requested product                                                 |
| product_modelname           | Model name of the requested product                                                  |
| customer_country.1          | Regional information based on the company in charge (continent)                      |
| customer_position           | Customer's company position                                                          |
| response_corporate          | Name of the corporate in charge                                                      |
| expected_timeline           | Customer's requested processing schedule                                             |
| ver_cus                     | Weight for specific Vertical Level 1 and customer type being an end-user             |
| ver_pro                     | Weight for specific Vertical Level 1 and specific product category                   |
| ver_win_rate_x              | Product of the ratio of total leads by vertical and conversion success rate by vertical|
| ver_win_ratio_per_bu        | Ratio of converted samples to total samples for specific Vertical Level 1 by business unit |
| business_area               | Customer's business area                                                             |
| business_subarea            | Customer's sub-business area                                                         |
| lead_owner                  | Name of the sales representative                                                     |
| **is_converted**            | **Sales conversion status. True if successful.**                                     |

### Target

- **is_converted**: Sales conversion status (True/False)

### Data Rules

- `id` is not present in the training data but is used in the test data where `is_converted` is empty.
- `id`: Unique customer number ensuring correct scoring even if customer order is shuffled. Removing or altering the `id` column results in a score of 0.

### Evaluation Metric

The competition uses the F1 Score as the classification metric for evaluating the results.

![Formula](https://github.com/KeonhoChu/LG_Ai/blob/main/mth.png?raw=true)

### Competition Results

| Overall Ranking | Total Score |
|-----------------|-------------|
| 59th / 844 teams| 0.766316    |

## Code Explanation

### 1. Library Import

Required libraries such as pandas, numpy, scikit-learn are imported.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import random
import numpy as np
```

### 2. Data Loading

Data is loaded from CSV files.

```python
df_train = pd.read_csv("train.csv") 
df_train2 = pd.read_csv("train.csv") 
df_test = pd.read_csv("submission.csv") 
```

### 3. Data Preprocessing

Handling missing values, encoding categorical data, and other preprocessing tasks are performed.

```python
# Example: Handling missing values

df_all["customer_country"] = df_all["customer_country"].str.split('/').str[-1].str.lower().str.strip()
filtered_countrys = df_all["customer_country"].value_counts()[df_all["customer_country"].value_counts() >= 30]

def filter_country(country):
    if pd.isna(country):
        return "none"
    elif country in filtered_countrys.index:
        return country
    else:
        return "else"
df_all["customer_country"] = df_all["customer_country"].apply(filter_country)

filtered_expected_timeline = df_all["expected_timeline"].str.replace(' ', '').str.replace('_', '').str.replace('.', '').value_counts()[df_all["expected_timeline"].str.replace(' ', '').str.replace('_', '').str.replace('.', '').value_counts() >= 100]

def filter_expected_timeline(expected_timeline):
    if pd.isna(expected_timeline):
            return "none" 
    elif expected_timeline in filtered_expected_timeline.index:
        return expected_timeline
    else:
        return "etc"

df_all["expected_timeline"] = df_all["expected_timeline"].str.replace(' ', '').str.replace('_', '').str.replace('.', '')
df_all["expected_timeline"] = df_all["expected_timeline"].apply(filter_expected_timeline)

def categorize_inquiry(value):
    if pd.isna(value):
        return "none"
    else :  # 문자열인 경우에만 처리
        value = str(value).lower()
        if "quotation" in value or "purchase" in value:
            return "quotation or purchase"
        elif "sales" in value:
            return "sales"
        elif "information" in value or "trainings" in value:
            return "information"
        elif "technic" in value or "usage" in value:
            return "technical"
        else:
            return "else"
            
    
df_all["inquiry_type"] = df_all["inquiry_type"].apply(categorize_inquiry)

df_all["customer_type"] = df_all["customer_type"].str.replace(' ', '').str.replace('/', '').str.replace('_', '').str.lower().str.replace('-', '')
filtered_customer_type = df_all["customer_type"].value_counts()[df_all["customer_type"].value_counts() >= 30]

def filter_customer_type(customer_type):
    if pd.isna(customer_type):
        return "none"
    elif customer_type in filtered_customer_type.index:
        return customer_type
    else:
        return "others."
filtered_product_subcategory = df_all["product_subcategory"].value_counts()[df_all["product_subcategory"].value_counts() >= 20]

def filter_product_subcategory(product_subcategory):
    if pd.isna(product_subcategory):
        return "none"
    elif product_subcategory in filtered_product_subcategory.index:
        return product_subcategory
    else:
        return "etc."
    
df_all["product_subcategory"] = df_all["product_subcategory"].apply(filter_product_subcategory)
df_all["customer_type"] = df_all["customer_type"].apply(filter_customer_type)

    
filtered_product_category = df_all["product_category"].value_counts()[df_all["product_category"].value_counts() >= 30]

def filter_product_category(category):
    if pd.isna(category):
        return "none"
    elif category in filtered_product_category.index:
        return category
    else:
        return "etc."
df_all["product_category"] = df_all["product_category"].apply(filter_product_category)

filtered_jobs = df_all["customer_job"].value_counts()[df_all["customer_job"].value_counts() >= 20]

def filter_jobs(job):
    if pd.isna(job):
        return "none"
    elif job in filtered_jobs.index:
        return job
    else:
        return "else"
df_all["customer_job"] = df_all["customer_job"].apply(filter_jobs).replace({"others": "else", "other": "else"})


# Example: Label encoding

def label_encoding(series: pd.Series) -> pd.Series:
    """범주형 데이터를 시리즈 형태로 받아 숫자형 데이터로 변환합니다."""

    my_dict = {}

    # 모든 요소를 문자열로 변환
    series = series.astype(str)

    for idx, value in enumerate(sorted(series.unique())):
        my_dict[value] = idx
    series = series.map(my_dict)

    return series

# 레이블 인코딩할 칼럼들
label_columns = [
    "business_unit",
    "enterprise",
    "customer_position",
    "response_corporate",
    "customer_country",
    "product_category",
    "inquiry_type",
    "expected_timeline",
    "customer_job",
    "product_subcategory",
    "customer_type"
]

df_all = pd.concat([F_D_F[label_columns], df_test[label_columns]])

for col in label_columns:
    df_all[col] = label_encoding(df_all[col])
```

### 4. Feature and Target Definition

Features and target variables for the model are defined.

```python
import xgboost as xgb
model_xgb = xgb.XGBClassifier(random_state = 0)

y_train = train_data["is_converted"]
X_train = train_data.drop(["is_converted"],axis = 1)
y_test = test_data["is_converted"]
X_test = test_data.drop(["is_converted"],axis = 1)
model_xgb.fit(X_train, y_train)


y_xgb = model_xgb.predict(X_test)


acc_xgb = round(accuracy_score(y_test, y_xgb),4)*100

print("\n\nAccuracy with untuned XGBoost is: " + str(acc_xgb) + " %")
```

### 5. Data Splitting

The dataset is split into training and validation sets.

```python
train_data, test_data = train_test_split(F_D_F, test_size=0.2, random_state=42)
```

### 6. Model Training

A XGBoost model is trained on the data.

```python
import xgboost as xgb
model_xgb = xgb.XGBClassifier(random_state = 0)
model_xgb.fit(X_train, y_train)
```

### 7. Model Evaluation

The model is evaluated on the validation set.

```python
y_pred = model.predict(X_valid)
f1 = f1_score(y_valid, y_pred)
print(f'Validation F1 Score: {f1}')
```

### 8. Predictions and Submission File Creation

Predictions are made on the test set and a submission file is created.

```python
test_predictions = model.predict(test[features])
submission = pd.DataFrame({
    'id': test['id'],
    'is_converted': test_predictions
})
submission.to_csv('submission.csv', index=False)
```

