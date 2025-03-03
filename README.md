# Trip Type Prediction using Machine Learning

## Overview
This project implements a machine learning model to predict shopping trip types based on transactional data. The dataset is preprocessed, analyzed, and used to train a Random Forest classifier to categorize different trip types.

## Features
- **Exploratory Data Analysis (EDA)**
  - Identifies missing values, duplicates, and class distribution.
  - Visualizes data distributions and relationships between features.
- **Feature Engineering**
  - Aggregates purchase data per trip.
  - Encodes categorical variables (e.g., weekdays, department descriptions).
  - Converts cyclic time variables using sine and cosine transformations.
- **Model Training and Evaluation**
  - Trains a Random Forest Classifier.
  - Evaluates model performance using accuracy, log loss, and classification reports.
  - Extracts feature importance scores.

## Dataset
The dataset includes information on customer transactions, such as:
- **TripType** - The type of shopping trip (target variable).
- **VisitNumber** - A unique identifier for each shopping trip.
- **Weekday** - The day of the week when the trip occurred.
- **Upc** - The unique product identifier.
- **ScanCount** - The quantity of each item purchased (negative values indicate returns).
- **DepartmentDescription** - The department where the item was purchased.
- **FinelineNumber** - A subcategory identifier for the product.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

### Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

### 1. Load Data
Modify the script to specify the correct dataset path:
```python
import pandas as pd

# Load dataset
df = pd.read_csv('train.csv')
```

### 2. Preprocess Data
- Removes duplicates.
- Aggregates items per trip.
- Encodes categorical variables.

```python
# Remove duplicate rows
df = df.drop_duplicates()

# Aggregate purchases per trip
df = df.groupby(['Upc', 'TripType', 'VisitNumber', 'Weekday',
            'DepartmentDescription', 'FinelineNumber'])['ScanCount'].sum().reset_index()
```

### 3. Feature Engineering
- One-hot encoding for department descriptions.
- Converts weekdays into cyclical features using sine and cosine transformations.

```python
day_of_week = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
df['sin_day'] = np.sin(2 * np.pi * df['Weekday'].map(day_of_week) / 7)
df['cos_day'] = np.cos(2 * np.pi * df['Weekday'].map(day_of_week) / 7)
```

### 4. Train the Model
Splits data into training and testing sets and trains a Random Forest classifier.
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(n_estimators=500, random_state=42)
rfc.fit(X_train, y_train)
```

### 5. Evaluate the Model
Computes accuracy and log loss.
```python
from sklearn.metrics import log_loss, accuracy_score

y_test_pred_prob = rfc.predict_proba(X_test)
y_test_pred = rfc.predict(X_test)

print('Log Loss:', log_loss(y_test, y_test_pred_prob))
print('Accuracy:', accuracy_score(y_train, rfc.predict(X_train)))
```

### 6. Feature Importance Analysis
Plots the most important features.
```python
import matplotlib.pyplot as plt
import seaborn as sns

importances = pd.DataFrame(rfc.feature_importances_, index=X_train.columns).reset_index()
importances.columns=['features', 'importance']
importances = importances.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 30))
sns.barplot(x='importance', y='features', data=importances, palette='mako')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance (GINI)')
plt.ylabel('Features')
plt.show()
```

## Example Output
```bash
Log Loss: 0.3456
Accuracy: 0.89
```

## Notes
- The dataset is imbalanced, which may affect model performance on minority classes.
- The model is non-linear and performs well, but further optimization (e.g., hyperparameter tuning) may improve results.

## License
This project is licensed under the MIT License.

