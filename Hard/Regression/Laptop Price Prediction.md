---
title: Laptop Price Prediction
company: Allegro
difficulty: Hard
category: Regression
date: 2025-07-28
---
_This data project has been used as a take-home assignment in the recruitment process for the data science positions at Allegro._

## Assignment

Your task is to define and train a machine learning model for predicting the price of a laptop (`buynow_price` column in the dataset) based on its attributes. When testing and comparing your models, aim to minimize the RMSE measure.

## Data Description

The dataset has already been randomly divided into the training, validation and test sets. It is stored in 3 files: `train_dataset.json`, `val_dataset.json` and `test_dataset.json` respectively. Each file is JSON saved in [`orient=’columns’` format](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html).

#### Example how to load the data:

```
>>> import pandas as pd
>>> dataset = pd.read_json("public-dataset.json")
>>> dataset.columns
Index(['buynow_price', 'graphic card type', 'communications', 'resolution (px)', 'CPU cores', 'RAM size', 'operating system', 'drive type', 'input devices', 'multimedia', 'RAM type', 'CPU clock speed (GHz)', 'CPU model', 'state', 'drive memory size (GB)', 'warranty', 'screen size'], dtype='object')
```

## Practicalities

Prepare a model in Jupyter Notebook using Python. Only use the training data for training the model and check the model's performance on unseen data using the test dataset to make sure it does not overfit.

Ensure that the notebook reflects your thought process. It’s better to show all the approaches, not only the final one (e.g. if you tested several models, you can show all of them). The path to obtaining the final model should be clearly shown.

# Solution

Given the nature of the request, which involves a take-home assignment for a data science position, I will outline a structured approach to solving this problem. I cannot directly execute a Jupyter Notebook or train models myself. However, I can provide a comprehensive plan, including code snippets and explanations, that you can follow to complete the assignment effectively.

Here's a step-by-step plan:

**Project Title: Laptop Price Prediction at Allegro**

**Objective:** Define and train a machine learning model to predict laptop prices (`buynow_price`) minimizing RMSE.

---

### **Step 1: Setup and Data Loading**

First, set up your environment and load the datasets.

Python

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the datasets
try:
    train_df = pd.read_json("train_dataset.json")
    val_df = pd.read_json("val_dataset.json")
    test_df = pd.read_json("test_dataset.json")
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("Ensure 'train_dataset.json', 'val_dataset.json', and 'test_dataset.json' are in the same directory.")
    # Exit or handle the error appropriately
    exit()

print(f"Train set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}")
print(f"Test set shape: {test_df.shape}")
```

### **Step 2: Exploratory Data Analysis (EDA)**

Understand the data, identify missing values, outliers, and relationships.

#### **2.1 Initial Inspection**

Python

```
# Display basic information about the training data
print("\n--- Train Data Info ---")
train_df.info()

# Display the first few rows of the training data
print("\n--- Train Data Head ---")
print(train_df.head())

# Describe numerical features
print("\n--- Train Data Description (Numerical) ---")
print(train_df.describe())

# Describe categorical features (object type)
print("\n--- Train Data Description (Categorical) ---")
print(train_df.describe(include='object'))
```

#### **2.2 Target Variable Analysis (`buynow_price`)**

Python

```
plt.figure(figsize=(10, 6))
sns.histplot(train_df['buynow_price'], bins=50, kde=True)
plt.title('Distribution of Laptop Prices (buynow_price)')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Log transform if the distribution is skewed
train_df['buynow_price_log'] = np.log1p(train_df['buynow_price'])
plt.figure(figsize=(10, 6))
sns.histplot(train_df['buynow_price_log'], bins=50, kde=True)
plt.title('Distribution of Log-transformed Laptop Prices')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.show()
```

#### **2.3 Feature Analysis**

Iterate through features to understand their distributions and relationships with the target variable.

Python

```
# Missing Values
print("\n--- Missing Values in Train Data ---")
print(train_df.isnull().sum())
print("\n--- Missing Value Percentage ---")
print(train_df.isnull().sum() / len(train_df) * 100)

# Categorical Feature Analysis
categorical_cols = train_df.select_dtypes(include='object').columns
for col in categorical_cols:
    if col != 'buynow_price': # Exclude target if it somehow got included
        print(f"\n--- Value Counts for {col} ---")
        print(train_df[col].value_counts())
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=col, y='buynow_price', data=train_df)
        plt.title(f'Price vs. {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Numerical Feature Analysis
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = numerical_cols.drop('buynow_price', errors='ignore') # Exclude target
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=col, y='buynow_price', data=train_df)
    plt.title(f'Price vs. {col}')
    plt.xlabel(col)
    plt.ylabel('Price')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(train_df[col].dropna(), bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Correlation Matrix (for numerical features)
plt.figure(figsize=(12, 10))
sns.heatmap(train_df[numerical_cols.tolist() + ['buynow_price']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()
```

#### **2.4 Feature Engineering (Initial Ideas)**

Based on EDA, think about potential new features or transformations.

- **'resolution (px)'**: This might be a string like '1920x1080'. Extract width and height, or compute pixel count.
    
- **'CPU model'**: Could be highly categorical, possibly requiring aggregation or more sophisticated encoding (e.g., embedding if many unique values).
    
- **'communications'**, **'input devices'**, **'multimedia'**: These might contain multiple values separated by commas. Consider one-hot encoding for each sub-feature or counting the number of features.
    
- **'RAM size'**, **'drive memory size (GB)'**, **'screen size'**: Check units and consistency.
    
- **'state'**: If it refers to new/used, it's an important categorical feature.
    

---

### **Step 3: Data Preprocessing Pipeline**

Define transformations for numerical and categorical features.

Python

```
# Separate target variable from features
X_train = train_df.drop('buynow_price', axis=1)
y_train = train_df['buynow_price'] # Use original price for RMSE calculation later

X_val = val_df.drop('buynow_price', axis=1)
y_val = val_df['buynow_price']

X_test = test_df.drop('buynow_price', axis=1)
y_test = test_df['buynow_price']

# Identify numerical and categorical features
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include='object').columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
# For features like 'communications', 'input devices', 'multimedia' that might contain multiple values:
# This needs a custom transformer or manual pre-processing before OneHotEncoder.
# For simplicity, initially, treat them as single categorical features.
# If they contain comma-separated values, OneHotEncoder will create features for each unique combined string.
# A more advanced approach would be to split these into multiple binary features.

# Let's handle 'resolution (px)' as an example of feature engineering within the pipeline setup
# This requires a custom transformer or processing it outside the main ColumnTransformer if it's complex.
# For now, let's assume it's directly one-hot encodable if it's a limited set of unique strings.
# If it's like '1920x1080', it's better to extract numerical values.

# Let's create a custom transformer for 'resolution (px)' and others that need specific handling
class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Handle 'resolution (px)'
        if 'resolution (px)' in X_copy.columns:
            resolution_split = X_copy['resolution (px)'].str.split('x', expand=True)
            X_copy['resolution_width'] = pd.to_numeric(resolution_split[0], errors='coerce')
            X_copy['resolution_height'] = pd.to_numeric(resolution_split[1], errors='coerce')
            X_copy['resolution_pixels'] = X_copy['resolution_width'] * X_copy['resolution_height']
            X_copy = X_copy.drop('resolution (px)', axis=1)

        # Handle features with multiple values (e.g., 'communications', 'input devices', 'multimedia')
        # This is a simplified approach, creating binary features for each unique item.
        # A more robust solution might involve TF-IDF or embeddings for very high cardinality.
        for col in ['communications', 'input devices', 'multimedia']:
            if col in X_copy.columns:
                # Fill NaN with an empty string for consistent splitting
                X_copy[col] = X_copy[col].fillna('')
                unique_values = X_copy[col].apply(lambda x: x.split(', ')).explode().unique()
                for val in unique_values:
                    if val: # Avoid creating a feature for empty strings if NaN was filled
                        X_copy[f'{col}_{val}'] = X_copy[col].apply(lambda x: 1 if val in x else 0)
                X_copy = X_copy.drop(col, axis=1)

        # Handle 'CPU clock speed (GHz)' - ensure it's numeric
        if 'CPU clock speed (GHz)' in X_copy.columns:
            X_copy['CPU clock speed (GHz)'] = pd.to_numeric(X_copy['CPU clock speed (GHz)'], errors='coerce')

        # Handle 'state' - simple one-hot encoding, but might need more specific handling if it's 'New'/'Used' etc.
        # Ensure it's treated as a regular categorical column for OneHotEncoder if it's not handled specifically here.

        return X_copy

from sklearn.base import BaseEstimator, TransformerMixin

# Identify which columns will be affected by the custom transformer
columns_to_process_with_custom_transformer = ['resolution (px)', 'communications', 'input devices', 'multimedia']
# These columns will be dropped by the custom transformer after processing.

# Update numerical and categorical features after the custom feature extraction conceptually
# This re-evaluation will happen once CustomFeatureExtractor is applied.
# For the ColumnTransformer, we need to specify initial column types.

# Separate initial numerical and categorical features
initial_numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
initial_categorical_features = X_train.select_dtypes(include='object').columns.tolist()

# Remove columns handled by CustomFeatureExtractor from initial lists
for col in columns_to_process_with_custom_transformer:
    if col in initial_categorical_features:
        initial_categorical_features.remove(col)
    if col in initial_numerical_features: # Unlikely for these specific columns, but good to check
        initial_numerical_features.remove(col)

# Preprocessing for categorical data (those not handled by custom extractor)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Use most_frequent for categorical
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # handle_unknown='ignore' for new categories in test set
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, initial_numerical_features),
        ('cat', categorical_transformer, initial_categorical_features)
    ],
    remainder='passthrough' # Keep other columns not explicitly transformed (e.g., newly created by custom transformer)
)

# Combine custom feature extraction and then the main preprocessor
# We need to apply CustomFeatureExtractor first, then identify the new numerical/categorical features.
# This makes the pipeline a bit more complex. A simpler way is to run CustomFeatureExtractor
# manually on X_train, X_val, X_test first, then define preprocessor.

# Let's apply CustomFeatureExtractor first to all datasets.
custom_extractor = CustomFeatureExtractor()
X_train_processed = custom_extractor.fit_transform(X_train)
X_val_processed = custom_extractor.transform(X_val)
X_test_processed = custom_extractor.transform(X_test)

# Now redefine numerical and categorical features based on the processed datasets
numerical_features_after_custom = X_train_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features_after_custom = X_train_processed.select_dtypes(include='object').columns.tolist()

# Final preprocessor for the post-custom-extracted data
preprocessor_final = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_after_custom),
        ('cat', categorical_transformer, categorical_features_after_custom)
    ],
    remainder='passthrough' # Should not be much left now.
)

print("\nPreprocessing pipeline defined.")
```

Important Note on ColumnTransformer and Custom Transformers:

When using ColumnTransformer directly, it's designed for column-wise transformations. For more complex, cross-column feature engineering or feature generation that changes the number/names of columns, it's often easier to implement it as a separate step before ColumnTransformer, or to create a Pipeline that contains both your custom transformer and then the ColumnTransformer. The second approach (applying CustomFeatureExtractor separately) is what I've demonstrated for clarity.

### **Step 4: Model Selection and Training**

Choose several models, train them, and evaluate their performance on the validation set.

#### **4.1 Model Candidates**

- **Linear Models:** `LinearRegression`, `Ridge`, `Lasso` (good baselines)
    
- **Tree-based Models:** `RandomForestRegressor`, `GradientBoostingRegressor` (often perform well)
    
- **Gradient Boosting Machines:** `XGBRegressor` (very powerful)
    

#### **4.2 Training and Evaluation Function**

Python

```
def evaluate_model(model, X_train_proc, y_train, X_val_proc, y_val):
    model.fit(X_train_proc, y_train)
    predictions = model.predict(X_val_proc)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)
    return rmse, r2

# Create full pipelines for each model
models = {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor_final),
                                        ('regressor', LinearRegression())]),
    'Ridge Regression': Pipeline(steps=[('preprocessor', preprocessor_final),
                                       ('regressor', Ridge(random_state=42))]),
    'Lasso Regression': Pipeline(steps=[('preprocessor', preprocessor_final),
                                       ('regressor', Lasso(random_state=42))]),
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor_final),
                                     ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))]),
    'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor_final),
                                        ('regressor', GradientBoostingRegressor(random_state=42))]),
    'XGBoost': Pipeline(steps=[('preprocessor', preprocessor_final),
                              ('regressor', XGBRegressor(random_state=42, n_jobs=-1))])
}

results = {}
for name, model in models.items():
    print(f"\nTraining and evaluating {name}...")
    # Need to pass the original X_train, y_train, X_val, y_val to the pipeline
    # The pipeline itself handles the custom feature extraction and then preprocessor_final.
    model.fit(X_train, y_train) # Fit the whole pipeline including custom_extractor
    y_pred_val = model.predict(X_val) # Predict on original X_val
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2_val = r2_score(y_val, y_pred_val)
    results[name] = {'RMSE': rmse_val, 'R2': r2_val}
    print(f"{name} - Validation RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}")

# Display all results
print("\n--- Model Comparison on Validation Set ---")
for name, metrics in results.items():
    print(f"{name}: RMSE = {metrics['RMSE']:.4f}, R2 = {metrics['R2']:.4f}")

# Sort by RMSE to find the best performing model
best_model_name = min(results, key=lambda k: results[k]['RMSE'])
best_model = models[best_model_name]
print(f"\nBest model on Validation Set: {best_model_name} with RMSE: {results[best_model_name]['RMSE']:.4f}")
```

**Important:** The `Pipeline` setup for `models` needs to incorporate `CustomFeatureExtractor` correctly. The current `preprocessor_final` is applied _after_ `CustomFeatureExtractor` has already been run. To make it a true end-to-end pipeline, `CustomFeatureExtractor` also needs to be part of the `Pipeline`.

Let's correct the pipeline definition:

Python

```
from sklearn.base import BaseEstimator, TransformerMixin

# Define CustomFeatureExtractor (as defined previously)
class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Handle 'resolution (px)'
        if 'resolution (px)' in X_copy.columns:
            # Fill NaN for resolution before splitting
            X_copy['resolution (px)'] = X_copy['resolution (px)'].fillna('')
            resolution_split = X_copy['resolution (px)'].str.split('x', expand=True)
            X_copy['resolution_width'] = pd.to_numeric(resolution_split[0], errors='coerce')
            X_copy['resolution_height'] = pd.to_numeric(resolution_split[1], errors='coerce')
            X_copy['resolution_pixels'] = X_copy['resolution_width'] * X_copy['resolution_height']
            X_copy = X_copy.drop('resolution (px)', axis=1)

        # Handle features with multiple values (e.g., 'communications', 'input devices', 'multimedia')
        for col in ['communications', 'input devices', 'multimedia']:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna('') # Fill NaN
                unique_items = set()
                # Collect all unique items across the entire column
                for item_list_str in X_copy[col].dropna():
                    unique_items.update(item_list_str.split(', '))

                for item in unique_items:
                    if item: # Avoid empty string if it was present
                        X_copy[f'{col}_{item.strip()}'] = X_copy[col].apply(lambda x: 1 if item.strip() in x.split(', ') else 0)
                X_copy = X_copy.drop(col, axis=1)

        # Handle 'CPU clock speed (GHz)' - ensure it's numeric
        if 'CPU clock speed (GHz)' in X_copy.columns:
            X_copy['CPU clock speed (GHz)'] = pd.to_numeric(X_copy['CPU clock speed (GHz)'], errors='coerce')

        return X_copy

# Now, define the full pipeline including CustomFeatureExtractor
# The order matters: CustomFeatureExtractor first, then the ColumnTransformer on the *transformed* data.

# First, apply CustomFeatureExtractor conceptually to determine columns for ColumnTransformer.
# This is a bit tricky with Pipeline, as ColumnTransformer needs fixed column names.
# A robust way is to define a dynamic ColumnTransformer, or determine the columns after a dummy run.

# For simplicity and correctness within a Pipeline, the best way is to let the ColumnTransformer
# operate on a fixed set of columns, and your custom transformer handles specific named columns
# and potentially adds new ones that the 'remainder' of ColumnTransformer can then handle.

# Let's adjust the preprocessing approach to handle dynamic columns within the pipeline
# by making the ColumnTransformer's column selection dynamic *after* the custom transform.
# This means we need a meta-pipeline or to fit the custom transformer and then define the CT.

# Simpler and more practical approach for the assignment:
# Apply CustomFeatureExtractor upfront to X_train, X_val, X_test.
# Then define ColumnTransformer based on the *results* of this transformation.

# Re-run CustomFeatureExtractor on the original dataframes to get the processed dataframes
custom_extractor = CustomFeatureExtractor()
X_train_processed = custom_extractor.fit_transform(X_train)
X_val_processed = custom_extractor.transform(X_val)
X_test_processed = custom_extractor.transform(X_test)

# Now, identify the numerical and categorical features from the *processed* dataframes
numerical_features_after_custom = X_train_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features_after_custom = X_train_processed.select_dtypes(include='object').columns.tolist()

# Define numerical and categorical transformers (as before)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the final preprocessor for the *already custom-extracted* data
preprocessor_final = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_after_custom),
        ('cat', categorical_transformer, categorical_features_after_custom)
    ],
    remainder='passthrough' # In case some columns are missed, keep them (though ideally none should be)
)


# Now, redefine the model pipelines using the 'preprocessor_final' on the already processed data.
# Note: The 'X_train' and 'X_val' passed to .fit() and .predict() will now be X_train_processed and X_val_processed.

models_final = {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor_final),
                                        ('regressor', LinearRegression())]),
    'Ridge Regression': Pipeline(steps=[('preprocessor', preprocessor_final),
                                       ('regressor', Ridge(random_state=42))]),
    'Lasso Regression': Pipeline(steps=[('preprocessor', preprocessor_final),
                                       ('regressor', Lasso(random_state=42))]),
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor_final),
                                     ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))]),
    'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor_final),
                                        ('regressor', GradientBoostingRegressor(random_state=42))]),
    'XGBoost': Pipeline(steps=[('preprocessor', preprocessor_final),
                              ('regressor', XGBRegressor(random_state=42, n_jobs=-1))])
}

results = {}
for name, model_pipeline in models_final.items():
    print(f"\nTraining and evaluating {name}...")
    model_pipeline.fit(X_train_processed, y_train) # Fit on processed training data
    y_pred_val = model_pipeline.predict(X_val_processed) # Predict on processed validation data
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2_val = r2_score(y_val, y_pred_val)
    results[name] = {'RMSE': rmse_val, 'R2': r2_val}
    print(f"{name} - Validation RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}")

# Display all results
print("\n--- Model Comparison on Validation Set ---")
for name, metrics in results.items():
    print(f"{name}: RMSE = {metrics['RMSE']:.4f}, R2 = {metrics['R2']:.4f}")

best_model_name = min(results, key=lambda k: results[k]['RMSE'])
best_model_pipeline = models_final[best_model_name]
print(f"\nBest model on Validation Set: {best_model_name} with RMSE: {results[best_model_name]['RMSE']:.4f}")
```

### **Step 5: Hyperparameter Tuning (for the best models)**

Focus on the top 1-2 performing models from Step 4. Use `GridSearchCV` or `RandomizedSearchCV` with cross-validation.

Python

```
from sklearn.model_selection import GridSearchCV

# Example for XGBoost (often a strong performer)
# Define a smaller parameter grid for demonstration, expand as needed.
# Note: This step can be very time-consuming.

# It's crucial that the `best_model_pipeline` is trained on `X_train_processed`.
# The search space should be for the 'regressor' step within the pipeline.

# Example for XGBoost:
if 'XGBoost' in models_final and results['XGBoost']['RMSE'] == results[best_model_name]['RMSE']:
    print("\n--- Hyperparameter Tuning for XGBoost ---")
    param_grid_xgb = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 7],
        'regressor__subsample': [0.7, 0.9, 1.0]
    }

    grid_search_xgb = GridSearchCV(
        estimator=best_model_pipeline, # Use the pipeline with preprocessor
        param_grid=param_grid_xgb,
        cv=KFold(n_splits=5, shuffle=True, random_state=42), # KFold for robust CV
        scoring='neg_root_mean_squared_error', # GridSearchCV minimizes, so use negative RMSE
        verbose=1,
        n_jobs=-1
    )

    grid_search_xgb.fit(X_train_processed, y_train) # Fit on processed training data

    print(f"Best parameters for XGBoost: {grid_search_xgb.best_params_}")
    best_xgb_rmse = -grid_search_xgb.best_score_ # Convert back to positive RMSE
    print(f"Best cross-validation RMSE for XGBoost: {best_xgb_rmse:.4f}")

    # Update best model if GridSearchCV improves it
    if best_xgb_rmse < results[best_model_name]['RMSE']:
        best_model_pipeline = grid_search_xgb.best_estimator_
        best_model_name = "Tuned XGBoost"
        results[best_model_name] = {'RMSE': best_xgb_rmse, 'R2': r2_score(y_val, best_model_pipeline.predict(X_val_processed))}
        print(f"Best model updated to Tuned XGBoost with improved RMSE: {results[best_model_name]['RMSE']:.4f}")

# Example for Random Forest:
if 'Random Forest' in models_final and results['Random Forest']['RMSE'] == results[best_model_name]['RMSE']:
    print("\n--- Hyperparameter Tuning for Random Forest ---")
    param_grid_rf = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None], # None means no limit
        'regressor__min_samples_split': [2, 5]
    }

    grid_search_rf = GridSearchCV(
        estimator=best_model_pipeline,
        param_grid=param_grid_rf,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    grid_search_rf.fit(X_train_processed, y_train)

    print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")
    best_rf_rmse = -grid_search_rf.best_score_
    print(f"Best cross-validation RMSE for Random Forest: {best_rf_rmse:.4f}")

    if best_rf_rmse < results[best_model_name]['RMSE']:
        best_model_pipeline = grid_search_rf.best_estimator_
        best_model_name = "Tuned Random Forest"
        results[best_model_name] = {'RMSE': best_rf_rmse, 'R2': r2_score(y_val, best_model_pipeline.predict(X_val_processed))}
        print(f"Best model updated to Tuned Random Forest with improved RMSE: {results[best_model_name]['RMSE']:.4f}")
```

### **Step 6: Final Evaluation on Test Set**

Once you have your final chosen model and hyperparameters, evaluate its performance on the completely unseen test set.

Python

```
# The 'best_model_pipeline' is already the best one after initial validation and tuning.
print(f"\n--- Final Evaluation on Test Set using {best_model_name} ---")

y_pred_test = best_model_pipeline.predict(X_test_processed)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
final_r2 = r2_score(y_test, y_pred_test)

print(f"Final Test RMSE: {final_rmse:.4f}")
print(f"Final Test R2: {final_r2:.4f}")

# Visualize predictions vs. actual values on the test set
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred_test, scatter_kws={'alpha':0.3})
plt.xlabel("Actual Prices (Test Set)")
plt.ylabel("Predicted Prices (Test Set)")
plt.title(f"Actual vs. Predicted Prices on Test Set ({best_model_name})")
plt.show()

# Residual plot
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Prices')
plt.show()
```

---

### **Step 7: Conclusion and Reflection**

Summarize your findings, discuss model limitations, and suggest future improvements.

**Conclusion:**

- State the best model found and its final RMSE on the test set.
    
- Briefly discuss the key features that seemed to influence the price (if you did feature importance analysis, otherwise speculate based on EDA).
    

**Reflection and Future Work:**

- **More Advanced Feature Engineering:**
    
    - Detailed parsing of 'CPU model' (e.g., extract brand, series, generation).
        
    - More sophisticated handling of multi-value categorical features (e.g., using TF-IDF for text-like attributes, or `MultiLabelBinarizer`).
        
    - Interactions between features (e.g., `RAM size` * `CPU clock speed`).
        
- **Outlier Treatment:** More robust methods beyond simple imputation, possibly removal or transformation.
    
- **Advanced Models:** Explore stacking/ensembling of the best performing models. Neural networks could also be considered for very complex relationships if data size permits.
    
- **Error Analysis:** Dive deeper into where the model makes large errors (e.g., specific price ranges, laptop configurations).
    
- **Data Quality:** If this were a real-world scenario, investigate the source of missing values and inconsistencies.
    
- **Deployment Considerations:** How would this model be deployed in a production environment? What latency requirements would it have?
    

---

This detailed plan should guide you through the Allegro take-home assignment. Remember to keep your Jupyter Notebook clean, well-commented, and reflective of your thought process, showing intermediate steps and discarded approaches. Good luck!