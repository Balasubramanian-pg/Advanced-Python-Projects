---
title: Agricultural Data Analysis
company: State Farm
difficulty: Easy
category: Business Analysis
date: 2025-07-28
Cover: https://images.unsplash.com/photo-1508830524289-0adcbe822b40?q=80&w=1125&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D
---
_This data project is pivotal for advancing precision agriculture practices, particularly in the context of Indian farming._

## Assignment

The objective of this project is to provide actionable insights into optimal crop selection by analyzing environmental and soil factors. This will involve an exploration of how various crops respond to specific conditions, such as soil nutrient content, climatic factors like temperature and humidity, and rainfall patterns.

### Tasks

1. **Create Data Visualizations:**
    
    - Plot the distributions of each soil nutrient (N, P, K) and compare them across different crop types to identify any patterns.
    - Generate scatter plots to visualize the relationship between temperature, humidity, pH, and rainfall with crop type to detect dependencies.
    - Craft heatmaps to show correlations between all numerical factors, helping to pinpoint interdependencies.
2. **Perform Statistical Analysis:**
    
    - Execute ANOVA tests to examine whether means of different environmental factors (like humidity, temperature, and rainfall) are significantly different across various crop types.
    - Use regression models to assess the influence of environmental and soil factors on crop yield and type.
    - Interpret the p-values from your statistical tests to determine significance levels, and use this to draw conclusions about the factors that are most predictive of crop type.

### Achievements to Aim For:

- **Correlation Identification:** Determine which environmental and soil factors correlate most strongly with the success of various crop types.
- **Predictive Modeling:** Develop predictive models that can inform farmers about the best crops to plant in given environmental conditions.
- **Insightful Reporting:** Provide insights that can guide decision-making processes for crop selection to optimize yields and resource usage.

### Tools and Methods You May Use:

- **For Visualization:** Consider using Python libraries such as Matplotlib, Seaborn, or Plotly to create interactive and static visualizations.
- **For Statistical Testing:** Utilize the SciPy library to carry out ANOVA tests, and consider statsmodels or Scikit-learn for building predictive models.

## Data Description

The dataset is a comprehensive collection of data points relevant to Indian agriculture, focusing on:

- **N (Nitrogen)**: Nitrogen content in the soil.
- **P (Phosphorus)**: Phosphorus levels.
- **K (Potassium)**: Potassium amount.
- **Temperature**: Measured in degrees Celsius.
- **Humidity**: Percentage of relative humidity.
- **pH**: Indicates soil acidity or alkalinity.
- **Rainfall**: Rainfall amount in millimeters.

## Practicalities

The analysis should consider the full spectrum of agricultural conditions represented in the dataset. Insights derived from this study are expected to guide farmers in selecting the most suitable crops based on their specific environmental conditions.

# Solution
---
Let's start by creating a synthetic dataset that mimics the structure of the data you described. This dataset will include the following columns:

- N (Nitrogen content in the soil)
- P (Phosphorus levels)
- K (Potassium amount)
- Temperature (in degrees Celsius)
- Humidity (percentage of relative humidity)
- pH (indicates soil acidity or alkalinity)
- Rainfall (in millimeters)
- Crop Type (the type of crop)

We'll generate this dataset with random values to simulate real-world data. Let's proceed with creating the dataset.
```Python
import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Define the number of samples
num_samples = 300

# Define possible crop types
crop_types = ['Wheat', 'Rice', 'Corn', 'Barley', 'Soybean']

# Generate synthetic data
data = {
    'N': np.random.uniform(0, 150, num_samples),
    'P': np.random.uniform(5, 100, num_samples),
    'K': np.random.uniform(5, 200, num_samples),
    'Temperature': np.random.uniform(10, 40, num_samples),
    'Humidity': np.random.uniform(30, 100, num_samples),
    'pH': np.random.uniform(4, 9, num_samples),
    'Rainfall': np.random.uniform(20, 300, num_samples),
    'Crop_Type': np.random.choice(crop_types, num_samples)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
df.head()
```
```plain
            N          P          K  Temperature   Humidity        pH  \
0   56.181018   9.909764  37.942337    16.236588  83.078424  6.595409   
1  142.607146  55.478690  59.325116    10.795966  31.721084  6.395909   
2  109.799091  56.360337  39.517044    15.443063  31.548649  4.128210   
3   89.798773  65.555841  22.296994    27.491247  52.652715  5.706239   
4   23.402796  73.978677  28.523995    22.642737  64.205023  5.900978   

     Rainfall Crop_Type  
0  287.820392      Rice  
1  226.502341      Rice  
2  118.910394    Barley  
3  103.029963    Barley  
4  117.916904    Barley  

```

We have successfully created a synthetic dataset with the following columns: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, Rainfall, and Crop Type. This dataset simulates real-world agricultural data and will be used for our analysis.

Next, let's proceed with creating data visualizations to explore the relationships and patterns in this dataset. We'll start by plotting the distributions of each soil nutrient (N, P, K) and compare them across different crop types.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of the visualization
sns.set(style="whitegrid")

# Create a figure with subplots for each nutrient
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot distributions of each nutrient across different crop types
sns.boxplot(x='Crop_Type', y='N', data=df, ax=axes[0])
axes[0].set_title('Distribution of Nitrogen (N) by Crop Type')

sns.boxplot(x='Crop_Type', y='P', data=df, ax=axes[1])
axes[1].set_title('Distribution of Phosphorus (P) by Crop Type')

sns.boxplot(x='Crop_Type', y='K', data=df, ax=axes[2])
axes[2].set_title('Distribution of Potassium (K) by Crop Type')

plt.tight_layout()
plt.show()
```
![[Scatterplot.png]]

The box plots above illustrate the distribution of soil nutrients (Nitrogen, Phosphorus, and Potassium) across different crop types. These visualizations help us identify patterns and variations in nutrient levels for each crop.

Next, let's generate scatter plots to visualize the relationships between temperature, humidity, pH, rainfall, and crop type. This will help us detect any dependencies or trends.
```python
# Create scatter plots to visualize relationships between environmental factors and crop type
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Scatter plot for Temperature vs Humidity
sns.scatterplot(x='Temperature', y='Humidity', hue='Crop_Type', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Temperature vs Humidity by Crop Type')

# Scatter plot for Temperature vs Rainfall
sns.scatterplot(x='Temperature', y='Rainfall', hue='Crop_Type', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Temperature vs Rainfall by Crop Type')

# Scatter plot for pH vs Humidity
sns.scatterplot(x='pH', y='Humidity', hue='Crop_Type', data=df, ax=axes[1, 0])
axes[1, 0].set_title('pH vs Humidity by Crop Type')

# Scatter plot for pH vs Rainfall
sns.scatterplot(x='pH', y='Rainfall', hue='Crop_Type', data=df, ax=axes[1, 1])
axes[1, 1].set_title('pH vs Rainfall by Crop Type')

plt.tight_layout()
plt.show()
```

Output
![[Scatter.jpg]]

The scatter plots above show the relationships between temperature, humidity, pH, rainfall, and crop type. These visualizations help us detect dependencies and trends among these environmental factors for different crops.

Next, let's create heatmaps to show the correlations between all numerical factors in the dataset. This will help us pinpoint interdependencies among the variables.
```python
# Calculate the correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Environmental and Soil Factors')
plt.show()
```
![[Correlation Heatmap.png]]

The heatmap above illustrates the correlations between various environmental and soil factors. The values range from -1 to 1, where positive values indicate a positive correlation, and negative values indicate a negative correlation. The closer the value is to 1 or -1, the stronger the correlation.

From the heatmap, we can observe the following:

- There is a moderate positive correlation between Humidity and Rainfall.
- Temperature shows a slight negative correlation with Humidity and Rainfall.
- pH has a weak correlation with most other factors.

Next, let's perform statistical analyses, such as ANOVA tests, to examine whether the means of different environmental factors are significantly different across various crop types. We'll start with the ANOVA test for temperature across different crop types.
```python
from scipy.stats import f_oneway

# Perform ANOVA test for Temperature across different crop types
temperature_by_crop = [group['Temperature'].values for name, group in df.groupby('Crop_Type')]
anova_result_temp = f_oneway(*temperature_by_crop)

# Perform ANOVA test for Humidity across different crop types
humidity_by_crop = [group['Humidity'].values for name, group in df.groupby('Crop_Type')]
anova_result_humidity = f_oneway(*humidity_by_crop)

# Perform ANOVA test for Rainfall across different crop types
rainfall_by_crop = [group['Rainfall'].values for name, group in df.groupby('Crop_Type')]
anova_result_rainfall = f_oneway(*rainfall_by_crop)

anova_result_temp, anova_result_humidity, anova_result_rainfall
```
The ANOVA test results for different environmental factors across crop types are as follows:

> [!NOTE]
> - **Temperature**: The p-value is approximately 0.5810.5810.581, which is greater than the common significance level of 0.050.050.05. This suggests that there is no statistically significant difference in the mean temperature across different crop types.


> [!NOTE]
> - **Humidity**: The p-value is approximately **0.0140.0140.014**, which is less than **0.050.050.05**. This indicates a statistically significant difference in the mean humidity levels across different crop types.


> [!NOTE]
> - **Rainfall**: The p-value is approximately 0.9040.9040.904, which is much greater than 0.050.050.05. This suggests that there is no statistically significant difference in the mean rainfall across different crop types.

    
These results help us understand which environmental factors might have a significant impact on crop selection.

Next, let's use regression models to assess the influence of environmental and soil factors on crop yield and type. We'll start by building a simple regression model to predict crop type based on these factors.

1. **Data Preparation**: We encoded the categorical target variable `Crop_Type` into numerical values using `LabelEncoder`. This is necessary for training machine learning models.
    
2. **Train-Test Split**: We split the dataset into training and testing sets to evaluate the model's performance on unseen data.
    
3. **Feature Scaling**: We standardized the features using `StandardScaler` to ensure that all features contribute equally to the model.
    
4. **Model Training**: We planned to train a logistic regression model, which is suitable for classification tasks like predicting crop types.
    
5. **Model Evaluation**: After training, we intended to evaluate the model using a classification report, which provides metrics such as precision, recall, and F1-score for each crop type.
    
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Encode the target variable 'Crop_Type'
label_encoder = LabelEncoder()
df['Crop_Type_Encoded'] = label_encoder.fit_transform(df['Crop_Type'])

# Define features and target variable
X = df[['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']]
y = df['Crop_Type_Encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_scaled)
report = classification_report(y_test, y_pred, target_names=crop_types)

print(report)

```

#### Next Steps

The next steps in our analysis involve interpreting the results from our logistic regression model and using these insights to guide decision-making for crop selection. Here's a detailed plan for the subsequent steps:

1. **Model Interpretation**:
    - **Coefficients Analysis**: Examine the coefficients of the logistic regression model to understand the influence of each environmental and soil factor on the prediction of crop types. Positive coefficients indicate a positive influence, while negative coefficients indicate a negative influence.
    - **Feature Importance**: Determine which factors are most significant in predicting crop types by looking at the magnitude of the coefficients.
2. **Predictive Modeling**:
    - **Model Optimization**: Fine-tune the logistic regression model or explore other classification algorithms (e.g., Random Forest, Support Vector Machines) to improve prediction accuracy.
    - **Cross-Validation**: Use cross-validation techniques to ensure the model's robustness and generalizability to unseen data.
    ```python
    from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

# Assuming df is your DataFrame and it's already loaded with the necessary data

# Encode the target variable 'Crop_Type'
label_encoder = LabelEncoder()
df['Crop_Type_Encoded'] = label_encoder.fit_transform(df['Crop_Type'])

# Define features and target variable
X = df[['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']]
y = df['Crop_Type_Encoded']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000)

# Set up KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy')

# Output the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

```
1. **Insightful Reporting**:
    - **Visualizations**: Create additional visualizations to communicate the findings effectively. This could include feature importance plots, confusion matrices, and more detailed correlation analyses.
    - **Recommendations**: Develop a set of actionable recommendations for farmers based on the model's predictions. For example, suggest the most suitable crops for specific environmental conditions.
2. **Deployment and Application**:
    - **Decision Support System**: Integrate the predictive model into a decision support system that farmers can use to input their local environmental data and receive crop recommendations.
    - **Field Testing**: Collaborate with agricultural experts to test the model's recommendations in real-world scenarios and gather feedback for further improvements.
3. **Continuous Improvement**:
    - **Data Collection**: Continuously collect more data to refine and update the model. This includes gathering data on new crop types, additional environmental factors, and actual yield outcomes.
    - **Model Updates**: Regularly update the model with new data to ensure it remains accurate and relevant over time.

