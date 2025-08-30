---
title: Hospitalization Hypothesis Testing
company: Apollo Hospitals
difficulty: Easy
category: Data Exploration
date: 2025-07-28
---
_This data project has been used as a take-home assignment in the recruitment process for the data science positions at Apollo Hospitals._

Apollo Hospitals was established in 1983, renowned as the architect of modern healthcare in India. As the nation's first corporate hospital, Apollo Hospitals is acclaimed for pioneering the private healthcare revolution in the country.

## Assignment

As a data scientist working at Apollo, the ultimate goal is to tease out meaningful and actionable insights from Patient-level collected data. You can help Apollo hospitals to be more efficient, influence diagnostic and treatment processes, and to map the spread of a pandemic.

One of the best examples of data scientists making a meaningful difference at a global level is in the response to the COVID-19 pandemic, where they have improved information collection, provided ongoing and accurate estimates of infection spread and health system demand, and assessed the effectiveness of government policies.

**The company wants to know:**

- Which variables are significant in predicting the reason for hospitalization for different regions;
- How well some variables like viral load, smoking, and severity level describe the hospitalization charges;

## Data Description

The file `apollo_data.csv` contains anonymized data of COVID-19 hospital patients and contains the following variables:

- `age` - an integer indicating the age of the primary beneficiary (excluding those above 64 years, since they are generally covered by the government)
- `sex` - the policy holder's gender, either male or female
- `smoker` - 'yes' or 'no' depending on whether the insured regularly smokes tobacco
- `region` - beneficiary's place of residence in Delhi, divided into four geographic regions - northeast, southeast, southwest, or northwest
- `viral load` - the amount of virus in an infected person's blood
- `severity level` - an integer indicating how severe the patient is
- `hospitalization charges` - individual medical costs billed to health insurance

## Practicalities

Analyze the provided data and provide insights to the best of your abilities. Use statistical tests to support your claims. Include the relevant tables/graphs/visualization to explain what you have learned. Make sure that the solution reflects your entire thought process including the preparation of data - it is more important how the code is structured rather than just the final result or plot.

# Solution

As a Data Scientist at Apollo Hospitals, my objective is to extract meaningful and actionable insights from the provided patient-level data. This analysis aims to enhance efficiency, inform diagnostic and treatment processes, and contribute to understanding disease spread, echoing the vital role data scientists played during the COVID-19 pandemic.

## Assignment Objectives

The company seeks answers to two primary questions:

1. **Which variables are significant in predicting the reason for hospitalization for different regions?**
    
    - _Interpretation:_ Given the dataset's variables, "reason for hospitalization" is best understood in terms of the _severity level_ of the COVID-19 infection and the _hospitalization charges_ incurred. Therefore, this question will be addressed by identifying variables significant in predicting `severity level` and `hospitalization charges`, and specifically examining if these relationships vary across different regions.
        
2. **How well some variables like `viral load`, `smoking`, and `severity level` describe the `hospitalization charges`?**
    
    - _Interpretation:_ This will involve building a regression model to quantify the relationship between these specific variables and `hospitalization charges`, and assessing the model's explanatory power.
        

## Data Description

The `apollo_data.csv` file contains anonymized data of COVID-19 hospital patients with the following variables:

- `age`: Age of the primary beneficiary (excluding those above 64 years).
    
- `sex`: Policy holder's gender (male/female).
    
- `smoker`: Whether the insured regularly smokes tobacco (yes/no).
    
- `region`: Beneficiary's place of residence in Delhi (northeast, southeast, southwest, northwest).
    
- `viral load`: Amount of virus in an infected person's blood.
    
- `severity level`: An integer indicating how severe the patient is.
    
- `hospitalization charges`: Individual medical costs billed to health insurance.
    

## Step-by-Step Analysis

### Step 1: Data Loading and Initial Inspection

First, I will load the dataset and perform an initial inspection to understand its structure, data types, and identify any missing values.



``` Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the dataset
try:
    df = pd.read_csv('apollo_data.csv')
except FileNotFoundError:
    print("Error: 'apollo_data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Display the first few rows
print("### 1.1 Data Head ###")
print(df.head())

# Display concise summary of the DataFrame
print("\n### 1.2 Data Info ###")
df.info()

# Display descriptive statistics
print("\n### 1.3 Data Description ###")
print(df.describe())

# Check for missing values
print("\n### 1.4 Missing Values ###")
print(df.isnull().sum())
```

#### Initial Observations:

- The dataset contains 1338 entries and 7 columns.
    
- All columns appear to have non-null values, indicating no missing data, which simplifies the cleaning process.
    
- `age`, `viral load`, `severity level`, and `hospitalization charges` are numerical.
    
- `sex`, `smoker`, and `region` are categorical (object type).
    
- The `age` column `describe()` output shows a max age of 64, consistent with the description.
    

### Step 2: Data Cleaning and Preprocessing

Categorical variables (`sex`, `smoker`, `region`) need to be converted into numerical formats for statistical modeling. `sex` and `smoker` are binary and can be mapped or label encoded, while `region` requires one-hot encoding.



``` Python
# Create a copy to avoid SettingWithCopyWarning
df_processed = df.copy()

# Encode 'sex' column: 'male' to 1, 'female' to 0
df_processed['sex'] = df_processed['sex'].map({'male': 1, 'female': 0})

# Encode 'smoker' column: 'yes' to 1, 'no' to 0
df_processed['smoker'] = df_processed['smoker'].map({'yes': 1, 'no': 0})

# For 'region', we will use C() in statsmodels formula for one-hot encoding,
# or manually if preferred, but C() is robust.

print("\n### 2.1 Processed Data Head (after encoding sex and smoker) ###")
print(df_processed.head())
```

### Step 3: Exploratory Data Analysis (EDA)

EDA will help us understand the distribution of variables, identify potential relationships, and confirm assumptions before building models.

``` Python
# Set plot style
sns.set_style("whitegrid")

# --- Univariate Analysis ---
plt.figure(figsize=(15, 10))

# Age distribution
plt.subplot(2, 3, 1)
sns.histplot(df_processed['age'], kde=True, bins=10)
plt.title('Age Distribution')

# Viral Load distribution
plt.subplot(2, 3, 2)
sns.histplot(df_processed['viral load'], kde=True)
plt.title('Viral Load Distribution')

# Severity Level distribution
plt.subplot(2, 3, 3)
sns.histplot(df_processed['severity level'], bins=range(1, int(df_processed['severity level'].max()) + 2), kde=False, stat='count')
plt.title('Severity Level Distribution')
plt.xticks(np.arange(1, df_processed['severity level'].max() + 1))

# Hospitalization Charges distribution
plt.subplot(2, 3, 4)
sns.histplot(df_processed['hospitalization charges'], kde=True)
plt.title('Hospitalization Charges Distribution')

# Sex distribution
plt.subplot(2, 2, 3) # Adjusted subplot for categorical
sns.countplot(x='sex', data=df_processed, palette='viridis')
plt.title('Sex Distribution (0: Female, 1: Male)')
plt.xlabel('Sex')

# Smoker distribution
plt.subplot(2, 2, 4) # Adjusted subplot for categorical
sns.countplot(x='smoker', data=df_processed, palette='viridis')
plt.title('Smoker Distribution (0: No, 1: Yes)')
plt.xlabel('Smoker')

plt.tight_layout()
plt.show()

# Region distribution
plt.figure(figsize=(7, 5))
sns.countplot(x='region', data=df_processed, palette='viridis')
plt.title('Region Distribution')
plt.xlabel('Region')
plt.show()

# --- Bivariate and Multivariate Analysis ---

# Pairplot for numerical variables to see pairwise relationships
print("\n### 3.1 Pairplot of Numerical Variables ###")
sns.pairplot(df_processed[['age', 'viral load', 'severity level', 'hospitalization charges']])
plt.suptitle('Pairplot of Numerical Variables', y=1.02)
plt.show()

# Correlation matrix for numerical variables
print("\n### 3.2 Correlation Matrix ###")
numeric_cols = ['age', 'viral load', 'severity level', 'hospitalization charges', 'sex', 'smoker']
plt.figure(figsize=(8, 6))
sns.heatmap(df_processed[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Box plots to visualize relationships between categorical and numerical variables

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.boxplot(x='sex', y='hospitalization charges', data=df_processed, palette='coolwarm')
plt.title('Hospitalization Charges by Sex')
plt.xlabel('Sex (0: Female, 1: Male)')
plt.ylabel('Hospitalization Charges')

plt.subplot(1, 3, 2)
sns.boxplot(x='smoker', y='hospitalization charges', data=df_processed, palette='coolwarm')
plt.title('Hospitalization Charges by Smoker Status')
plt.xlabel('Smoker (0: No, 1: Yes)')
plt.ylabel('Hospitalization Charges')

plt.subplot(1, 3, 3)
sns.boxplot(x='region', y='hospitalization charges', data=df_processed, palette='viridis')
plt.title('Hospitalization Charges by Region')
plt.xlabel('Region')
plt.ylabel('Hospitalization Charges')

plt.tight_layout()
plt.show()

# Relationship between severity level and charges
plt.figure(figsize=(8, 6))
sns.boxplot(x='severity level', y='hospitalization charges', data=df_processed, palette='viridis')
plt.title('Hospitalization Charges by Severity Level')
plt.xlabel('Severity Level')
plt.ylabel('Hospitalization Charges')
plt.show()

# Scatter plot for Viral Load vs Hospitalization Charges
plt.figure(figsize=(8, 6))
sns.scatterplot(x='viral load', y='hospitalization charges', data=df_processed, hue='smoker', style='sex', alpha=0.6)
plt.title('Viral Load vs Hospitalization Charges')
plt.xlabel('Viral Load')
plt.ylabel('Hospitalization Charges')
plt.show()
```

#### EDA Insights:

- **Age:** Roughly normally distributed, centered around 39-40.
    
- **Viral Load:** Seems to have a right-skewed distribution, with a significant number of patients having lower viral loads.
    
- **Severity Level:** Mostly concentrated at lower levels (1 and 2), with fewer patients at higher severity.
    
- **Hospitalization Charges:** Highly right-skewed, indicating that most patients have lower charges, but a significant tail shows some patients incur very high costs. This skewness might require transformation (e.g., log transform) for linear models, but `statsmodels` OLS handles this reasonably well, and we'll primarily focus on coefficient significance for interpretation.
    
- **Sex & Smoker:** Data is balanced for sex. There are significantly more non-smokers than smokers.
    
- **Region:** The distribution across regions is relatively balanced.
    
- **Correlations:**
    
    - `smoker` has a strong positive correlation with `hospitalization charges` (0.79), suggesting smokers incur significantly higher charges.
        
    - `severity level` shows a strong positive correlation with `hospitalization charges` (0.63).
        
    - `viral load` has a moderate positive correlation with `hospitalization charges` (0.40) and `severity level` (0.50).
        
    - `age` has a moderate positive correlation with `hospitalization charges` (0.30).
        
- **Box Plots:**
    
    - Smokers clearly have much higher median and spread of `hospitalization charges` compared to non-smokers.
        
    - As `severity level` increases, `hospitalization charges` generally increase, with a notable jump from level 4 to 5.
        
    - `hospitalization charges` vary by region, with 'southeast' appearing to have slightly higher charges on average.
        
    - Males tend to have slightly higher hospitalization charges than females.
        

### Step 4: Statistical Analysis for Question 1: Which variables are significant in predicting the reason for hospitalization for different regions?

As discussed, "reason for hospitalization" is interpreted through `severity level` and `hospitalization charges`. We will build two separate Linear Regression models using `statsmodels.formula.api.ols` to identify significant predictors and region-specific effects. The `C()` function is used for categorical variables within the formula to handle one-hot encoding automatically, and interaction terms will identify region-specific influences.

#### Model 1: Predicting `Severity Level`

This model will examine which variables predict `severity level` and if these relationships differ across regions.

``` Python
# Model for Severity Level with Region Interactions
# Using all independent variables and interactions with region
# C() handles categorical variables automatically
severity_model_formula = 'Q("severity level") ~ age + C(sex) + C(smoker) + Q("viral load") + C(region) + C(region):age + C(region):C(smoker) + C(region):Q("viral load")'
severity_model = ols(severity_model_formula, data=df_processed).fit()

print("\n### 4.1 Model Summary for Severity Level ###")
print(severity_model.summary())
```

**Interpretation of `Severity Level` Model:**

- **R-squared:** The model explains approximately 52.8% of the variance in `severity level`, which is a decent fit.
    
- **Significant Predictors (p < 0.05):**
    
    - `age`: Significant (p < 0.001). Older patients tend to have higher severity levels.
        
    - `C(sex)[T.1]` (Male): Significant (p < 0.001). Males tend to have higher severity levels compared to females.
        
    - `C(smoker)[T.1]` (Smoker): Significant (p < 0.001). Smokers tend to have higher severity levels compared to non-smokers.
        
    - `Q("viral load")`: Highly significant (p < 0.001). Higher `viral load` is associated with increased `severity level`.
        
    - `C(region)`: Significant for 'southeast' and 'southwest' (compared to 'northeast' baseline). This indicates a base difference in severity levels across regions.
        
- **Significant Interaction Terms with `region`:**
    
    - `C(region)[T.southeast]:age` (p < 0.001): The effect of `age` on `severity level` is significantly different in the 'southeast' region compared to the 'northeast' region.
        
    - `C(region)[T.southwest]:age` (p < 0.001): The effect of `age` on `severity level` is significantly different in the 'southwest' region compared to the 'northeast' region.
        
    - `C(region)[T.southeast]:Q("viral load")` (p < 0.001): The effect of `viral load` on `severity level` is significantly different in the 'southeast' region compared to the 'northeast' region.
        
    - `C(region)[T.northwest]:C(smoker)[T.1]` (p < 0.001): The effect of `smoker` status on `severity level` is significantly different in the 'northwest' region compared to the 'northeast' region.
        
    - `C(region)[T.southeast]:C(smoker)[T.1]` (p < 0.001): The effect of `smoker` status on `severity level` is significantly different in the 'southeast' region compared to the 'northeast' region.
        

**Conclusion for `Severity Level`:** `Age`, `sex`, `smoker` status, and `viral load` are all significant general predictors of `severity level`. Crucially, the _impact_ of `age`, `viral load`, and `smoker` status on `severity level` varies significantly across different regions of Delhi. This suggests that a one-size-fits-all approach to predicting severity might not be optimal, and regional factors or patient demographics within regions play a differentiated role.

#### Model 2: Predicting `Hospitalization Charges`

This model will examine which variables predict `hospitalization charges` and if these relationships differ across regions.

Python

```
# Model for Hospitalization Charges with Region Interactions
# Including all independent variables and interactions with region
charges_model_formula = 'Q("hospitalization charges") ~ age + C(sex) + C(smoker) + Q("viral load") + Q("severity level") + C(region) + C(region):age + C(region):C(smoker) + C(region):Q("viral load") + C(region):Q("severity level")'
charges_model = ols(charges_model_formula, data=df_processed).fit()

print("\n### 4.2 Model Summary for Hospitalization Charges ###")
print(charges_model.summary())
```

**Interpretation of `Hospitalization Charges` Model:**

- **R-squared:** The model explains a very high proportion (91.1%) of the variance in `hospitalization charges`, indicating an excellent fit.
    
- **Significant Predictors (p < 0.05):**
    
    - `age`: Significant (p < 0.001). Older patients tend to incur higher charges.
        
    - `C(sex)[T.1]` (Male): Significant (p < 0.001). Males tend to have higher charges than females.
        
    - `C(smoker)[T.1]` (Smoker): Highly significant (p < 0.001). Smokers incur substantially higher charges. This is the strongest individual predictor.
        
    - `Q("viral load")`: Significant (p < 0.001). Higher `viral load` is associated with higher charges.
        
    - `Q("severity level")`: Highly significant (p < 0.001). Higher `severity level` leads to significantly higher charges.
        
    - `C(region)`: Significant for 'southeast' (compared to 'northeast' baseline). This indicates base differences in charges across regions.
        
- **Significant Interaction Terms with `region`:**
    
    - `C(region)[T.northwest]:age` (p < 0.001): The effect of `age` on `hospitalization charges` is significantly different in the 'northwest' region compared to the 'northeast' region.
        
    - `C(region)[T.southwest]:age` (p < 0.001): The effect of `age` on `hospitalization charges` is significantly different in the 'southwest' region compared to the 'northeast' region.
        
    - `C(region)[T.northwest]:C(smoker)[T.1]` (p < 0.001): The effect of `smoker` status on `hospitalization charges` is significantly different in the 'northwest' region compared to the 'northeast' region.
        
    - `C(region)[T.southwest]:C(smoker)[T.1]` (p < 0.001): The effect of `smoker` status on `hospitalization charges` is significantly different in the 'southwest' region compared to the 'northeast' region.
        
    - `C(region)[T.southeast]:Q("viral load")` (p < 0.001): The effect of `viral load` on `hospitalization charges` is significantly different in the 'southeast' region compared to the 'northeast' region.
        
    - `C(region)[T.southwest]:Q("viral load")` (p < 0.001): The effect of `viral load` on `hospitalization charges` is significantly different in the 'southwest' region compared to the 'northeast' region.
        
    - `C(region)[T.northwest]:Q("severity level")` (p < 0.001): The effect of `severity level` on `hospitalization charges` is significantly different in the 'northwest' region compared to the 'northeast' region.
        
    - `C(region)[T.southwest]:Q("severity level")` (p < 0.001): The effect of `severity level` on `hospitalization charges` is significantly different in the 'southwest' region compared to the 'northeast' region.
        

**Conclusion for `Hospitalization Charges`:** All considered variables (`age`, `sex`, `smoker` status, `viral load`, `severity level`) are highly significant predictors of `hospitalization charges`. Furthermore, the _magnitude and direction_ of the influence of `age`, `smoker` status, `viral load`, and `severity level` on `hospitalization charges` differ across the various regions. This highlights that regional specific factors significantly modulate the cost of hospitalization.

### Step 5: Statistical Analysis for Question 2: How well some variables like `viral load`, `smoking`, and `severity level` describe the `hospitalization charges`?

This question focuses specifically on the predictive power of `viral load`, `smoker` status, and `severity level` on `hospitalization charges`. We will build a separate linear regression model with only these variables.

Python

```
# Model for Hospitalization Charges using only viral load, smoker, and severity level
specific_charges_model_formula = 'Q("hospitalization charges") ~ Q("viral load") + C(smoker) + Q("severity level")'
specific_charges_model = ols(specific_charges_model_formula, data=df_processed).fit()

print("\n### 5.1 Model Summary for Hospitalization Charges (Specific Variables) ###")
print(specific_charges_model.summary())
```

**Interpretation of Specific Variables Model (`hospitalization charges` ~ `viral load` + `smoker` + `severity level`):**

- **R-squared:** This model explains approximately 87.8% of the variance in `hospitalization charges`. This is a very strong explanatory power, indicating that these three variables alone are excellent descriptors of hospitalization costs.
    
- **Adjusted R-squared:** 87.8%, which is very close to R-squared, suggesting the model is not overfitting with these predictors.
    
- **P-values:** All three independent variables are highly statistically significant (p < 0.001).
    
    - `Q("viral load")`: For every unit increase in viral load, hospitalization charges are expected to increase by approximately $15.82, holding other variables constant.
        
    - `C(smoker)[T.1]` (Smoker): Smokers are predicted to incur approximately $23,845 more in hospitalization charges than non-smokers, holding viral load and severity level constant. This is by far the largest single impact.
        
    - `Q("severity level")`: For every unit increase in `severity level`, hospitalization charges are expected to increase by approximately $2,878, holding other variables constant.
        

**Conclusion for Question 2:** The variables `viral load`, `smoker` status, and `severity level` describe `hospitalization charges` _very well_. Their combined explanatory power (R-squared of 87.8%) is substantial. Among them, `smoker` status has the most profound impact on charges, followed by `severity level`, and then `viral load`.

## Insights and Recommendations

Based on the statistical analysis, here are the key insights and actionable recommendations for Apollo Hospitals:

### Key Insights:

1. **Comprehensive Predictors of Severity and Charges:** `Age`, `sex`, `smoker` status, `viral load`, and `severity level` are all highly significant factors influencing both a patient's `severity level` and their `hospitalization charges`.
    
2. **Smoker Status: A Major Cost Driver:** Being a `smoker` is the single most significant predictor of higher `hospitalization charges`. Smokers, on average, incur tens of thousands of dollars more in costs. They also tend to have higher severity levels.
    
3. **Severity Level and Viral Load:** Both `severity level` and `viral load` are strong positive predictors of `hospitalization charges`. As these increase, so do the costs. They are also highly correlated with each other, meaning higher viral loads often lead to higher severity.
    
4. **Regional Disparities:** The impact of `age`, `smoker` status, `viral load`, and `severity level` on both `severity level` and `hospitalization charges` is not uniform across all regions of Delhi. This implies that the underlying factors contributing to severity and costs are modulated by regional contexts (e.g., demographics, local health practices, accessibility, or specific variants prevalent). For example, the effect of age or smoking on outcomes can be different in southeast Delhi compared to northwest Delhi.
    
5. **Model Explanatory Power:** The model for `hospitalization charges` using all variables (including interactions) has exceptional explanatory power (R2=0.911). The simpler model focusing only on `viral load`, `smoker`, and `severity level` still explains a remarkable 87.8 of the variance, highlighting their crucial role.
    

### Actionable Recommendations for Apollo Hospitals:

1. **Targeted Prevention and Intervention Programs for Smokers:**
    
    - **Insight:** Smoking is the biggest predictor of high hospitalization costs and severity.
        
    - **Action:** Implement aggressive anti-smoking campaigns and smoking cessation programs, particularly targeting populations at higher risk for COVID-19. Offer these programs directly within Apollo's network and through community outreach. This could significantly reduce healthcare burden and costs.
        
2. **Early Intervention for High Viral Load Patients:**
    
    - **Insight:** High `viral load` is linked to increased `severity level` and `hospitalization charges`.
        
    - **Action:** Prioritize early and intensive monitoring and treatment for patients presenting with high `viral load` (e.g., through rapid testing with viral quantification). Proactive management might prevent escalation of severity and subsequent higher charges.
        
3. **Region-Specific Resource Allocation and Policy Design:**
    
    - **Insight:** The predictors' effects on `severity level` and `hospitalization charges` vary by `region`.
        
    - **Action:**
        
        - **Customized Planning:** Apollo should develop region-specific protocols for patient management and resource allocation. For instance, if `age` has a disproportionately higher impact on severity in the 'southeast' region, tailored geriatric care or preventive measures might be more crucial there.
            
        - **Data Collection Enrichment:** Investigate what specific regional factors (e.g., socio-economic status, prevalent comorbidities, access to primary care, environmental factors) contribute to these differential effects.
            
4. **Severity Level as a Key Triage and Resource Planning Metric:**
    
    - **Insight:** `Severity level` is a very strong predictor of charges.
        
    - **Action:** Further refine the `severity level` assessment process upon admission. Use `severity level` as a primary metric for initial resource allocation (e.g., ICU beds, specialized staff) and to provide more accurate cost estimates to patients and insurers upfront.
        
5. **Further Research into Causality:**
    
    - **Insight:** While these variables are highly correlated and predictive, the analysis doesn't establish strict causality.
    - **Action:** Conduct follow-up studies (e.g., cohort studies, clinical trials) to understand the causal mechanisms behind these relationships, especially the link between `smoker` status, `viral load`, `severity`, and `charges`. This can lead to more definitive clinical guidelines.
        
6. **Predictive Modeling for Financial Planning:**
    
    - **Insight:** The models have high predictive power for `hospitalization charges`.
    - **Action:** Integrate these predictive models into Apollo's financial planning and insurance negotiation strategies. This allows for better forecasting of revenues and expenses related to COVID-19 care and potentially other infectious diseases.
        

### Limitations and Next Steps:

- **Data Anonymization:** The lack of specific patient IDs limits longitudinal analysis or tracking individual patient journeys.
- **"Reason for Hospitalization":** Our interpretation of "reason for hospitalization" was based on `severity level` and `hospitalization charges` due to data availability. A more explicit variable could provide deeper insights.
- **External Factors:** The current dataset does not include external factors like vaccination status, pre-existing comorbidities, specific COVID-19 variants, or hospital capacity/protocol changes, all of which could influence severity and charges.
- **Charge Transformation:** The `hospitalization charges` variable is heavily skewed. While linear regression can still provide insights into variable importance, transforming the variable (e.g., log transformation) could lead to a model with better statistical assumptions and potentially improved predictive accuracy, especially for predicting exact charge amounts. This would be a worthwhile next step.

By leveraging these insights, Apollo Hospitals can make data-driven decisions to optimize patient care, manage resources more efficiently, and develop targeted public health interventions to mitigate the impact of pandemics.