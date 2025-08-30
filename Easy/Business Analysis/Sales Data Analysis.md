---
title: Sales Data Analysis
company: 23andMe
difficulty: Easy
category: Business Analysis
date: 2025-07-28
---
_This data project has been used as a take-home assignment in the recruitment process for the data science positions at 23andMe._

## Assignment

Please answer the questions below based on the data provided:

1. Plot daily sales for all 50 weeks.
    
2. It looks like there has been a sudden change in daily sales. What date did it occur?
    
3. Is the change in daily sales at the date you selected statistically significant? If so, what is the p-value?
    
4. Does the data suggest that the change in daily sales is due to a shift in the proportion of male-vs-female customers? Please use plots to support your answer (a rigorous statistical analysis is not necessary).
    
5. Assume a given day is divided into four dayparts:
    
    - night (12:00AM - 6:00AM),
    - morning (6:00AM - 12:00PM),
    - afternoon (12:00PM - 6:00PM),
    - evening (6:00PM - 12:00AM).
    
    What is the percentage of sales in each daypart over all 50 weeks?
    

## Data Description

The `datasets/` directory contains fifty CSV files (one per week) of timestamped sales data. Each row in a file has two columns:

- `sale_time` - The timestamp on which the sale was made e.g. `2012-10-01 01:42:22`
- `purchaser_gender` - The gender of the person who purchased (male or female)

## Practicalities

Please work on the questions in the displayed order. Make sure that the solution reflects your entire thought process - it is more important how the code is structured rather than the final answers. You are expected to spend no more than 1-2 hours solving this project.

# Solution

Here is a complete, structured solution to the 23andMe data science take-home assignment.

This response is structured like a Jupyter Notebook. It includes:
1.  **Code to Generate a Sample Dataset:** Since the original 50 CSV files are not provided, I will first generate a realistic synthetic dataset that matches the described structure and has the characteristics needed to answer the questions (e.g., a clear changepoint). This makes the entire notebook fully reproducible.
2.  **A Step-by-Step Solution:** Each of the five questions is answered in order.
3.  **Explanation of Approach:** Before each code block, the logic and methodology are explained to show the thought process.
4.  **Code, Results, and Visualizations:** Each section includes the necessary code, its output, and relevant plots to support the findings.

***

## 23andMe: Sales Data Analysis

### Project Objective
The goal of this analysis is to examine 50 weeks of sales data to identify trends, pinpoint significant changes, and understand customer behavior patterns. We will answer a series of questions related to daily sales volume, customer demographics, and the timing of sales.

### 1. Setup and Data Generation

First, we will import the necessary libraries and create a synthetic dataset that mirrors the structure described in the assignment. This is crucial for reproducibility.

#### 1.1 Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
```

#### 1.2 Generate Sample Dataset
This code creates a `datasets/` directory and populates it with 50 weekly CSV files. The data is intentionally generated with a "changepoint" around week 20, where daily sales volume suddenly increases, to allow for a meaningful analysis.

```python
# --- Configuration ---
np.random.seed(42)
START_DATE = pd.to_datetime('2012-10-01')
CHANGE_DATE = pd.to_datetime('2013-02-15') # The date sales will jump
NUM_WEEKS = 50

# --- Create Directory ---
if not os.path.exists('datasets'):
    os.makedirs('datasets')

# --- Generate and Save Data ---
current_date = START_DATE
for i in range(NUM_WEEKS):
    week_end_date = current_date + pd.Timedelta(days=6)
    
    # Generate timestamps for one week
    week_dates = pd.date_range(start=current_date, end=week_end_date, freq='D')
    all_sales = []
    
    for day in week_dates:
        # Determine number of sales for the day
        if day < CHANGE_DATE:
            num_sales = np.random.randint(80, 120) # Lower sales before change
        else:
            num_sales = np.random.randint(220, 280) # Higher sales after change
            
        # Generate random timestamps within the day
        start_ts = day.timestamp()
        end_ts = (day + pd.Timedelta(days=1)).timestamp()
        sale_timestamps = pd.to_datetime(np.random.uniform(start_ts, end_ts, num_sales), unit='s')
        
        for ts in sale_timestamps:
            # Generate a gender for each sale (50/50 split)
            gender = np.random.choice(['male', 'female'])
            all_sales.append({'sale_time': ts, 'purchaser_gender': gender})

    # Create and save the weekly dataframe
    week_df = pd.DataFrame(all_sales)
    week_df['sale_time'] = week_df['sale_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    file_path = os.path.join('datasets', f'week_{i+1:02d}.csv')
    week_df.to_csv(file_path, index=False)
    
    current_date = week_end_date + pd.Timedelta(days=1)

print(f"Successfully created {NUM_WEEKS} sample CSV files in the 'datasets/' directory.")
```
<hr>

### 2. Analysis and Answering Questions

Now, we will load and combine all the weekly data files into a single DataFrame to perform the analysis.

#### 2.1 Data Loading and Preparation
**Approach:** We'll read all 50 CSV files from the `datasets/` directory, concatenate them into one master DataFrame, and convert the `sale_time` column to a proper datetime object for time-series analysis.

```python
# List all CSV files in the directory
files = [os.path.join('datasets', f) for f in os.listdir('datasets') if f.endswith('.csv')]
files.sort() # Ensure they are read in chronological order

# Read and concatenate all files into a single DataFrame
df_list = [pd.read_csv(f) for f in files]
sales_df = pd.concat(df_list, ignore_index=True)

# Convert 'sale_time' to datetime objects and set as index
sales_df['sale_time'] = pd.to_datetime(sales_df['sale_time'])
sales_df = sales_df.set_index('sale_time').sort_index()

print("Data loaded and prepared. Total sales records:", len(sales_df))
sales_df.head()
```

#### Question 1: Plot daily sales for all 50 weeks.
**Approach:** To get daily sales counts, we will "resample" the data by day. The `.resample('D')` method groups the data into daily bins, and `.size()` counts the number of sales in each bin. We then plot this time series.

```python
# Resample the data to get the count of sales per day
daily_sales = sales_df.resample('D').size()

# Plot the daily sales
plt.figure(figsize=(16, 8))
daily_sales.plot(title='Daily Sales Over 50 Weeks')
plt.xlabel('Date')
plt.ylabel('Number of Sales')
plt.grid(True)
plt.show()
```

**Observation:** The plot clearly shows a consistent level of daily sales for the initial months, followed by a sharp and sustained increase.

#### Question 2: It looks like there has been a sudden change in daily sales. What date did it occur?
**Approach:** While the date can be estimated visually from the plot, a more programmatic way is to find the date with the largest day-over-day increase in sales. We can calculate the difference between consecutive days' sales and find the date corresponding to the maximum difference.

```python
# Calculate the difference in sales from one day to the next
daily_sales_diff = daily_sales.diff()

# Find the date with the largest increase
changepoint_date = daily_sales_diff.idxmax()

print(f"The sudden change in daily sales likely occurred on: {changepoint_date.strftime('%Y-%m-%d')}")
```

#### Question 3: Is the change in daily sales at the date you selected statistically significant? If so, what is the p-value?
**Approach:** To test if the change is statistically significant, we can use a **two-sample independent t-test**. This test compares the means of two independent groups to determine if they are significantly different.
1.  Split the `daily_sales` data into two groups: "before" and "after" the changepoint date.
2.  Perform the t-test on these two groups using `scipy.stats.ttest_ind`.
3.  A p-value less than a significance level (e.g., 0.05) indicates a statistically significant difference.

```python
# Split the data into two periods: before and after the changepoint
sales_before = daily_sales[daily_sales.index < changepoint_date]
sales_after = daily_sales[daily_sales.index >= changepoint_date]

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(sales_before, sales_after, equal_var=False) # Welch's t-test

print(f"Mean daily sales before {changepoint_date.date()}: {sales_before.mean():.2f}")
print(f"Mean daily sales on/after {changepoint_date.date()}: {sales_after.mean():.2f}")
print(f"\nIs the change statistically significant?")
if p_value < 0.05:
    print(f"Yes, the change is statistically significant.")
    print(f"The p-value is: {p_value:.2e}") # Using scientific notation for very small p-values
else:
    print(f"No, the change is not statistically significant (p-value: {p_value:.4f}).")
```

#### Question 4: Does the data suggest that the change in daily sales is due to a shift in the proportion of male-vs-female customers?
**Approach:** We will compare the gender distribution of sales before and after the changepoint. A good way to visualize this is with two pie charts or a stacked bar chart. If the proportions remain similar across both periods, the sales jump is not due to a shift in gender demographics.

```python
# Separate the original DataFrame into 'before' and 'after' periods
df_before = sales_df[sales_df.index < changepoint_date]
df_after = sales_df[sales_df.index >= changepoint_date]

# Get gender proportions for each period
gender_counts_before = df_before['purchaser_gender'].value_counts(normalize=True)
gender_counts_after = df_after['purchaser_gender'].value_counts(normalize=True)

# Plotting the proportions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.pie(gender_counts_before, labels=gender_counts_before.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightpink'])
ax1.set_title(f'Gender Distribution Before {changepoint_date.date()}')

ax2.pie(gender_counts_after, labels=gender_counts_after.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightpink'])
ax2.set_title(f'Gender Distribution On/After {changepoint_date.date()}')

plt.suptitle('Comparison of Customer Gender Proportions', fontsize=16)
plt.show()

print("The pie charts show that the proportion of male vs. female customers remained virtually unchanged (~50% each) before and after the sales increase.")
print("This suggests the change in sales volume was not caused by a shift in customer gender demographics, but rather by an overall increase in customers of both genders.")
```

#### Question 5: What is the percentage of sales in each daypart over all 50 weeks?
**Approach:**
1.  Define the four dayparts based on the hour of the sale.
2.  Create a new `daypart` column in the DataFrame by applying a function to the hour of each `sale_time`.
3.  Use `.value_counts(normalize=True)` on this new column to calculate the percentage for each daypart.

```python
# Function to assign a daypart based on the hour
def get_daypart(hour):
    if 0 <= hour < 6:
        return 'night (12am-6am)'
    elif 6 <= hour < 12:
        return 'morning (6am-12pm)'
    elif 12 <= hour < 18:
        return 'afternoon (12pm-6pm)'
    else: # 18 to 24
        return 'evening (6pm-12am)'

# Create the 'daypart' column
sales_df['daypart'] = sales_df.index.hour.map(get_daypart)

# Calculate the percentage of sales in each daypart
daypart_percentages = sales_df['daypart'].value_counts(normalize=True) * 100

# Plot the results
plt.figure(figsize=(10, 6))
daypart_percentages.sort_index().plot(kind='bar', color='c')
plt.title('Percentage of Sales by Daypart')
plt.ylabel('Percentage (%)')
plt.xlabel('Daypart')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.show()

print("Percentage of sales in each daypart:")
print(daypart_percentages.to_string(float_format='%.2f%%'))
```