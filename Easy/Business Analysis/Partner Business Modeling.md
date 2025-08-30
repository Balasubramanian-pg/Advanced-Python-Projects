---
title: Partner Business Modeling
company: Uber
difficulty: Easy
category: Business Analysis
date: 2025-07-28
---
_This data project has been used as a take-home assignment in the recruitment process for the data science positions at Uber._

## Assignment

_**Scenario 1:**_ _It is going to be a huge Saturday and there will need to be many more cars on the road than last week. In order to get drivers to go online, we're assessing the following two bonus options in terms of cost:_

- _Option 1: $50 for each driver that is online at least 8 hours, accepts 90% of requests, completes 10 trips, and has a rating of 4.7 or better during the time frame;_
- _Option 2: $4/trip for all drivers who complete 12 trips, and have a 4.7 or better rating._

Using the dataset provided and given Scenario 1, provide answers to the questions below:

1. How much would the total bonus payout be with Option 1?
2. How much would the total bonus payout be with Option 2?
3. How many drivers would qualify for a bonus under Option 1 but not under Option 2?
4. What percentages of drivers online completed less than 10 trips, had an acceptance rate of less than 90%, and had a rating of 4.7 or higher?

_**Scenario 2:** A taxi driver currently generates $200 per day in fares (before expenses), works six days a week, takes three weeks off, and has the following expenses:_

- _Gas - $200 per week_
- _Insurance - $400 per month_
- _Vehicle rent (by the week) - $500_

_The driver doesn't pay gas and rent expenses on off weeks._

_Now, let's assume that the same driver would buy a Town Car and partner with Uber. If he does, his gas expenses would go up by 5%, his insurance expense would decrease by 20%, and he would no longer be renting a vehicle. However, he would need to buy a car. The driver would still take three weeks off per year._

Given Scenario 2, provide answers to the questions below:

5. How much money (after expenses) does the taxi driver make per year without partnering with Uber?
6. You are convincing the same driver above to buy a Town Car and partner with Uber. Assuming the new car is 40,000 USD, how much would the driver's gross fares need to increase per week to fully pay for the car in year 1 and maintain the same yearly profit margin as before?

## Data Description

The dataset is given in the file `dataset_2.csv`. The data come from rides on a busy Saturday and are aggregated on a driver level. The dataset provides information on the total number of completed trips, the driver's acceptance rate, the total number of hours on duty, and the average rating.

Only use this dataset to answer questions 1-4. There is no dataset provided for questions 5 and 6 as these are solely based on a given scenario.

## Practicalities

Please work on the questions in the displayed order. Make sure that the solution reflects your entire thought process - it is more important how the code is structured rather than the final answers. You are expected to spend no more than 2 hours solving this project.

# Solution
Here is a complete, structured solution to the Uber data science take-home assignment, covering both scenarios.

This response is structured like a Jupyter Notebook. It includes:
1.  **Code to Generate a Sample Dataset:** As `dataset_2.csv` is not provided, I will first generate a realistic synthetic dataset for Scenario 1 to ensure the code is fully runnable.
2.  **A Step-by-Step Solution:** Each question is answered in the specified order.
3.  **Explanation of Approach:** Before each code block or calculation, the logic and methodology are clearly explained.
4.  **Code and Calculations:** The notebook provides the necessary code for Scenario 1 and detailed manual calculations for Scenario 2.

***

## Uber: Driver Bonus and Profitability Analysis

### Project Objective
This analysis addresses two independent business scenarios. The first involves calculating the cost of two potential driver bonus structures based on performance data. The second involves a profitability analysis to help a taxi driver understand the financial implications of switching to the Uber platform.

### 1. Setup and Data Generation (for Scenario 1)

Before tackling the questions for Scenario 1, we will import the necessary libraries and create a synthetic `dataset_2.csv` that reflects the described data attributes.

#### 1.1 Import Libraries
```python
import pandas as pd
import numpy as np
```

#### 1.2 Generate Sample Dataset for Scenario 1
This code creates `dataset_2.csv` with 1,000 drivers. The data is randomized but designed to have a distribution that makes the bonus qualification criteria meaningful.

```python
# --- Configuration ---
NUM_DRIVERS = 1000
np.random.seed(42) # For reproducibility

# --- Create Driver Data ---
driver_data = {
    'driver_id': np.arange(1, NUM_DRIVERS + 1),
    'trips_completed': np.random.randint(0, 25, size=NUM_DRIVERS),
    'acceptance_rate': np.clip(np.random.normal(0.92, 0.1, size=NUM_DRIVERS), 0.7, 1.0).round(2),
    'hours_on_duty': np.random.uniform(2, 12, size=NUM_DRIVERS).round(1),
    'average_rating': np.clip(np.random.normal(4.75, 0.15, size=NUM_DRIVERS), 4.0, 5.0).round(2)
}
df = pd.DataFrame(driver_data)

# Save to CSV to simulate loading
df.to_csv('dataset_2.csv', index=False)

print("Sample 'dataset_2.csv' created for Scenario 1.")
df.head()
```

<hr>

### 2. Scenario 1: Driver Bonus Payout Analysis

Now, we will use the generated dataset to answer the first four questions related to the bonus options.

#### Initial Data Loading
```python
# Load the dataset for Scenario 1
driver_df = pd.read_csv('dataset_2.csv')
```

#### Question 1: How much would the total bonus payout be with Option 1?
**Approach:** Filter the DataFrame to find all drivers who meet all four criteria for Option 1. The total payout is the number of qualifying drivers multiplied by $50.

*   **Criteria:**
    *   Hours on duty >= 8
    *   Acceptance rate >= 90% (0.90)
    *   Completed trips >= 10
    *   Average rating >= 4.7

```python
# Define the conditions for Option 1
option1_conditions = (
    (driver_df['hours_on_duty'] >= 8) &
    (driver_df['acceptance_rate'] >= 0.90) &
    (driver_df['trips_completed'] >= 10) &
    (driver_df['average_rating'] >= 4.7)
)

# Filter the DataFrame to get qualifying drivers
drivers_option1 = driver_df[option1_conditions]

# Calculate the total bonus payout
num_drivers_option1 = len(drivers_option1)
bonus_payout_option1 = num_drivers_option1 * 50

print(f"Number of drivers qualifying for Option 1: {num_drivers_option1}")
print(f"Total bonus payout with Option 1 would be: ${bonus_payout_option1:,.2f}")
```

#### Question 2: How much would the total bonus payout be with Option 2?
**Approach:** Filter the DataFrame for drivers meeting the two criteria for Option 2. The total payout is the sum of all completed trips by these qualifying drivers, multiplied by $4.

*   **Criteria:**
    *   Completed trips >= 12
    *   Average rating >= 4.7

```python
# Define the conditions for Option 2
option2_conditions = (
    (driver_df['trips_completed'] >= 12) &
    (driver_df['average_rating'] >= 4.7)
)

# Filter the DataFrame to get qualifying drivers
drivers_option2 = driver_df[option2_conditions]

# Calculate the total bonus payout
total_trips_option2 = drivers_option2['trips_completed'].sum()
bonus_payout_option2 = total_trips_option2 * 4

print(f"Number of drivers qualifying for Option 2: {len(drivers_option2)}")
print(f"Total trips by these drivers: {total_trips_option2}")
print(f"Total bonus payout with Option 2 would be: ${bonus_payout_option2:,.2f}")
```

#### Question 3: How many drivers would qualify for a bonus under Option 1 but not under Option 2?
**Approach:** We need to find the drivers who are in the "Option 1 qualified" set but not in the "Option 2 qualified" set. We can do this by using the driver IDs (or indices) of each group and finding the difference.

```python
# Get the IDs of drivers who qualify for each option
ids_option1 = set(drivers_option1['driver_id'])
ids_option2 = set(drivers_option2['driver_id'])

# Find the IDs in set 1 that are not in set 2
ids_only_in_option1 = ids_option1.difference(ids_option2)

# The number of such drivers is the length of the resulting set
num_drivers_only_option1 = len(ids_only_in_option1)

print(f"Number of drivers who would qualify for Option 1 but not for Option 2: {num_drivers_only_option1}")
```

#### Question 4: What percentages of drivers online completed less than 10 trips, had an acceptance rate of less than 90%, and had a rating of 4.7 or higher?
**Approach:** Filter the DataFrame based on the three specified conditions. Then, divide the number of drivers who meet these criteria by the total number of drivers in the dataset and multiply by 100 to get the percentage.

```python
# Define the conditions for the specified segment
segment_conditions = (
    (driver_df['trips_completed'] < 10) &
    (driver_df['acceptance_rate'] < 0.90) &
    (driver_df['average_rating'] >= 4.7)
)

# Count the number of drivers in this segment
num_drivers_in_segment = len(driver_df[segment_conditions])

# Get the total number of drivers
total_drivers = len(driver_df)

# Calculate the percentage
percentage_in_segment = (num_drivers_in_segment / total_drivers) * 100

print(f"Number of drivers in the specified segment: {num_drivers_in_segment}")
print(f"Percentage of drivers in this segment: {percentage_in_segment:.2f}%")
```
<hr>

### 3. Scenario 2: Taxi Driver Profitability Analysis

This scenario does not require a dataset. The answers are derived from the information provided in the problem description.

#### Question 5: How much money (after expenses) does the taxi driver make per year without partnering with Uber?
**Approach:** We will calculate the driver's total annual income and subtract total annual expenses. We must be careful with time units (daily, weekly, monthly).

*   **Working weeks per year:** 52 total weeks - 3 weeks off = **49 weeks**
*   **Working days per year:** 49 weeks * 6 days/week = **294 days**

**Annual Income:**
*   `294 days/year * $200/day = $58,800`

**Annual Expenses:**
1.  **Gas:** Paid only during working weeks.
    *   `$200/week * 49 weeks = $9,800`
2.  **Insurance:** Paid every month.
    *   `$400/month * 12 months = $4,800`
3.  **Vehicle Rent:** Paid only during working weeks.
    *   `$500/week * 49 weeks = $24,500`

*   **Total Annual Expenses:** `$9,800 + $4,800 + $24,500 = $39,100`

**Annual Profit (Net Income):**
*   `$58,800 (Income) - $39,100 (Expenses) = $19,700`

**Answer:** The taxi driver makes **$19,700** per year after expenses.

#### Question 6: How much would the driver's gross fares need to increase per week to fully pay for the car in year 1 and maintain the same yearly profit margin as before?
**Approach:** This is a multi-step calculation.
1.  Calculate the original profit margin.
2.  Calculate the new annual expenses for the Uber driver in Year 1.
3.  Use the target profit margin and new expenses to find the required new annual fares.
4.  Convert the new annual fares to new weekly fares and find the increase.

**Step 1: Calculate Original Profit Margin**
*   Profit Margin = (Annual Profit / Annual Fares)
*   Profit Margin = `$19,700 / $58,800 ≈ 0.335034` or **33.50%**

**Step 2: Calculate New Annual Expenses (as Uber driver in Year 1)**
*   The driver still works 49 weeks.
1.  **Gas:** Increases by 5%.
    *   `($200 * 1.05) * 49 weeks = $210/week * 49 weeks = $10,290`
2.  **Insurance:** Decreases by 20%.
    *   `($400 * 0.80) * 12 months = $320/month * 12 months = $3,840`
3.  **Car Cost (Year 1 only):**
    *   `$40,000`

*   **Total New Annual Expenses (Year 1):** `$10,290 + $3,840 + $40,000 = $54,130`

**Step 3: Calculate Required New Annual Fares**
*   We need to find the `New Annual Fares` (let's call it `NAF`) such that the profit margin remains the same.
*   `Profit Margin = (NAF - New Annual Expenses) / NAF`
*   `0.335034 = (NAF - $54,130) / NAF`
*   `0.335034 * NAF = NAF - $54,130`
*   `$54,130 = NAF - (0.335034 * NAF)`
*   `$54,130 = NAF * (1 - 0.335034)`
*   `$54,130 = NAF * 0.664966`
*   `NAF = $54,130 / 0.664966 ≈ $81,402.65`

**Step 4: Calculate the Required Weekly Fare Increase**
*   **Required New Weekly Fare:**
    *   `$81,402.65 / 49 weeks ≈ $1,661.28/week`
*   **Original Weekly Fare:**
    *   `$200/day * 6 days/week = $1,200/week`
*   **Required Increase per Week:**
    *   `$1,661.28 - $1,200.00 = $461.28`

**Answer:** To pay for the car in Year 1 and maintain the same profit margin, the driver's gross fares would need to increase by approximately **$461.28 per week**.