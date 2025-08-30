---
title: Outliers Detection
company: Physician Partners
difficulty: Medium
category: Data Engineering
date: 2025-07-28
---
_This data project has been used as a take-home assignment in the recruitment process for the data science positions at Physician Partners._

## Assignment

In this exercise, we are providing you with an example of financial data (all numbers are fictional).

You need to create an algorithm that can find outliers in this data by one column / several columns. E.g. some members have extremely high costs in the current month and your solution should be able to detect such records. Think about features and how you would explain them to business people.

## Data Description

The dataset `sfr_test.csv` contains fictional financial information about customers of the company.

Below is the description of the selected columns from this dataset:

- `member_unique_id` - member's ID
- `gender` - member's gender
- `dob` - member's date of birth
- `eligible_year` - year
- `eligible_month` - month
- `affiliation_type` - doctor's type
- `pbp_group` - health plan group
- `plan_name` - health plan name
- `npi` - doctor's ID
- `line_of_business` - health plan type
- `esrd` - True if patient is on dialysis
- `hospice` - True if patient is in hospice

The remaining columns contain various financial indicators. **All values in these financial columns contain a $ sign.**

## Practicalities

Please provide a Jupyter notebook describing your approach as a result of this task.

Since it is an open-ended task, your solution will need to include your thought process, including data exploration, outlier detection approaches, results, and interpretation. Your code must be well-structured, and all outputs (numerical and visual) are properly documented and easily readable. Always keep in mind that the final document would be read by someone with a business background, so be sure to clearly explain any technical terms and metrics. You should spend no more than 2-3 hours on this task.