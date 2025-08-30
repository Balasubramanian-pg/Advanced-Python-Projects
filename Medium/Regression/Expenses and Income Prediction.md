---
title: Expenses and Income Prediction
company: N26
difficulty: Medium
category: Regression
date: 2025-07-28
---
_This data project has been used as a take-home assignment in the recruitment process for the data science positions at N26._

## Assignment

Given the data set of transactions of 10k users, define regression models for predicting what are the total expenses and the income of a user.

## Data Description

The file `transactions_data_training.csv` contains records of transactions. The other two files, `transaction_types.csv` and `mcc_group_definition.csv` are dictionaries providing explanations about values in `transaction_type` and `mcc_group` columns from the main dataset respectively.

Each transaction, depending on its type, may represent money flowing either into or from a user's account. Before defining the model, you should therefore extract the target variables: the total expenses and total income of each user based on the combination of `amount_n26_currency` and `transaction_type` columns.

## Practicalities

Define, train and evaluate predictive models that take as the input the data provided. You may want to split the data into training, testing and validation sets, according to your discretion. Do not use external data for this project. You may use any algorithm of your choice or compare multiple models.

Make sure that the solution reflects your entire thought process - it is more important how the code is structured rather than the final metrics. You are expected to spend no more than 3 hours working on this project.