---
title: Fraudsters Detection
company: Revolut
difficulty: Medium
category: Classification
date: 2025-07-28
---
_This data project has been used as a take-home assignment in the recruitment process for the data science positions at Revolut._

## Assignment

Explore the data for identified fraudsters and other users. What are your preliminary observations? Utilizing your findings from and some creativity, create some features. Explain your reasoning behind the features. Create an ML model which identifies fraudsters. Assess the quality of your model and explain.

## Data Description

The file `users.csv` contains data on a subset of fictional banking users. The abbreviation 'KYC' stands for 'Know Your Customer' - a process of identifying and verifying the client's identity when opening an account and periodically over time. The variable `IS_FRAUDSTER` from this dataset is your target variable. The file `transactions.csv` contains details of fictional transactions of these users.

The files `countries.csv` and `currency_details.csv` are dictionaries that provide explanations of abbreviations used in columns `COUNTRY` of the users dataset and `CURRENCY` of the transactions dataset respectively. These dictionaries may be useful but you don't need to use them when solving this task.

## Practicalities

Define, train and evaluate predictive models that take as the input the data provided. You may want to split the data into training, testing and validation sets, according to your discretion. Do not use external data for this project. You may use any algorithm of your choice or compare multiple models.

Make sure that the solution reflects your entire thought process - it is more important how the code is structured rather than the final metrics. You are expected to spend no more than 3 hours working on this project.
