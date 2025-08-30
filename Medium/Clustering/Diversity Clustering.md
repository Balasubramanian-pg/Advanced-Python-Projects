---
title: Diversity Clustering
company: Google
difficulty: Medium
category: Clustering
date: 2025-07-28
---
_This data project is used as a take-home assignment in the recruitment process for data science roles at Google._

Google LLC, founded in 1998, is a global leader in search, advertising, and cloud technology. Known for its data-driven culture and ethical AI initiatives, Google openly shares its Diversity Annual Reports to promote transparency and accountability in hiring and inclusion efforts.

## Assignment

As a data scientist at Google, you are asked to explore underlying patterns in the company's workforce demographic data using unsupervised learning techniques. The goal is to uncover clusters of sectors or industries with similar diversity compositions and identify outlier groups that might require further policy attention.

The dataset spans several years of demographic distributions — such as gender and ethnicity — across different sectors and industries.

**The company is interested in answering the following:**

1. Which sectors have similar demographic structures across gender and ethnicity?
2. Which three sectors are most gender-imbalanced and how does that relate to total workforce size?
3. Which three sectors have demonstrated the most notable shifts in workforce diversity between 2021 and 2023?
4. Which sectors are dominated by a single ethnic group and how extreme is this imbalance?
5. Which sectors show the highest ethnic diversity and how does this relate to sector size?
6. Can we group industries based on similar ethnic compositions?

You are encouraged to apply clustering methods such as KMeans or Hierarchical clustering and visual methods like PCA to interpret and explain your findings.

## Data Description

The file `dataset-google.csv` contains publicly available, anonymized summary data extracted from Google’s Diversity Annual Report. The data structure includes:

- `year` – Year the demographic data was recorded
- `sector` – High-level economic category of employment
- `subsector` – Mid-level breakdown within sectors
- `industry_group` – Granular grouping of industries
- `industry` – Specific industry name (if available)
- `total_employed_in_thousands` – Workforce size in thousands
- `percent_women` – Percentage of female employees
- `percent_white` – Percentage of white-identifying employees
- `percent_black_or_african_american` – Percentage of Black employees
- `percent_asian` – Percentage of Asian employees
- `percent_hispanic_or_latino` – Percentage of Hispanic/Latino employees

## Practicalities

Use clustering algorithms and dimensionality reduction methods to uncover insights from the data. Your deliverable should include:

- Clear explanation of data preprocessing
- Visualizations supporting any discovered clusters or anomalies
- Interpretation of any demographic imbalances or notable patterns
- Structured and reproducible code (notebook format)
- Your thought process behind model and feature choices

Clarity and analytical reasoning are more valuable than just code output. You are encouraged to explain trade-offs and design decisions throughout the notebook.