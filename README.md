# Diabetes Prediction with Pima Indians Diabetes Dataset

## Overview

This project involves predicting the onset of diabetes using the Pima Indians Diabetes Dataset. The dataset contains medical data for female patients of Pima Indian heritage, and the goal is to create machine learning models that can accurately predict diabetes based on various diagnostic features.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

The Pima Indians Diabetes Dataset provides a basis for predicting diabetes onset. The dataset includes various features related to diagnostic measurements, and the objective is to build models that can classify whether a patient is diabetic or not.

## Dataset

The dataset used in this project is the **Pima Indians Diabetes Dataset**, available from the UCI Machine Learning Repository. It includes the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history
- **Age**: Age of the person (years)
- **Outcome**: Class variable (0 if non-diabetic, 1 if diabetic)

## Project Overview

This project covers the following aspects:

1. **Exploratory Data Analysis (EDA)**: Understanding the dataset through visualizations and summary statistics.
2. **Data Preprocessing**: Cleaning and preparing the data for modeling.
3. **Model Building**: Training and evaluating various machine learning models.
4. **Results**: Analyzing and comparing model performance.

## Installation

### Prerequisites

Make sure you have Python 3.6 or later installed. You will also need Jupyter Notebook or any other Python IDE.

### Required Libraries

To install the necessary libraries, run the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
## Usage
### Running the Jupyter Notebook
1. **Clone the Repository:**
```bash
git clone https://github.com/krishnavamsimudigonda/Diabeties-Prediction.git
 ```
 2.**Change Directory:**
 ```bash
 cd Diabeties-Prediction
 ```
 3.**Open the Jupyter Notebook:**
 ```bash
 jupyter notebook
 ```
 3.**Run All Cells:**
 - Open the Jupyter Notebook interface.
 - Navigate to diabetes_prediction.ipynb.
 - Run all cells to execute the code and view the results.

 ## Exploratory Data Analysis (EDA)
 EDA involves visualizing and analyzing the dataset to understand its characteristics. Key visualizations include:
- **Correlation Heatmap:** Shows correlations between features.
- **Boxplot:** Compares distributions of BMI for diabetic and non-diabetic patients.
- **Histograms:** Displays distributions of features such as Glucose and Age.
- **Scatter Plot:** Examines relationships between features like Age and Glucose.

### Example Code for Correlation Heatmap
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation Matrix
corr_matrix = df.corr()

# Plot the heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of Pima Indians Diabetes Dataset')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```

## Data Preprocessing
Data preprocessing steps include:

- **Handling Missing Values:** Replacing zero values in certain columns with the median value of that column.
- **Feature Scaling:** Standardizing features using **'StandardScaler'** to improve model performance.
```python
# Replace zero values in certain columns with the median (except for pregnancies)
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for column in columns_to_replace:
    df[column] = df[column].replace(0, df[column].median())
```

## Conclusion

StandardScaler is used help to improve model performance, reduce the impact of outliers, and ensure that the data is on the same scale.Graphs provide visual representation of the data.

## References

- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
