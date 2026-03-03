# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt

# %% [markdown]
# # Explore the Midwest Survey dataset
#
# In this notebook, we will explore the **Midwest Survey** dataset from [skrub](https://skrub-data.org/).
#
# This dataset contains survey responses from people across the United States,
# asking them about their perception of the Midwest region.
#
# The goal is to predict the **Census Region** where a respondent lives,
# based on their survey answers.

# %% [markdown]
# ## Load the dataset

# %%
from skrub.datasets import fetch_midwest_survey

dataset = fetch_midwest_survey()

# X contains the features (the survey answers)
X = dataset.X
# y contains the target (the Census Region)
y = dataset.y
# %% [markdown]
# ## Question 1: How many examples are there in the dataset?
#
# Use the `.shape` attribute to find out the number of rows and columns.

# %%
# Display the number of rows and columns
# [5 rows x 28 columns]

# %%
# You can also look at the first few rows of the dataset
print("X HEAD : ", X.head())

# %% [markdown]
# ## Question 2: What is the distribution of the target?
#
# The target variable `y` tells us the Census Region of each respondent.
# Let's see how many respondents belong to each region.

# %%
# Count how many respondents belong to each region

# https://stackoverflow.com/questions/34076177/matplotlib-horizontal-bar-chart-barh-is-upside-down
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html
counts = y.value_counts()
print("Counts : ", counts)

counts.plot.barh()
plt.xlabel("target")
plt.title("Question 2: What is the distribution of the target?")
plt.tight_layout()
plt.show()

# %%
# Visualize the target distribution with a bar plot
# hint: use barh


# %% [markdown]
# Is the target balanced (roughly the same number of examples per class) or imbalanced?

# %% [markdown]
# ## Question 3: What are the features that can be used to predict the target?
#
# Let's look at the column names and their data types.

# %%
# List all column names
print("Colonnes :", list(X.columns))

# %%
# Show data types for each column
print("Types de données :", X.dtypes)
# %% [markdown]
# How many features are numerical? How many are categorical (text)?

# %%

# %%
from skrub import TableReport
TableReport(X)

# %% [markdown]
# ## Question 4: Are there any missing values in the dataset?
#
# Missing values can cause problems for machine learning models.
# Let's check if there are any.

# %%
# Check for NaN missing values
missing = X.isna().sum()

# %% [markdown]
# Missing values can sometimes be encoded differently. Let's look at some columns more closely.
print("Missing values : ", missing)

# %%
# Look at unique values for the Household_Income column
# #X["Household_Income"].??
print("Colonnes NaN : ", missing["Household_Income"])
# %%
# Look at unique values for the Education column

# %% [markdown]
# Do you see a special value that could represent missing data?

# %% [markdown]
# ## Question 5: What is the most common answer to "How much do you personally identify as a Midwesterner"?
#
# Let's explore this important feature.

# %%
# TODO: display the value counts for the column
# "How_much_do_you_personally_identify_as_a_Midwesterner"
counts_midwesterner = X["How_much_do_you_personally_identify_as_a_Midwesterner"].value_counts()
print("Counts Midwesterner : ", counts_midwesterner)

# %%
# TODO: make a bar plot of the results
counts_midwesterner.plot.barh()
plt.xlabel("target")
plt.title("Question 5: What is the most common answer to 'How much do you personally identify as a Midwesterner'?")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Bonus: Explore another feature
#
# Pick another column and explore its distribution.
# For example: `Gender`, `Age`, or one of the
# "Do you consider X state as part of the Midwest" columns.

# %%
# TODO: explore a column of your choice

