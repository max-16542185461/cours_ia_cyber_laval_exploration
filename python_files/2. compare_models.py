# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (pixi)
#     language: python
#     name: cours_ia_cyber_laval_exploration
# ---

from sklearn.metrics import classification_report
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

# %% [markdown]
# # Compare machine learning models
#
# In this notebook, we will compare 3 pre-trained models that predict the
# **Census Region** of a respondent based on their survey answers.
#
# The 3 models are:
# - **Logistic Regression**: a simple linear model
# - **Random Forest**: a model based on many decision trees
# - **Gradient Boosting**: a model that builds trees sequentially

# %% [markdown]
# ## Load the dataset

# %%
from skrub.datasets import fetch_midwest_survey

dataset = fetch_midwest_survey()
X = dataset.X
y = dataset.y

# %%
# To simplify evaluation, we will group categories in the target to deal with a binary classification problem instead of a multiclass one.
y = y.apply(lambda x: "North Central" if x in ["East North Central", "West North Central"] else "other")

# %%
sample_idx = X.sample(n=1000, random_state=1).index
X_train = X.loc[sample_idx].reset_index(drop=True)
y_train = y.loc[sample_idx].reset_index(drop=True)
X_test = X.drop(sample_idx).reset_index(drop=True)
y_test = y.drop(sample_idx).reset_index(drop=True)

# %% [markdown]
# ## Load the 3 models
#
# The models were saved as `.pkl` files. We use `joblib` to load them.

# %%
import joblib
from midwest_survey_models.transformers import NumericalStabilizer

model_lr = joblib.load("../model_logistic_regression.pkl")
model_rf = joblib.load("../model_random_forest.pkl")
model_gb = joblib.load("../model_gradient_boosting.pkl")

# %% [markdown]
# Let's inspect what each model looks like. They are **pipelines**: they
# first transform the data, then make predictions.

# %%
model_lr

# %%
model_rf

# %%
model_gb

# %% [markdown]
# ## Evaluate the models with cross-validation
#
# To fairly evaluate each model, we use **cross-validation**.
# This means we train and test the model on different parts of the data multiple times, so we can see how well it generalizes.
#
# We use `cross_val_score` to get the score for every fold in cross-validation.

# %%
from sklearn.model_selection import cross_val_score

cv_lr = cross_val_score(model_lr, X, y, cv=5)
cv_rf = cross_val_score(model_rf, X, y, cv=5)
cv_gb = cross_val_score(model_gb, X, y, cv=5)

# %% [markdown]
# ## Question 6: Among the three models, which one has the best recall?
#
# The **classification report** shows precision, recall, and f1-score for each class.
#
# - **Precision**: among all predictions for a class, how many were correct?
# - **Recall**: among all real examples of a class, how many were found?
# - **F1-score**: a balance between precision and recall
#
# We will define the positive class as "North Central".

# %%
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_gb = model_gb.predict(X_test)
# %%
from skore import EstimatorReport
report_lr = EstimatorReport(estimator = model_lr,
                X_test = X_test,
                y_test = y_test)
report_rf = EstimatorReport(estimator = model_rf,
                X_test = X_test,
                y_test = y_test)
report_gb = EstimatorReport(estimator = model_gb,
                X_test = X_test,
                y_test = y_test)
# %%

# %% [markdown]
# Which model has the highest recall?
recall_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
    "Recall (North Central)": [
        float(classification_report(y_test, y_pred_lr, output_dict=True)["North Central"]["recall"]),
        float(classification_report(y_test, y_pred_rf, output_dict=True)["North Central"]["recall"]),
        float(classification_report(y_test, y_pred_gb, output_dict=True)["North Central"]["recall"])
    ]
})
print(recall_comparison)
print("Highest recall:", recall_comparison.loc[recall_comparison["Recall (North Central)"].idxmax()])
# %% [markdown]
# ## Question 7: Which model has the best practical application?
#
# Let's say that it costs 10 to make a false positive error, while it costs 1 to make a false negative error. Correctly predicting a positive example gains 5, while correctly predicting a negative example gains 2.

# %%

# %% [markdown]
# Which model makes the most meaningful predictions in practice?
cm_lr = confusion_matrix(y_test, y_pred_lr, labels=["other", "North Central"])
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=["other", "North Central"])
cm_gb = confusion_matrix(y_test, y_pred_gb, labels=["other", "North Central"])

def calculate_profit(cm):
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    profit = (tp * 5) + (tn * 2) - (fp * 10) - (fn * 1)
    return profit

profit_lr = calculate_profit(cm_lr)
profit_rf = calculate_profit(cm_rf)
profit_gb = calculate_profit(cm_gb)

profit_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
    "Profit": [profit_lr, profit_rf, profit_gb]
})
print("Best practical model:", profit_comparison.loc[profit_comparison["Profit"].idxmax()])
# %% [markdown]
# ## Question 8: Which model generalizes the best?
#
# To understand generalization, we compare the **training score** (how well the model fits the data it was trained on) with the **test score** (how well it performs on unseen data).
#
# A big gap between the two means the model is **overfitting**.  
#
# We don't want to do this only once, but several times. Use cross-validation for that. You can either use cross-validation from scikit-learn, or the CrossValidationReport from skore.

# %%

# %% [markdown]
# Which model has the smallest gap between train and test accuracy?
# That model generalizes the best.
#
# Which model has the largest gap? That model is likely **overfitting**.

cv_results_lr = cross_validate(model_lr, X, y, cv=5, return_train_score=True)
cv_results_rf = cross_validate(model_rf, X, y, cv=5, return_train_score=True)
cv_results_gb = cross_validate(model_gb, X, y, cv=5, return_train_score=True)

generalization_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
    "Train Score": [
        cv_results_lr["train_score"].mean(),
        cv_results_rf["train_score"].mean(),
        cv_results_gb["train_score"].mean()
    ],
    "Test Score": [
        cv_results_lr["test_score"].mean(),
        cv_results_rf["test_score"].mean(),
        cv_results_gb["test_score"].mean()
    ]
})

generalization_comparison["Gap"] = generalization_comparison["Train Score"] - generalization_comparison["Test Score"]

print("\nBest generalization (smallest gap):", generalization_comparison.loc[generalization_comparison["Gap"].idxmin()])
print("Most overfitting (largest gap):", generalization_comparison.loc[generalization_comparison["Gap"].idxmax()])
# %%
# TODO: Based on the results above, which model would you choose
# for a real application? Write your answer as a comment below.

# My choice: ...
# Reason: ...

