# Databricks notebook source
import mlflow
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Bucketizer, StandardScaler, VectorAssembler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.autolog(disable=True)
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Development Pipeline with Sklearn

# COMMAND ----------

fe = FeatureEngineeringClient()
credit_risk_df = fe.read_table(
  name="mlops_dev.credit_risk.credit_risk_features"
).toPandas()
display(credit_risk_df)

# COMMAND ----------

label_encoder = LabelEncoder()
credit_risk_df["Risk_enc"] = label_encoder.fit_transform(credit_risk_df["Risk"])
display(credit_risk_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train-Val-Test Split

# COMMAND ----------

features = ['Age_bucket', 'Credit_amount', 'Duration', 'Job', 'Sex_enc', 'Housing_enc', 'Saving_accounts_enc', 'Checking_account_enc', 'Purpose']
target = "Risk_enc"

X = credit_risk_df[features]
y = credit_risk_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression Pipeline

# COMMAND ----------

numerical_features_to_preprocess = ["Credit_amount", "Duration"]
categorical_features_to_preprocess = ["Purpose"]

numerical_transformer = Pipeline(
  steps=[
    ("scaler", StandardScaler())
  ]
)
categorical_transformer = Pipeline(
  steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
  ]
)
preprocessor = ColumnTransformer(
  transformers=[
    ("numerical", numerical_transformer, numerical_features_to_preprocess),
    ("categorical", categorical_transformer, categorical_features_to_preprocess)
  ]
)
logistic_regression_pipeline = Pipeline(
  steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
  ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter Tuning

# COMMAND ----------


