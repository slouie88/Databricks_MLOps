# Databricks notebook source
# MAGIC %md
# MAGIC ### DISCLAIMER
# MAGIC These notebooks were written for quick experimentation in Databricks. These Databricks Notebooks should be run in a Databricks workspace, corresponding dir path and file name variables may need to be changed according to your own needs.

# COMMAND ----------

# MAGIC %pip install -e ..
# MAGIC %restart_python

# COMMAND ----------

from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Packages

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Bucketizer, StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
import mlflow
# from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

mlflow.autolog(disable=True)
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initial EDA

# COMMAND ----------

credit_df = pd.read_csv("../data/german_credit_data.csv")
credit_df = spark.createDataFrame(credit_df) \
    .withColumn("Saving accounts", F.when(F.col("Saving accounts") == "NA", F.lit(None)).otherwise(F.col("Saving accounts"))) \
    .withColumn("Saving accounts", F.when(F.col("Saving accounts") == "quite rich", "rich").otherwise(F.col("Saving accounts"))) \
    .withColumn("Checking account", F.when(F.col("Checking account") == "NA", F.lit(None)).otherwise(F.col("Checking account")))

# COMMAND ----------

dbutils.data.summarize(credit_df)

# COMMAND ----------

credit_df = credit_df.fillna("no account/unknown", subset=["Saving accounts", "Checking account"])
credit_df.show(10)

# COMMAND ----------

display(credit_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Categorical Feature Ideas
# MAGIC - Keep Job as is for an ordinal encoded catagorical feature.
# MAGIC - 0/1 encode Sex.
# MAGIC - Ordinal encode Saving accounts and Checking account features based on proportion of "bad" risk (low proportion to high proportion).
# MAGIC - Combine "free" (free housing) and "rent" of Housing feature to be the same category (i.e. non-ownership) and 0/1 encode as a feature. "free" and "rent" have very similar count proportion of "bad" risk, and on their own make a very small propertion of the data.
# MAGIC - Combine "domestic appliances", "repairs", and "vacation/others" as a single "other" Purpose category, and then one-hot encode.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering Experimentation

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Reduce cardinality of Housing and Purpose categorical variables

# COMMAND ----------

credit_df = credit_df \
    .withColumn("Housing", F.when((F.col("Housing") == "free") | (F.col("Housing") == "rent"), "non-own") \
        .otherwise(F.col("Housing"))) \
    .withColumn("Purpose", F.when((F.col("Purpose") == "domestic appliances") | (F.col("Purpose") == "repairs") | (F.col("Purpose") == "vacation/others"), "other") \
        .otherwise(F.col("Purpose")))
credit_df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Ordinal encoding for Saving accounts and Checking account columns

# COMMAND ----------

def ordinal_encode_cols(df, inputCols, ordinal_dicts, outputCols):
    for i in range(len(inputCols)):
        df = (
            df
            .withColumn(outputCols[i], F.col(inputCols[i])) # Create duplicate of original col.
            .replace(to_replace=ordinal_dicts[i], subset=[outputCols[i]])   # Map to ordered values.
            .withColumn(outputCols[i], F.col(outputCols[i]).cast("double")) # Cast to numerical.
        )
    return df


# COMMAND ----------

ordered_saving_accounts = {
    "rich": "3",
    "no account/unknown": "2",
    "moderate": "1",
    "little": "0"
}
ordered_checking_accounts = {
    "no account/unknown": "3",
    "rich": "2",
    "moderate": "1",
    "little": "0"
}
ordinal_dicts = [ordered_saving_accounts, ordered_checking_accounts]
inputCols = ["Saving accounts", "Checking account"]
outputCols = ["Saving accounts_ordinal_enc", "Checking account_ordinal_enc"]

credit_df = ordinal_encode_cols(
    credit_df, 
    inputCols=inputCols, 
    ordinal_dicts=ordinal_dicts, 
    outputCols=outputCols
)
credit_df.show(10)
