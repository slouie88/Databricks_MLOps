# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Bucketizer, StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

mlflow.autolog(disable=True)
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initial EDA

# COMMAND ----------

credit_df = spark.sql(
    """
    SELECT *
    FROM scb_wh_dev.scratch_dev.raw_credit_info;
    """
)\
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
# MAGIC - Bucketize Age into age bracket bins/buckets and treat as a categorical variable since Age is discrete.
# MAGIC - Keep Job as is for an ordinal encoded catagorical feature.
# MAGIC - 0/1 encode Sex.
# MAGIC - Ordinal encode Saving accounts and Checking account features based on proportion of "bad" risk (low proportion to high proportion).
# MAGIC - Combine "free"and "rent" of Housing feature to be the same category (i.e. non-ownership) and 0/1 encode as a feature. "free" and "rent" have very similar count proportion of "bad" risk.
# MAGIC - Combine "domestic appliances", "repairs", and "vacation/others" as a single "other" Purpose category, and then one-hot encode.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering

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

display(credit_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 0/1 encoding for Sex and Housing

# COMMAND ----------

def label_encode_cols(df, inputCols, outputCols):
    string_indexer = StringIndexer(inputCols=inputCols, outputCols=outputCols, handleInvalid="error")
    result = string_indexer.fit(credit_df).transform(credit_df)
    return result

# COMMAND ----------

inputCols = ["Sex", "Housing"]
outputCols = ["Sex_enc", "Housing_enc"]

credit_df = label_encode_cols(credit_df, inputCols=inputCols, outputCols=outputCols)
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
outputCols = ["Saving accounts_enc", "Checking account_enc"]

credit_df = ordinal_encode_cols(
    credit_df, 
    inputCols=inputCols, 
    ordinal_dicts=ordinal_dicts, 
    outputCols=outputCols
)
credit_df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### One-hot encode Purpose column

# COMMAND ----------

def one_hot_encode_cols(df, inputCols, outputCols):
    one_hot_encoder = OneHotEncoder(inputCols=inputCols, outputCols=outputCols, handleInvalid="error", dropLast=True)
    result = one_hot_encoder.fit(df).transform(df)
    return result

# COMMAND ----------

inputCols = ["Purpose"]
outputCols1 = ["Purpose_index"]
outputCols2 = ["Purpose_enc"]

credit_df = label_encode_cols(
    credit_df, 
    inputCols=inputCols, 
    outputCols=outputCols1
)
credit_df = one_hot_encode_cols(
    credit_df, 
    inputCols=outputCols1, 
    outputCols=outputCols2
)
credit_df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Bucketize/discretize Age column

# COMMAND ----------

def bucketize_cols(df, inputCols, outputCols, splitsArray):
    for i in range(len(inputCols)):
        bucketizer = Bucketizer(inputCol=inputCols[i], outputCol=outputCols[i], splits=splitsArray[i])
        df = bucketizer.transform(df)
    return df

# COMMAND ----------

inputCols = ["Age"]
outputCols = ["Age_bucket"]
splitsArray = [[0, 25, 40, 55, 70, float("inf")]]

credit_df = bucketize_cols(
    credit_df, 
    inputCols=inputCols, 
    outputCols=outputCols, 
    splitsArray=splitsArray
)
credit_df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Standardize Credit amount and Duration

# COMMAND ----------

def standard_scale_continuous_cols(df, inputCols):
    vector_assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
    df = vector_assembler.transform(df)

    scaler = StandardScaler(inputCol="features", outputCol="scaled_continuous_features")
    df = scaler.fit(df).transform(df)

    return df

# COMMAND ----------

inputCols = ["Credit amount", "Duration"]

credit_df = standard_scale_continuous_cols(
    credit_df, 
    inputCols=inputCols
)
credit_df.show(10)

# COMMAND ----------

inputCols = ["scaled_continuous_features", "Job", "Age_bucket", "Sex_enc", "Housing_enc", "Saving accounts_enc", "Checking account_enc", "Purpose_enc"]
outputCol = "ml_features"
features = VectorAssembler(inputCols=inputCols, outputCol=outputCol).transform(credit_df)
features.select("ml_features").show(10)

# COMMAND ----------

display(features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Store Table

# COMMAND ----------

features = ["Age_bucket", "Credit amount", "Duration", "Job"] + \
    ["Sex_enc", "Housing_enc", "Saving accounts_enc", "Checking account_enc", "Purpose", "Risk"]
# features = ["Age_bucket", "Credit amount", "Duration", "Job"] 

credit_risk_features = credit_df.select(*features).withColumn("id", F.monotonically_increasing_id())
credit_risk_features = credit_risk_features.select([F.col(col).alias(col.replace(' ', '_')) \
    for col in credit_risk_features.columns])
credit_risk_features.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### With FeatureEngineering Client

# COMMAND ----------

fe = FeatureEngineeringClient()

fe.create_table(
    name="scratch_dev.credit_risk_features", 
    primary_keys=["id"],
    # df=credit_risk_features, 
    schema=credit_risk_features.schema,
    description="Credit Risk Features"
)

# COMMAND ----------

fe.write_table(
    name="scratch_dev.credit_risk_features", 
    df=credit_risk_features, 
    mode="merge"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### With Databricks SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS scratch_dev.credit_risk_features(
# MAGIC   id INT NOT NULL,
# MAGIC   Age_bucket FLOAT,
# MAGIC   Credit_amount INT,
# MAGIC   Duration INT,
# MAGIC   Job INT,
# MAGIC   Sex_enc FLOAT,
# MAGIC   Housing_enc FLOAT,
# MAGIC   Saving_accounts_enc FLOAT,
# MAGIC   Checking_account_enc FLOAT,
# MAGIC   Purpose STRING,
# MAGIC   Risk STRING,
# MAGIC   CONSTRAINT credit_risk_features_pk PRIMARY KEY (id)
# MAGIC )

# COMMAND ----------

fe.write_table(
    name="scratch_dev.credit_risk_features", 
    df=credit_risk_features, 
    mode="merge"
)
# Or can create a temporary view with the features spark df and write an upsert SQL statement with it (but am too lazy to write).
