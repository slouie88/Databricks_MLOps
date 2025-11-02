# Databricks notebook source
# MAGIC %pip install ../dist/credit_risk-0.1.0-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import time
import os
import requests
from pyspark.sql import SparkSession
from mlflow import mlflow
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from credit_risk.config import Config
from credit_risk.endpoint_deployments.model_serving import ModelServing
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuration

# COMMAND ----------
# spark session
spark = SparkSession.builder.getOrCreate()

w = WorkspaceClient()
os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1800).token_value


def is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


# Load project config
config_path = "../project_config_credit_risk.yml"
config = Config.from_yaml(config_path=config_path, env="dev")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Serving Endpoint

# COMMAND ----------
# Initialize model serving
model_serving = ModelServing(
    model_name=f"{config.catalog_name}.{config.schema_name}.pyfunc_credit_risk_model", 
    endpoint_name="credit-risk-model-serving-dev"
)

# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Model Serving Endpoint

# COMMAND ----------
# Create a sample request body
required_columns = [
    "Credit_amount",
    "Duration",
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Purpose",
    "Saving_accounts_ordinal_enc",
    "Checking_account_ordinal_enc"
]

# Sample some records from feature table
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.credit_risk_features").toPandas()

# Sample records from the training set
sample_records = test_set[required_columns].sample(n=100, replace=True)

# Replace NaN values with None, which will be serialized as null in JSON
sample_records = sample_records.replace({np.nan: None}).to_dict(orient="records")
df_records = [[record] for record in sample_records]

# COMMAND ----------
# Call the endpoint with one sample record
serving_endpoint = f"{os.environ['DBR_HOST']}/serving-endpoints/credit-risk-model-serving-dev/invocations"    
print(f"Calling endpoint: {serving_endpoint}")

response = requests.post(
    serving_endpoint,
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={"dataframe_records": df_records[0]},
)
print(f"Response Status: {response.status_code}")
print(f"Response Text: {response.text}")

# COMMAND ----------
# Call the endpoint with all sampled records
for i in range(len(df_records)):
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": df_records[i]},
    )
    print(f"Response Status: {response.status_code}")
    print(f"Response Text: {response.text}")
    time.sleep(0.2) 

# COMMAND ----------
