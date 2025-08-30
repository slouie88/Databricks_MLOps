# Databricks notebook source
# MAGIC %md
# MAGIC ### Import Packages

# COMMAND ----------

import yaml
from loguru import logger
from pyspark.sql import SparkSession
import pandas as pd
from credit_risk.config import Config
from credit_risk.data_extractor import DataExtractor

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuration

# COMMAND ----------

project_config_yml_filename = "project_config_credit_risk.yml" 
config_path = f"Databricks_MLOps/{project_config_yml_filename}"
config = Config.from_yaml(config_path=config_path, env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Data to Feature Table

# COMMAND ----------

# Load the source csv ataset
spark = SparkSession.builder.getOrCreate()
csv_data_filename = "german_credit_data.csv"
data_filepath = f"Databricks_MLOps/data/{csv_data_filename}"
pd_df = pd.read_csv(data_filepath)
logger.info("Source csv data loaded for processing.")

# Initialize DataExtractor
data_extractor = DataExtractor(config, spark)

# Preprocess the data
data_extractor.extract_to_feature_table(pd_df)
logger.info("Data extracted to Unity Catalog feature table.")
