import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from credit_risk.config import Config


class DataExtractor:
    """Class for extracting data from source datasets in the `data` directory and writing them to Unity Catalog tables."""

    def __init__(
        self,
        pd_df: pd.DataFrame,
        config: Config,
        spark: SparkSession,
    ) -> None:
        self.pd_df = pd_df
        self.config = config
        self.spark = spark


    def preprocess_column_names(self) -> None:
        """Preprocess column names to remove special characters and spaces."""
        self.pd_df.columns = self.pd_df.columns.str.replace(r"[^a-zA-Z0-9]", "_", regex=True)
        self.pd_df.columns = self.pd_df.columns.str.replace(r"\s+", "_", regex=True)


    def preprocess_features(self) -> None:
        """Preprocess features in the pandas DataFrame."""
        


    def extract_to_feature_table(self) -> None:
        """Extract data from the source dataset and write to Unity Catalog Feature Table.

        """
        # Preprocess pandas DataFrame column names and convert to Spark DataFrame
        self.preprocess_column_names()
        spark_df = self.spark.createDataFrame(self.pd_df)

        # Write to Unity Catalog Feature Table using Databricks
        # Create the table if it doesn't exist, otherwise, replace existing table
        table_name = f"{self.config.catalog_name}.{self.config.schema_name}.credit_risk_features"
        (
            spark_df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(table_name)
        )
        # spark_df.writeTo(table_name).createOrReplace()

        # Enforce primary key constraint on id column to ensure table is a feature table in unity catalog
        primary_keys = self.config.primary_keys
        for key in primary_keys:
            self.spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {key} SET NOT NULL")
        self.spark.sql(f"ALTER TABLE {table_name} ADD CONSTRAINT pk_{table_name.split('.')[-1]} PRIMARY KEY ({', '.join(primary_keys)})")