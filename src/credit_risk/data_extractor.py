import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from credit_risk.config import Config



class DataExtractor:
    """Class for extracting data from source datasets in the `data` directory and writing them to Unity Catalog tables."""

    def __init__(
        self,
        config: Config,
        spark: SparkSession,
    ) -> None:
        self.config = config
        self.spark = spark


    def extract_to_feature_table(self, pd_df: pd.DataFrame) -> None:
        """Extract data from the source dataset and write to Unity Catalog Feature Table.

        :param pd_df: Input pandas DataFrame to be written to feature table
        """
        # Convert pandas DataFrame to Spark DataFrame
        spark_df = self.spark.createDataFrame(pd_df)

        # Write to Unity Catalog Feature Table using Databricks 
        table_name = f"{self.config.catalog_name}.{self.config.schema_name}.credit_risk"
        spark_df.writeTo(table_name).createOrReplace()  # Create the table if it doesn't exist. Otherwise, replace existing table.

        # Enforce primary key constraint on id column to ensure table is a feature table in unity catalog
        primary_keys = self.config.primary_keys
        for key in primary_keys:
            self.spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {key} SET NOT NULL")
        self.spark.sql(f"ALTER TABLE {table_name} ADD CONSTRAINT pk_{table_name.split('.')[-1]} PRIMARY KEY ({', '.join(primary_keys)})")