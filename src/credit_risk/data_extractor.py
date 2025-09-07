import time
import numpy as np
import pandas as pd
import pyspark.sql
import pyspark.sql.functions as F
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


    def ordinal_encode_cols(
        self, 
        df: pyspark.sql.DataFrame, 
        inputCols: list[str], 
        ordinal_dicts: dict[str, str],
        outputCols: list[str]
    ) -> pyspark.sql.DataFrame:
        """Ordinal encode categorical columns based on provided mapping dictionaries.

        :param df: Input Spark DataFrame.
        :param inputCols: List of input column names to be ordinal encoded.
        :param ordinal_dicts: List of dictionaries mapping original categorical values to ordered numerical values.
        :param outputCols: List of output column names for the ordinal encoded columns.
        :return df: Spark DataFrame with ordinal encoded columns.
        """
        for i in range(len(inputCols)):
            df = (
                df
                .withColumn(outputCols[i], F.col(inputCols[i])) # Create duplicate of original col.
                .replace(to_replace=ordinal_dicts[i], subset=[outputCols[i]])   # Map to ordered values.
                .withColumn(outputCols[i], F.col(outputCols[i]).cast("integer")) # Cast to numerical.
            )
        return df


    def initial_feature_preprocessing(self, spark_df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Initial generation of features using custom logic before writing to Unity Catalog Feature Table.

        Other standard feature engineering steps that may be required, i.e. one-hot-encoding, scaling, etc. should be done as part of model pipeline.
        
        :param spark_df: Spark DataFrame to preprocess.
        :return spark_df: Preprocessed Spark DataFrame.
        """
        # Convert string values "NA" to nulls in Saving accounts and Checking account columns + combine "quite rich" and "rich" categories in Saving accounts column to reduce cardinality
        spark_df = spark_df.withColumn("Saving_accounts", F.when(F.col("Saving_accounts") == "NA", F.lit(None)).otherwise(F.col("Saving_accounts"))) \
            .withColumn("Saving_accounts", F.when(F.col("Saving_accounts") == "quite rich", "rich").otherwise(F.col("Saving_accounts"))) \
            .withColumn("Checking_account", F.when(F.col("Checking_account") == "NA", F.lit(None)).otherwise(F.col("Checking_account")))
        
        # Fill nulls in Saving accounts and Checking account columns with "no account/unknown" category
        spark_df = spark_df.fillna("no account/unknown", subset=["Saving_accounts", "Checking_account"])

        # Reduce cardinality of Housing and Purpose categories
        spark_df = spark_df.withColumn("Housing", F.when((F.col("Housing") == "free") | (F.col("Housing") == "rent"), "non-own").otherwise(F.col("Housing"))) \
            .withColumn("Purpose", F.when((F.col("Purpose") == "domestic appliances") | (F.col("Purpose") == "repairs") | (F.col("Purpose") == "vacation/others"), "other").otherwise(F.col("Purpose")))
        
        # Ordinal encoding for Saving accounts and Checking account columns
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
        inputCols = ["Saving_accounts", "Checking_account"]
        outputCols = ["Saving_accounts_ordinal_enc", "Checking_account_ordinal_enc"]
        spark_df = self.ordinal_encode_cols(spark_df, inputCols, ordinal_dicts, outputCols)

        # Label encoding for target column, good = 0, bad = 1
        target_dict = {
            "good": "0", 
            "bad": "1"
        }
        spark_df = spark_df.replace(to_replace=target_dict, subset=["Risk"])
        spark_df = spark_df.withColumn("Risk", F.col("Risk").cast("integer"))

        return spark_df


    def extract_to_feature_table(self) -> None:
        """Extract data from the source dataset and write to Unity Catalog Feature Table."""
        # Convert to Spark DataFrame and perform initial feature preprocessing
        self.preprocess_column_names()
        spark_df = self.spark.createDataFrame(self.pd_df)
        spark_df = self.initial_feature_preprocessing(spark_df)

        # Write to Unity Catalog Feature Table using Databricks, create the table if it doesn't exist, otherwise replace existing table
        table_name = f"{self.config.catalog_name}.{self.config.schema_name}.credit_risk_features"
        (
            spark_df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(table_name)
        )

        # Enforce primary key constraint on id column to ensure table is a feature table in unity catalog
        primary_keys = self.config.primary_keys
        for key in primary_keys:
            self.spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {key} SET NOT NULL")
        self.spark.sql(f"ALTER TABLE {table_name} ADD CONSTRAINT pk_{table_name.split('.')[-1]} PRIMARY KEY ({', '.join(primary_keys)})")