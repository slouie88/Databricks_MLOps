import sys
from pathlib import Path
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch

# Add the src directory to the Python path if not already added
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import project modules after path setup
from credit_risk.data_extractor import DataExtractor
from credit_risk.config import Config


@pytest.fixture
def mock_spark():
    """Create a mock SparkSession."""
    return MagicMock()


@pytest.fixture
def mock_config():
    """Create a mock Config object."""
    config = Mock(spec=Config)
    config.catalog_name = "test_catalog"
    config.schema_name = "test_schema"
    config.primary_keys = ["id"]
    return config


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "Saving-accounts": ["rich", "little", "NA"],
        "Checking account": ["rich", "NA", "moderate"],
        "Housing": ["own", "free", "rent"],
        "Purpose": ["car", "domestic appliances", "repairs"],
        "Risk": ["good", "bad", "good"]
    })


@pytest.fixture
def data_extractor(sample_dataframe, mock_config, mock_spark):
    """Create a DataExtractor instance for testing."""
    return DataExtractor(sample_dataframe.copy(), mock_config, mock_spark)


def test_preprocess_column_names(data_extractor):
    """Test that column names are preprocessed correctly."""
    data_extractor.preprocess_column_names()
    expected_columns = ["id", "Saving_accounts", "Checking_account", "Housing", "Purpose", "Risk"]
    assert list(data_extractor.pd_df.columns) == expected_columns


def test_preprocess_column_names_removes_special_chars(data_extractor):
    """Test that special characters are removed from column names."""
    data_extractor.pd_df.columns = ["col@#$", "col!!!"]
    data_extractor.preprocess_column_names()
    assert "col" in data_extractor.pd_df.columns[0]
    assert "_" in data_extractor.pd_df.columns[0]


def test_ordinal_encode_cols(data_extractor, mock_spark):
    """Test ordinal encoding of categorical columns."""
    spark_df = MagicMock()
    spark_df.withColumn.return_value = spark_df
    spark_df.replace.return_value = spark_df
    
    ordinal_dicts = [{"rich": "1", "little": "0"}]
    result = data_extractor.ordinal_encode_cols(
        spark_df, 
        ["Saving_accounts"], 
        ordinal_dicts, 
        ["Saving_accounts_encoded"]
    )
    
    assert spark_df.withColumn.called
    assert spark_df.replace.called
    assert result == spark_df


def test_ordinal_encode_cols_multiple_columns(data_extractor):
    """Test ordinal encoding with multiple columns."""
    spark_df = MagicMock()
    spark_df.withColumn.return_value = spark_df
    spark_df.replace.return_value = spark_df
    
    ordinal_dicts = [
        {"rich": "2", "moderate": "1", "little": "0"},
        {"rich": "2", "moderate": "1", "little": "0"}
    ]
    
    result = data_extractor.ordinal_encode_cols(
        spark_df,
        ["Saving_accounts", "Checking_account"],
        ordinal_dicts,
        ["Saving_encoded", "Checking_encoded"]
    )
    
    assert result == spark_df
    assert spark_df.withColumn.call_count >= 2


def test_initial_feature_preprocessing(data_extractor, mock_spark):
    """Test initial feature preprocessing."""
    spark_df = MagicMock()
    spark_df.withColumn.return_value = spark_df
    spark_df.fillna.return_value = spark_df
    spark_df.replace.return_value = spark_df
    
    result = data_extractor.initial_feature_preprocessing(spark_df)
    
    assert spark_df.withColumn.called
    assert spark_df.fillna.called
    assert spark_df.replace.called
    assert result == spark_df


def test_initial_feature_preprocessing_calls_ordinal_encode(data_extractor, mock_spark):
    """Test that ordinal encoding is called during preprocessing."""
    spark_df = MagicMock()
    spark_df.withColumn.return_value = spark_df
    spark_df.fillna.return_value = spark_df
    spark_df.replace.return_value = spark_df
    
    with patch.object(data_extractor, 'ordinal_encode_cols', return_value=spark_df) as mock_encode:
        data_extractor.initial_feature_preprocessing(spark_df)
        assert mock_encode.called


def test_extract_to_feature_table(data_extractor, mock_spark):
    """Test extract_to_feature_table method."""
    spark_df = MagicMock()
    spark_df.write.format.return_value.mode.return_value.option.return_value = MagicMock()
    
    mock_spark.createDataFrame.return_value = spark_df
    
    with patch.object(data_extractor, 'initial_feature_preprocessing', return_value=spark_df):
        data_extractor.extract_to_feature_table()
    
    mock_spark.createDataFrame.assert_called_once()
    mock_spark.sql.assert_called()


def test_extract_to_feature_table_calls_preprocess(data_extractor, mock_spark):
    """Test that preprocess_column_names is called in extract_to_feature_table."""
    spark_df = MagicMock()
    mock_spark.createDataFrame.return_value = spark_df
    spark_df.write.format.return_value.mode.return_value.option.return_value = MagicMock()
    
    with patch.object(data_extractor, 'preprocess_column_names') as mock_preprocess:
        with patch.object(data_extractor, 'initial_feature_preprocessing', return_value=spark_df):
            data_extractor.extract_to_feature_table()
        mock_preprocess.assert_called_once()