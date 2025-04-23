# 🔍 Data Engineering & Testing Guide

This guide provides a beginner-friendly introduction to key data engineering concepts, tools, and testing approaches. Click the links below to access detailed guides on specific topics:

- [Functional Testing for Data](functional_testing_readme.md)
- [Data Quality Checks](data_quality_readme.md)

## 📊 What is Databricks?

**Databricks** is a cloud-based data engineering platform that combines the best of data warehouses and data lakes into a "lakehouse" architecture. It's designed to help teams collaborate on processing and analyzing large amounts of data.

### 🌟 Key Features of Databricks:

1. **🔄 Unified Analytics Platform**: Combines data engineering, data science, and business analytics in one place.

2. **⚡ Built on Apache Spark**: Uses Spark's powerful distributed computing to process massive datasets quickly.

3. **📓 Notebook Interface**: Similar to Jupyter but with more collaboration features and better Spark integration.

4. **🧪 MLflow Integration**: Built-in tracking and management for machine learning experiments.

5. **🔐 Delta Lake**: Reliable data storage with ACID transactions, making data more trustworthy.

### 🧪 Testing in Databricks:

Testing in Databricks typically involves:

```python
# Load data into a DataFrame
df = spark.read.format("csv").option("header", "true").load("/path/to/file.csv")

# Simple test: Check if required columns exist
required_columns = ["customer_id", "email", "signup_date"]
for column in required_columns:
    assert column in df.columns, f"Missing required column: {column}"
print("✅ All required columns present!")

# Test for no nulls in key fields
from pyspark.sql.functions import col, count, when, isnan

def test_no_nulls_in_key_fields(df, key_fields):
    # Count nulls in important columns
    null_counts = {c: df.filter(col(c).isNull() | isnan(c)).count() for c in key_fields}
    
    # Assert that there are no nulls
    for column, count in null_counts.items():
        assert count == 0, f"Column {column} contains {count} null values"
    
    print("✅ Test passed! No nulls in key fields.")

# Run the test
test_no_nulls_in_key_fields(df, ["customer_id", "email", "signup_date"])
```

## 🔄 What is ETL?

**ETL** stands for **Extract, Transform, Load**. It's the core process used in data integration to blend data from multiple sources.

### 🧩 ETL Process Breakdown:

1. **📥 Extract**: Pull data from source systems (databases, APIs, files, etc.)
2. **🔧 Transform**: Clean, validate, reformat, enrich, and structure the data
3. **📤 Load**: Write the processed data to a target system (data warehouse, data lake, etc.)

### 🧪 Testing ETL Processes:

Testing ETL involves verifying each stage works correctly:

1. **📥 Source Data Testing**: Verify data extraction is complete and accurate
2. **🔧 Transformation Testing**: Ensure business rules are correctly applied
3. **📤 Target Data Testing**: Validate the data is correctly loaded
4. **🔄 End-to-End Testing**: Test the entire pipeline

```python
def test_customer_transformation():
    # Sample input data
    input_data = [
        {"customer_id": "1001", "name": "JOHN DOE", "spend": "1234.56"}
    ]
    
    # Expected output after transformation
    expected_output = [
        {"customer_id": 1001, "name": "John Doe", "spend": 1234.56}
    ]
    
    # Apply transformation
    actual_output = transform_customer_data(input_data)
    
    # Assert the transformation worked correctly
    assert actual_output == expected_output, f"Expected {expected_output}, got {actual_output}"
    print("✅ Transformation test passed!")
```

## 📁 Loading Different Types of Data

### 📊 CSV Files

CSV (Comma Separated Values) files are simple text files with data in rows and columns separated by commas.

#### 🐍 Using Python (Pandas):

```python
import pandas as pd

# Basic CSV loading
df = pd.read_csv('data.csv')

# With options
df = pd.read_csv('data.csv', 
                 sep=',',              # Delimiter (could be tab '\t', semicolon ';', etc.)
                 header=0,             # Row to use as column names (0 = first row)
                 skiprows=2,           # Skip the first 2 rows
                 na_values=['NA', '?'], # Values to treat as NaN
                 dtype={'id': int},    # Specify column data types
                 parse_dates=['date']) # Parse date columns

# Preview data
print(df.head())

# Save as CSV
df.to_csv('processed_data.csv', index=False)
```

#### ⚡ Using PySpark:

```python
# Read CSV
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/path/to/data.csv")

# Write CSV
df.write.format("csv") \
    .option("header", "true") \
    .mode("overwrite") \
    .save("/path/to/output")
```

#### 🧪 Testing CSV Loading:

```python
def test_csv_loading():
    # Load the CSV
    df = pd.read_csv('test_data.csv')
    
    # Test data was loaded
    assert not df.empty, "CSV file loaded but dataframe is empty"
    
    # Test expected columns exist
    expected_columns = ['id', 'name', 'value']
    for col in expected_columns:
        assert col in df.columns, f"Expected column {col} missing from CSV"
    
    # Test row count
    assert len(df) > 0, "CSV loaded but contains no rows"
    
    print("✅ CSV loading test passed!")
```

### 📑 Excel Files

Excel files can contain multiple sheets, formulas, and formatting.

#### 🐍 Using Python (Pandas):

```python
import pandas as pd

# Basic Excel loading
df = pd.read_excel('data.xlsx')

# With options
df = pd.read_excel('data.xlsx',
                  sheet_name='Sheet1',  # Specific sheet (can be index or name)
                  header=0,             # Row to use as column names
                  skiprows=2,           # Skip the first 2 rows
                  usecols="A:C")        # Only read columns A through C

# Read multiple sheets
all_sheets = pd.read_excel('data.xlsx', sheet_name=None)  # Returns dict of dataframes

# Save as Excel
df.to_excel('processed_data.xlsx', sheet_name='Processed', index=False)
```

#### ⚡ Using PySpark:

```python
# Read Excel (requires spark-excel package)
df = spark.read.format("com.crealytics.spark.excel") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("dataAddress", "'Sheet1'!A1:Z10000") \
    .load("/path/to/data.xlsx")

# Write Excel
df.write.format("com.crealytics.spark.excel") \
    .option("header", "true") \
    .mode("overwrite") \
    .save("/path/to/output.xlsx")
```

#### 🧪 Testing Excel Loading:

```python
def test_excel_loading():
    # Load the Excel file
    df = pd.read_excel('test_data.xlsx')
    
    # Test data was loaded
    assert not df.empty, "Excel file loaded but dataframe is empty"
    
    # Test specific sheets exist when loading multiple sheets
    all_sheets = pd.read_excel('test_data.xlsx', sheet_name=None)
    expected_sheets = ['Data', 'Metadata']
    for sheet in expected_sheets:
        assert sheet in all_sheets, f"Expected sheet {sheet} missing from Excel"
    
    print("✅ Excel loading test passed!")
```

### 🗄️ MySQL Database

MySQL is a popular open-source relational database.

#### 🐍 Using Python (SQLAlchemy):

```python
import pandas as pd
from sqlalchemy import create_engine

# Create connection
engine = create_engine('mysql+pymysql://username:password@hostname:3306/database_name')

# Read data
query = "SELECT * FROM customers WHERE signup_date > '2023-01-01'"
df = pd.read_sql(query, engine)

# Write data
df.to_sql('processed_customers', engine, if_exists='replace', index=False)
```

#### ⚡ Using PySpark:

```python
# Connection properties
jdbc_url = "jdbc:mysql://hostname:3306/database_name"
connection_properties = {
    "user": "username",
    "password": "password",
    "driver": "com.mysql.cj.jdbc.Driver"
}

# Read from MySQL
df = spark.read.jdbc(url=jdbc_url,
                    table="customers",
                    properties=connection_properties)

# Write to MySQL
df.write.jdbc(url=jdbc_url,
             table="processed_customers",
             mode="overwrite",
             properties=connection_properties)
```

#### 🧪 Testing MySQL Connection and Queries:

```python
def test_mysql_connection():
    try:
        # Create connection
        engine = create_engine('mysql+pymysql://username:password@hostname:3306/database_name')
        
        # Test connection with simple query
        result = pd.read_sql("SELECT 1 as test", engine)
        assert result.test[0] == 1, "Connection test failed"
        
        # Test specific table exists
        tables = pd.read_sql("SHOW TABLES", engine)
        required_table = 'customers'
        assert required_table in tables.values, f"Required table {required_table} not found in database"
        
        print("✅ MySQL connection test passed!")
        return True
    except Exception as e:
        print(f"❌ MySQL connection test failed: {str(e)}")
        return False
```

### 🐘 PostgreSQL Database

PostgreSQL is a powerful, open-source object-relational database system.

#### 🐍 Using Python (SQLAlchemy):

```python
import pandas as pd
from sqlalchemy import create_engine

# Create connection
engine = create_engine('postgresql://username:password@hostname:5432/database_name')

# Read data
query = "SELECT * FROM users WHERE user_type = 'premium'"
df = pd.read_sql(query, engine)

# Write data
df.to_sql('premium_users', engine, if_exists='replace', index=False)
```

#### ⚡ Using PySpark:

```python
# Connection properties
jdbc_url = "jdbc:postgresql://hostname:5432/database_name"
connection_properties = {
    "user": "username",
    "password": "password",
    "driver": "org.postgresql.Driver"
}

# Read from PostgreSQL
df = spark.read.jdbc(url=jdbc_url,
                    table="users",
                    properties=connection_properties)

# Write to PostgreSQL
df.write.jdbc(url=jdbc_url,
             table="premium_users",
             mode="overwrite",
             properties=connection_properties)
```

#### 🧪 Testing PostgreSQL Query Results:

```python
def test_postgres_query_results():
    # Create connection
    engine = create_engine('postgresql://username:password@hostname:5432/database_name')
    
    # Run query
    df = pd.read_sql("SELECT user_id, status FROM users LIMIT 100", engine)
    
    # Test no null user_ids
    assert df['user_id'].isnull().sum() == 0, "Found null user_ids in results"
    
    # Test status values are valid
    valid_statuses = ['active', 'inactive', 'suspended']
    invalid_statuses = df[~df['status'].isin(valid_statuses)]['status'].unique()
    assert len(invalid_statuses) == 0, f"Found invalid status values: {invalid_statuses}"
    
    print("✅ PostgreSQL query test passed!")
```

### 💾 Databricks Delta Lake Tables

Delta Lake provides ACID transactions, schema enforcement, and time travel capabilities for Apache Spark.

#### ⚡ Using PySpark in Databricks:

```python
# Read from Delta table
df = spark.read.format("delta").load("/path/to/delta/table")

# or
df = spark.table("my_delta_table")

# Write to Delta table
df.write.format("delta").mode("overwrite").save("/path/to/delta/table")

# or
df.write.format("delta").mode("overwrite").saveAsTable("my_delta_table")

# Time travel (read data as of a specific version)
df = spark.read.format("delta").option("versionAsOf", "5").load("/path/to/delta/table")
```

#### 🧪 Testing Delta Lake Operations:

```python
def test_delta_write_read():
    # Create test data
    test_data = spark.createDataFrame([
        (1, "Alice", 100),
        (2, "Bob", 200),
        (3, "Charlie", 300)
    ], ["id", "name", "value"])
    
    # Write to Delta
    test_path = "/tmp/test_delta_table"
    test_data.write.format("delta").mode("overwrite").save(test_path)
    
    # Read it back
    read_df = spark.read.format("delta").load(test_path)
    
    # Test count matches
    orig_count = test_data.count()
    read_count = read_df.count()
    assert orig_count == read_count, f"Record count mismatch: {orig_count} vs {read_count}"
    
    # Test schema matches (column names and types)
    orig_columns = [f"{f.name}:{f.dataType}" for f in test_data.schema.fields]
    read_columns = [f"{f.name}:{f.dataType}" for f in read_df.schema.fields]
    assert orig_columns == read_columns, f"Schema mismatch: {orig_columns} vs {read_columns}"
    
    print("✅ Delta Lake write/read test passed!")
```

### 🌐 JSON Data

JSON (JavaScript Object Notation) is a common format for web APIs and configuration files.

#### 🐍 Using Python (Pandas):

```python
import pandas as pd
import json

# From JSON file
df = pd.read_json('data.json')

# From JSON string
json_str = '{"name": "John", "age": 30, "city": "New York"}'
data = json.loads(json_str)
df = pd.DataFrame([data])

# With complex nested JSON
df = pd.json_normalize(data, 
                      record_path=['children'],
                      meta=['name', 'age'])

# Write to JSON
df.to_json('processed_data.json', orient='records')
```

#### ⚡ Using PySpark:

```python
# Read JSON
df = spark.read.format("json").load("/path/to/data.json")

# Write JSON
df.write.format("json").mode("overwrite").save("/path/to/output")
```

#### 🧪 Testing JSON Parsing:

```python
def test_json_parsing():
    # Sample JSON
    json_str = '''
    {
        "records": [
            {"id": 1, "name": "Alice", "tags": ["tag1", "tag2"]},
            {"id": 2, "name": "Bob", "tags": ["tag3"]}
        ]
    }
    '''
    
    # Parse JSON
    data = json.loads(json_str)
    
    # Test structure
    assert "records" in data, "JSON missing 'records' key"
    assert isinstance(data["records"], list), "'records' is not a list"
    assert len(data["records"]) == 2, "Expected 2 records, got " + str(len(data["records"]))
    
    # Test record content
    assert data["records"][0]["id"] == 1, "First record ID mismatch"
    assert data["records"][1]["name"] == "Bob", "Second record name mismatch"
    assert "tag1" in data["records"][0]["tags"], "Missing expected tag"
    
    print("✅ JSON parsing test passed!")
```

### 🗃️ Parquet Files

Parquet is a columnar storage format optimized for big data processing.

#### 🐍 Using Python (Pandas):

```python
import pandas as pd

# Read Parquet
df = pd.read_parquet('data.parquet', engine='pyarrow')

# Write Parquet
df.to_parquet('processed_data.parquet', 
              compression='snappy',    # Compression algorithm
              engine='pyarrow')        # Parquet engine to use
```

#### ⚡ Using PySpark:

```python
# Read Parquet
df = spark.read.format("parquet").load("/path/to/data.parquet")

# Write Parquet
df.write.format("parquet") \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .save("/path/to/output")
```

#### 🧪 Testing Parquet Reading/Writing:

```python
def test_parquet_roundtrip():
    # Create test data
    import numpy as np
    original_df = pd.DataFrame({
        'id': range(1000),
        'value': np.random.rand(1000),
        'category': np.random.choice(['A', 'B', 'C'], size=1000)
    })
    
    # Write to Parquet
    test_path = 'test_data.parquet'
    original_df.to_parquet(test_path)
    
    # Read it back
    read_df = pd.read_parquet(test_path)
    
    # Test shape matches
    assert original_df.shape == read_df.shape, f"Shape mismatch: {original_df.shape} vs {read_df.shape}"
    
    # Test column dtypes match
    for col in original_df.columns:
        assert original_df[col].dtype == read_df[col].dtype, f"Dtype mismatch for {col}"
    
    # Test values match
    pd.testing.assert_frame_equal(original_df, read_df)
    
    print("✅ Parquet roundtrip test passed!")
```

## 📓 Jupyter Notebooks Setup

Jupyter Notebooks are interactive computing environments that allow you to create documents with live code, visualizations, and explanatory text.

### 🚀 Setting Up Jupyter:

1. **📥 Install Python**: If you don't have Python, download and install it from [python.org](https://python.org)

2. **🔧 Set up a Virtual Environment**:
   ```bash
   # Create a virtual environment
   python -m venv data_testing_env
   
   # Activate the environment (Windows)
   data_testing_env\Scripts\activate
   
   # Activate the environment (Mac/Linux)
   source data_testing_env/bin/activate
   ```

3. **📦 Install Jupyter and Required Packages**:
   ```bash
   # Install Jupyter
   pip install jupyter
   
   # Install basic data packages
   pip install pandas numpy matplotlib
   
   # Install testing packages
   pip install pytest great_expectations
   
   # Install database connectors
   pip install sqlalchemy pymysql psycopg2-binary
   
   # Install file format packages
   pip install openpyxl pyarrow fastparquet
   
   # Save your environment packages
   pip freeze > requirements.txt
   ```

4. **▶️ Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

5. **📝 Create a New Notebook**: Click "New" → "Python 3" to create a new notebook

### 🧪 Testing in Jupyter Notebooks:

You can run tests directly in Jupyter:

```python
# Import libraries
import pandas as pd
import numpy as np

# Load sample data
df = pd.read_csv('sample_data.csv')

# Run a simple data quality test
def test_no_missing_values():
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    
    # Assert no missing values
    assert missing_count == 0, f"Found {missing_count} missing values in the dataset"
    print("✅ No missing values test passed!")

# Execute the test
test_no_missing_values()
```

## 💻 Working with Virtual Environments

Virtual environments help isolate project dependencies from your system Python installation.

### 🔤 Basic Commands:

```bash
# Create a new virtual environment
python -m venv my_test_env

# Activate the environment (Windows)
my_test_env\Scripts\activate

# Activate the environment (Mac/Linux)
source my_test_env/bin/activate

# Deactivate the environment
deactivate

# Install packages
pip install package_name

# Install from requirements file
pip install -r requirements.txt
```

### 🧪 Testing Environment Setup:

```bash
# Create a test script (test_env.py)
echo "import pandas; import numpy; import matplotlib; print('✅ Environment test passed!')" > test_env.py

# Run the test script
python test_env.py

# If it runs without errors, your environment is set up correctly!
```

## 📚 Further Resources

For more detailed guides on data testing, check out:
- [Functional Testing for Data](functional_testing_readme.md)
- [Data Quality Checks](data_quality_readme.md)

Happy Testing! 🎉
