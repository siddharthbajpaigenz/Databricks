# 🔍 Data Quality Checks

This guide explains what data quality is, why it matters, and how to implement data quality checks in your projects, focusing on testing approaches. We'll start with basic concepts and progress to more advanced techniques.

## 📊 What is Data Quality?

Data quality refers to how well data meets the requirements for its intended use. High-quality data is:

- **✅ Accurate**: Correctly represents real-world values
- **🔄 Complete**: Contains all necessary information
- **🧩 Consistent**: Follows the same patterns and rules
- **⏱️ Timely**: Available when needed
- **🔍 Valid**: Conforms to defined rules and formats
- **🔎 Unique**: Free from duplication

## ❓ Why Data Quality Matters

Poor data quality can lead to:
- 🚫 Incorrect business decisions
- 💔 Failed analytics projects
- ⏱️ Wasted time and resources
- ⚠️ Compliance violations
- 🔒 Loss of trust in data systems

According to Gartner research, poor data quality costs organizations an average of $12.9 million annually! 💰

## 🔰 Basic Data Quality Checks

Here are simple checks that anyone can implement to start improving data quality:

### 1️⃣ Completeness Checks

Check for missing values in your data:

```python
import pandas as pd
import numpy as np

def check_completeness(df):
    """Check for missing values in each column"""
    # Count missing values per column
    missing_counts = df.isnull().sum()
    
    # Calculate percentage of missing values
    missing_percentage = (missing_counts / len(df)) * 100
    
    # Create a summary
    completeness_report = pd.DataFrame({
        'Missing Values': missing_counts,
        'Missing Percentage': missing_percentage.round(2)
    })
    
    # Output the report
    print("=== 🔍 COMPLETENESS CHECK ===")
    print(completeness_report)
    print("============================")
    
    # Return True if no missing values
    return missing_counts.sum() == 0

# Example usage
df = pd.read_csv('customer_data.csv')
is_complete = check_completeness(df)
if is_complete:
    print("✅ Data is complete - no missing values!")
else:
    print("⚠️ Data has missing values - check the report above")
```

### 2️⃣ Uniqueness Checks

Check for duplicate records:

```python
def check_uniqueness(df, key_columns):
    """Check for duplicate records based on key columns"""
    # Count total records
    total_records = len(df)
    
    # Count unique records
    unique_records = len(df.drop_duplicates(subset=key_columns))
    
    # Calculate duplicates
    duplicate_count = total_records - unique_records
    duplicate_percentage = (duplicate_count / total_records) * 100 if total_records > 0 else 0
    
    # Output the report
    print("=== 🔍 UNIQUENESS CHECK ===")
    print(f"Total Records: {total_records}")
    print(f"Unique Records: {unique_records}")
    print(f"Duplicate Records: {duplicate_count} ({duplicate_percentage:.2f}%)")
    print("==========================")
    
    # Show examples of duplicates if they exist
    if duplicate_count > 0:
        print("Examples of duplicate records:")
        # Find duplicate rows
        duplicates = df[df.duplicated(subset=key_columns, keep='first')]
        print(duplicates.head(3))  # Show first 3 duplicates
    
    # Return True if no duplicates
    return duplicate_count == 0

# Example usage
is_unique = check_uniqueness(df, key_columns=['customer_id'])
if is_unique:
    print("✅ Data has no duplicates based on key columns!")
else:
    print("⚠️ Data has duplicates - review the report above")
```

### 3️⃣ Range Checks

Validate that numeric values fall within expected ranges:

```python
def check_value_ranges(df, range_rules):
    """
    Check if values are within specified ranges
    
    Args:
        df: Pandas DataFrame
        range_rules: Dict of column names and their min/max values
                    e.g., {'age': {'min': 0, 'max': 120}}
    """
    all_valid = True
    
    print("=== 🔍 RANGE CHECK ===")
    
    for column, rules in range_rules.items():
        if column not in df.columns:
            print(f"❌ Column '{column}' not found in the data")
            all_valid = False
            continue
            
        min_val = rules.get('min')
        max_val = rules.get('max')
        
        # Check minimum values if specified
        if min_val is not None:
            below_min = df[df[column] < min_val]
            if len(below_min) > 0:
                print(f"⚠️ {len(below_min)} values in '{column}' are below minimum {min_val}")
                print(below_min.head(3))  # Show examples
                all_valid = False
        
        # Check maximum values if specified
        if max_val is not None:
            above_max = df[df[column] > max_val]
            if len(above_max) > 0:
                print(f"⚠️ {len(above_max)} values in '{column}' are above maximum {max_val}")
                print(above_max.head(3))  # Show examples
                all_valid = False
    
    print("====================")
    
    if all_valid:
        print("✅ All values are within expected ranges!")
    
    return all_valid

# Example usage
range_rules = {
    'age': {'min': 18, 'max': 100},
    'transaction_amount': {'min': 0, 'max': 10000}
}

values_in_range = check_value_ranges(df, range_rules)
```

### 4️⃣ Format Checks

Validate that data follows expected formats (like email addresses):

```python
import re

def check_formats(df, format_rules):
    """
    Check if values follow expected formats using regex patterns
    
    Args:
        df: Pandas DataFrame
        format_rules: Dict of column names and their regex patterns
                     e.g., {'email': r'[^@]+@[^@]+\.[^@]+'}
    """
    all_valid = True
    
    print("=== 🔍 FORMAT CHECK ===")
    
    for column, pattern in format_rules.items():
        if column not in df.columns:
            print(f"❌ Column '{column}' not found in the data")
            all_valid = False
            continue
        
        # Create a regex pattern
        regex = re.compile(pattern)
        
        # Apply the pattern to each value in the column
        # First convert to string and handle NaN values
        invalid_mask = ~df[column].fillna('').astype(str).apply(lambda x: bool(regex.match(x)))
        invalid_rows = df[invalid_mask]
        
        if len(invalid_rows) > 0:
            print(f"⚠️ {len(invalid_rows)} values in '{column}' do not match expected format")
            print(invalid_rows.head(3))  # Show examples
            all_valid = False
    
    print("=====================")
    
    if all_valid:
        print("✅ All values follow expected formats!")
    
    return all_valid

# Example usage
format_rules = {
    'email': r'^[^@]+@[^@]+\.[^@]+$',
    'phone': r'^\d{10}$'
}

formats_valid = check_formats(df, format_rules)
```

### 5️⃣ Data Type Checks

Verify that columns have the correct data types:

```python
def check_data_types(df, expected_types):
    """
    Check if columns have the expected data types
    
    Args:
        df: Pandas DataFrame
        expected_types: Dict of column names and their expected types
                        e.g., {'age': 'int', 'name': 'object'}
    """
    all_valid = True
    
    print("=== 🔍 DATA TYPE CHECK ===")
    
    for column, expected_type in expected_types.items():
        if column not in df.columns:
            print(f"❌ Column '{column}' not found in the data")
            all_valid = False
            continue
        
        # Get actual type
        actual_type = df[column].dtype.name
        
        # Check if types match
        if actual_type != expected_type:
            print(f"⚠️ Column '{column}' has type '{actual_type}', expected '{expected_type}'")
            all_valid = False
    
    print("=========================")
    
    if all_valid:
        print("✅ All columns have the expected data types!")
    
    return all_valid

# Example usage
expected_types = {
    'customer_id': 'int64',
    'name': 'object',
    'signup_date': 'datetime64[ns]',
    'active': 'bool'
}

types_valid = check_data_types(df, expected_types)
```

## 🧩 Intermediate Data Quality Checks

Now let's look at more sophisticated data quality checks:

### 1️⃣ Consistency Checks

Verify that related data values are consistent with each other:

```python
def check_consistency(df, consistency_rules):
    """
    Check consistency across multiple columns
    
    Args:
        df: Pandas DataFrame
        consistency_rules: List of dicts with 'condition' (string) and 'message' (string)
    """
    all_consistent = True
    
    print("=== 🔍 CONSISTENCY CHECK ===")
    
    for rule in consistency_rules:
        condition = rule['condition']
        message = rule['message']
        
        # Evaluate the condition
        inconsistent_rows = df.query(f"not ({condition})")
        
        if len(inconsistent_rows) > 0:
            print(f"⚠️ {len(inconsistent_rows)} rows violate consistency rule: {message}")
            print(inconsistent_rows.head(3))  # Show examples
            all_consistent = False
    
    print("==========================")
    
    if all_consistent:
        print("✅ All consistency checks passed!")
    
    return all_consistent

# Example usage
consistency_rules = [
    {
        'condition': 'order_date <= delivery_date',
        'message': 'Delivery date must be on or after order date'
    },
    {
        'condition': 'discount_amount <= total_amount',
        'message': 'Discount cannot exceed total amount'
    }
]

is_consistent = check_consistency(df, consistency_rules)
```

### 2️⃣ Statistical Checks

Identify outliers and anomalies using statistical methods:

```python
def check_statistical_outliers(df, columns_to_check, n_std=3):
    """
    Identify statistical outliers (values more than n standard deviations from mean)
    
    Args:
        df: Pandas DataFrame
        columns_to_check: List of numeric columns to check
        n_std: Number of standard deviations to use as threshold
    """
    has_outliers = False
    
    print("=== 🔍 STATISTICAL OUTLIERS CHECK ===")
    
    for column in columns_to_check:
        if column not in df.columns:
            print(f"❌ Column '{column}' not found in the data")
            continue
            
        # Skip if not numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"⚠️ Column '{column}' is not numeric, skipping outlier check")
            continue
            
        # Skip if all null
        if df[column].isnull().all():
            print(f"⚠️ Column '{column}' only contains null values, skipping outlier check")
            continue
            
        # Calculate mean and standard deviation
        mean_val = df[column].mean()
        std_val = df[column].std()
        
        # Define thresholds
        lower_threshold = mean_val - (n_std * std_val)
        upper_threshold = mean_val + (n_std * std_val)
        
        # Find outliers
        outliers = df[(df[column] < lower_threshold) | (df[column] > upper_threshold)]
        
        if len(outliers) > 0:
            print(f"⚠️ {len(outliers)} outliers found in '{column}'")
            print(f"   Mean: {mean_val:.2f}, Std Dev: {std_val:.2f}")
            print(f"   Thresholds: [{lower_threshold:.2f}, {upper_threshold:.2f}]")
            print(outliers.head(3))  # Show examples
            has_outliers = True
    
    print("==================================")
    
    if not has_outliers:
        print("✅ No statistical outliers found!")
    
    return not has_outliers

# Example usage
numeric_columns = ['age', 'income', 'order_value']
no_outliers = check_statistical_outliers(df, numeric_columns, n_std=3)
```

### 3️⃣ Referential Integrity Checks

Verify that references between datasets are valid:

```python
def check_referential_integrity(df, foreign_key, reference_df, primary_key):
    """
    Check referential integrity between datasets
    
    Args:
        df: DataFrame with foreign keys
        foreign_key: Column name of the foreign key
        reference_df: DataFrame with primary keys
        primary_key: Column name of the primary key
    """
    # Get unique values from foreign key column (skip nulls)
    foreign_key_values = df[df[foreign_key].notnull()][foreign_key].unique()
    
    # Get primary key values
    primary_key_values = set(reference_df[primary_key])
    
    # Find values that don't exist in the reference dataset
    invalid_references = [value for value in foreign_key_values 
                         if value not in primary_key_values]
    
    print("=== 🔍 REFERENTIAL INTEGRITY CHECK ===")
    
    if invalid_references:
        print(f"⚠️ {len(invalid_references)} values in '{foreign_key}' don't have matching '{primary_key}' values")
        print(f"Examples of invalid references: {invalid_references[:5]}")
        
        # Show rows with invalid references
        invalid_rows = df[df[foreign_key].isin(invalid_references)]
        print(invalid_rows.head(3))
        
        return False
    else:
        print(f"✅ All '{foreign_key}' values have matching '{primary_key}' values")
        return True

# Example usage
# Check if all product_ids in orders exist in the products table
orders_df = pd.read_csv('orders.csv')
products_df = pd.read_csv('products.csv')

integrity_valid = check_referential_integrity(
    orders_df, 'product_id', 
    products_df, 'id'
)
```

### 4️⃣ Pattern and Distribution Checks

Analyze patterns and distributions in your data:

```python
def check_value_distribution(df, column, expected_distribution=None):
    """
    Check the distribution of values in a column
    
    Args:
        df: Pandas DataFrame
        column: Column name to check
        expected_distribution: Optional dict of expected value frequencies
    """
    import matplotlib.pyplot as plt
    
    if column not in df.columns:
        print(f"❌ Column '{column}' not found in the data")
        return False
    
    # Get actual distribution
    actual_distribution = df[column].value_counts(normalize=True)
    
    print(f"=== 🔍 DISTRIBUTION CHECK: {column} ===")
    print(f"Top 5 values:")
    print(actual_distribution.head(5))
    
    # If an expected distribution is provided, compare
    if expected_distribution:
        print("\nComparing to expected distribution:")
        for value, expected_freq in expected_distribution.items():
            actual_freq = actual_distribution.get(value, 0)
            diff = abs(actual_freq - expected_freq)
            
            if diff > 0.05:  # More than 5% difference
                print(f"⚠️ Value '{value}': Expected ~{expected_freq:.2f}, Got {actual_freq:.2f}")
            else:
                print(f"✅ Value '{value}': Expected ~{expected_freq:.2f}, Got {actual_freq:.2f}")
    
    # Plot distribution
    plt.figure(figsize=(10, 5))
    if df[column].nunique() < 10:  # For categorical with few values
        actual_distribution.plot(kind='bar')
    else:  # For numeric or many categories
        df[column].hist()
    
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()
    
    print("=====================================")
    return True

# Example usage
# Expected category distribution
expected_categories = {
    'Premium': 0.10,  # 10% should be Premium
    'Gold': 0.20,     # 20% should be Gold
    'Silver': 0.30,   # 30% should be Silver
    'Bronze': 0.40    # 40% should be Bronze
}

check_value_distribution(df, 'customer_category', expected_categories)
```

## 🛠️ Advanced Data Quality Techniques

Now for some more advanced approaches:

### 1️⃣ Automated Data Profiling

Generate comprehensive data profiles automatically:

```python
def profile_dataset(df):
    """Generate a comprehensive data profile"""
    # Basic dataset info
    print("=== 📊 DATASET PROFILE ===")
    print(f"Records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Column profiles
    profiles = []
    
    for column in df.columns:
        # Determine data type category
        dtype = df[column].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            data_type = 'numeric'
        elif pd.api.types.is_datetime64_dtype(dtype):
            data_type = 'datetime'
        else:
            data_type = 'categorical/text'
        
        # Calculate common metrics
        missing_count = df[column].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        unique_values = df[column].nunique()
        unique_percent = (unique_values / len(df)) * 100
        
        profile = {
            'column': column,
            'data_type': data_type,
            'missing_count': missing_count,
            'missing_percent': f"{missing_percent:.2f}%",
            'unique_values': unique_values,
            'unique_percent': f"{unique_percent:.2f}%"
        }
        
        # Add type-specific metrics
        if data_type == 'numeric':
            profile.update({
                'min': df[column].min(),
                'max': df[column].max(),
                'mean': df[column].mean(),
                'std_dev': df[column].std(),
                'median': df[column].median()
            })
        elif data_type == 'categorical/text':
            # Get most common values
            value_counts = df[column].value_counts()
            if not value_counts.empty:
                most_common = value_counts.index[0]
                most_common_count = value_counts.iloc[0]
                most_common_percent = (most_common_count / len(df)) * 100
                profile.update({
                    'most_common': most_common,
                    'most_common_count': most_common_count,
                    'most_common_percent': f"{most_common_percent:.2f}%"
                })
        
        profiles.append(profile)
    
    # Convert to DataFrame for nice display
    profile_df = pd.DataFrame(profiles)
    print("\n=== 📋 COLUMN PROFILES ===")
    print(profile_df)
    
    # Return the profile data
    return profile_df

# Example usage
profile = profile_dataset(df)
```

### 2️⃣ Using Great Expectations

[Great Expectations](https://greatexpectations.io/) is a powerful Python library for data validation:

```python
import great_expectations as ge

def validate_with_great_expectations(df):
    """Validate data with Great Expectations"""
    # Convert pandas DataFrame to Great Expectations DataFrame
    ge_df = ge.from_pandas(df)
    
    print("=== 🔍 GREAT EXPECTATIONS VALIDATION ===")
    
    # Define and validate expectations
    results = ge_df.expect_compound_columns_to_be_unique(
        column_list=['customer_id', 'order_date'],
        result_format='COMPLETE'
    )
    print(f"✅ Unique customer orders: {results['success']}")
    
    results = ge_df.expect_column_values_to_be_between(
        column='order_amount',
        min_value=0,
        max_value=10000,
        result_format='COMPLETE'
    )
    print(f"✅ Order amounts in range: {results['success']}")
    
    results = ge_df.expect_column_values_to_match_regex(
        column='email',
        regex=r'^[^@]+@[^@]+\.[^@]+$',
        result_format='COMPLETE'
    )
    print(f"✅ Valid email formats: {results['success']}")
    
    # Save expectations to a file for future use
    ge_df.save_expectation_suite('my_expectations.json')
    
    print("Great Expectations suite saved to my_expectations.json")
    print("=======================================")

# Example usage
validate_with_great_expectations(df)
```

### 3️⃣ Building Quality Checks into Data Pipelines

Here's an example of incorporating quality checks into an ETL pipeline:

```python
def etl_pipeline_with_quality_checks():
    """ETL pipeline with integrated data quality checks"""
    # 1. Extract
    print("📥 Extracting data...")
    raw_data = pd.read_csv('source_data.csv')
    
    # 2. Quality check on raw data
    print("🔍 Checking raw data quality...")
    raw_data_complete = check_completeness(raw_data)
    if not raw_data_complete:
        print("⚠️ Warning: Raw data has missing values, proceeding with caution")
    
    # 3. Transform
    print("🔄 Transforming data...")
    # Example transformation: Clean customer names, calculate total
    transformed_data = raw_data.copy()
    transformed_data['customer_name'] = transformed_data['customer_name'].str.title()
    transformed_data['total'] = transformed_data['quantity'] * transformed_data['price']
    
    # 4. Quality check on transformed data
    print("🔍 Checking transformed data quality...")
    range_rules = {'total': {'min': 0}}
    totals_valid = check_value_ranges(transformed_data, range_rules)
    if not totals_valid:
        print("❌ Error: Transformed data has invalid totals, stopping pipeline")
        return False
    
    # 5. Load
    print("📤 Loading data...")
    transformed_data.to_csv('processed_data.csv', index=False)
    
    print("✅ ETL pipeline completed successfully!")
    return True

# Run the pipeline
success = etl_pipeline_with_quality_checks()
```

## 🧪 Implementing Comprehensive Data Quality Testing

Here's how to build a more comprehensive testing framework:

### 1️⃣ Create a Data Quality Test Suite

```python
class DataQualityTestSuite:
    """A comprehensive data quality test suite"""
    
    def __init__(self, df):
        self.df = df
        self.results = {}
    
    def run_all_tests(self):
        """Run all data quality tests"""
        print("=== 🧪 RUNNING DATA QUALITY TEST SUITE ===")
        
        # Run each test and store results
        self.results['completeness'] = self.test_completeness()
        self.results['uniqueness'] = self.test_uniqueness(['customer_id'])
        self.results['value_ranges'] = self.test_value_ranges({
            'age': {'min': 18, 'max': 120},
            'order_amount': {'min': 0}
        })
        self.results['formats'] = self.test_formats({
            'email': r'^[^@]+@[^@]+\.[^@]+$',
            'phone': r'^\d{10}$'
        })
        
        # Calculate overall pass rate
        passed_tests = sum(1 for result in self.results.values() if result['passed'])
        total_tests = len(self.results)
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print("\n=== 📋 TEST SUITE SUMMARY ===")
        print(f"Tests Passed: {passed_tests}/{total_tests} ({pass_rate:.1f}%)")
        for test_name, result in self.results.items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        print("================================")
        
        return pass_rate >= 80  # Pass if at least 80% of tests pass
    
    def test_completeness(self):
        """Test for missing values"""
        missing_counts = self.df.isnull().sum()
        passed = missing_counts.sum() == 0
        
        return {
            'passed': passed,
            'details': {
                'missing_counts': missing_counts.to_dict()
            }
        }
    
    def test_uniqueness(self, key_columns):
        """Test for duplicate records"""
        duplicates = self.df.duplicated(subset=key_columns).sum()
        passed = duplicates == 0
        
        return {
            'passed': passed,
            'details': {
                'duplicate_count': duplicates,
                'key_columns': key_columns
            }
        }
    
    def test_value_ranges(self, range_rules):
        """Test for values within expected ranges"""
        results = {}
        all_passed = True
        
        for column, rules in range_rules.items():
            if column not in self.df.columns:
                results[column] = {
                    'passed': False,
                    'error': 'Column not found'
                }
                all_passed = False
                continue
            
            min_val = rules.get('min')
            max_val = rules.get('max')
            
            below_min_count = 0
            above_max_count = 0
            
            if min_val is not None:
                below_min_count = (self.df[column] < min_val).sum()
            
            if max_val is not None:
                above_max_count = (self.df[column] > max_val).sum()
            
            column_passed = below_min_count == 0 and above_max_count == 0
            
            results[column] = {
                'passed': column_passed,
                'below_min_count': below_min_count,
                'above_max_count': above_max_count
            }
            
            if not column_passed:
                all_passed = False
        
        return {
            'passed': all_passed,
            'details': results
        }
    
    def test_formats(self, format_rules):
        """Test for values matching expected formats"""
        import re
        results = {}
        all_passed = True
        
        for column, pattern in format_rules.items():
            if column not in self.df.columns:
                results[column] = {
                    'passed': False,
                    'error': 'Column not found'
                }
                all_passed = False
                continue
            
            regex = re.compile(pattern)
            invalid_count = sum(1 for val in self.df[column].fillna('').astype(str) 
                               if not bool(regex.match(val)))
            
            column_passed = invalid_count == 0
            
            results[column] = {
                'passed': column_passed,
                'invalid_count': invalid_count
            }
            
            if not column_passed:
                all_passed = False
        
        return {
            'passed': all_passed,
            'details': results
        }

# Example usage
test_suite = DataQualityTestSuite(df)
overall_quality_good = test_suite.run_all_tests()
```

### 2️⃣ Scheduled Data Quality Monitoring

Set up regular monitoring of data quality metrics:

```python
def schedule_data_quality_monitoring():
    """Example of how to schedule regular data quality monitoring"""
    import schedule
    import time
    
    def run_daily_quality_checks():
        """Function to run daily data quality checks"""
        print(f"🕒 Running daily data quality checks at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load the latest data
        try:
            df = pd.read_csv('latest_data.csv')
            
            # Run test suite
            test_suite = DataQualityTestSuite(df)
            quality_good = test_suite.run_all_tests()
            
            # Send alerts if quality is poor
            if not quality_good:
                send_alert("⚠️ Data quality issues detected!")
            
            # Save quality metrics for trending
            save_quality_metrics(test_suite.results)
            
        except Exception as e:
            print(f"❌ Error in data quality checks: {str(e)}")
            send_alert(f"❌ Data quality check failed with error: {str(e)}")
    
    def send_alert(message):
        """Send alert (placeholder function)"""
        print(f"🚨 ALERT: {message}")
        # In a real system, this would send an email, Slack message, etc.
    
    def save_quality_metrics(results):
        """Save quality metrics for trending (placeholder)"""
        print("📊 Saving quality metrics for trending...")
        # In a real system, this would save to a database
    
    # Schedule the job to run daily at 8:00 AM
    schedule.every().day.at("08:00").do(run_daily_quality_checks)
    
    print("⏰ Data quality monitoring scheduled!")
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)

# Example of how you would call this (don't actually run in a notebook)
# schedule_data_quality_monitoring()
```

## 🧪 Testing Different Data Sources

### 1️⃣ Testing CSV Data Quality

```python
def test_csv_data_quality(csv_file):
    """Test data quality of a CSV file"""
    import pandas as pd
    
    print(f"=== 🔍 TESTING CSV DATA QUALITY: {csv_file} ===")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load CSV: {str(e)}")
        return False
    
    # Basic checks
    
    # 1. Check for missing values
    missing_by_column = df.isnull().sum()
    total_missing = missing_by_column.sum()
    if total_missing > 0:
        print(f"⚠️ Found {total_missing} missing values across all columns")
        print("Top columns with missing values:")
        print(missing_by_column[missing_by_column > 0].sort_values(ascending=False).head())
    else:
        print("✅ No missing values found")
    
    # 2. Check for duplicates
    if 'id' in df.columns:
        duplicate_ids = df['id'].duplicated().sum()
        if duplicate_ids > 0:
            print(f"⚠️ Found {duplicate_ids} duplicate IDs")
        else:
            print("✅ No duplicate IDs found")
    
    # 3. Check numeric columns for outliers
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            print(f"⚠️ Found {len(outliers)} outliers in column '{col}'")
    
    print("============================================")
    return True

# Example usage
test_csv_data_quality('customers.csv')
```

### 2️⃣ Testing Excel Data Quality

```python
def test_excel_data_quality(excel_file, sheet_name=None):
    """Test data quality of an Excel file"""
    import pandas as pd
    
    print(f"=== 🔍 TESTING EXCEL DATA QUALITY: {excel_file} ===")
    
    # Load Excel
    try:
        if sheet_name:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            print(f"✅ Successfully loaded sheet '{sheet_name}' with {len(df)} rows and {len(df.columns)} columns")
        else:
            # Get all sheets
            xlsx = pd.ExcelFile(excel_file)
            print(f"✅ Excel file contains {len(xlsx.sheet_names)} sheets: {xlsx.sheet_names}")
            
            # Load first sheet
            df = pd.read_excel(excel_file, sheet_name=0)
            print(f"✅ Loaded first sheet '{xlsx.sheet_names[0]}' with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load Excel file: {str(e)}")
        return False
    
    # Check for common Excel issues
    
    # 1. Check for hidden rows or columns (if possible)
    print("⚠️ Note: Cannot automatically detect hidden rows/columns in pandas")
    
    # 2. Check for merged cells (indirectly by looking for duplicate headers)
    if df.columns.duplicated().any():
        print(f"⚠️ Found {df.columns.duplicated().sum()} duplicate column names, possibly from merged cells")
    
    # 3. Check for formulas (indirectly)
    for col in df.columns:
        sample_values = df[col].astype(str).head(10)
        if any(val.startswith('=') for val in sample_values):
            print(f"⚠️ Column '{col}' may contain Excel formulas")
    
    # 4. Basic data quality checks
    test_csv_data_quality(df)  # Reuse the CSV quality function
    
    print("==============================================")
    return df

# Example usage
test_excel_data_quality('customers.xlsx', sheet_name='Customer Data')
```

### 3️⃣ Testing Database Data Quality

```python
def test_database_data_quality(connection_string, table_name):
    """Test data quality of a database table"""
    import pandas as pd
    from sqlalchemy import create_engine, inspect
    
    print(f"=== 🔍 TESTING DATABASE DATA QUALITY: {table_name} ===")
    
    try:
        # Create engine
        engine = create_engine(connection_string)
        
        # Get table info
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        primary_keys = inspector.get_primary_keys(table_name)
        
        print(f"✅ Connected to database, table has {len(columns)} columns")
        print(f"✅ Primary keys: {primary_keys}")
        
        # Load sample data (limit to avoid memory issues)
        query = f"SELECT * FROM {table_name} LIMIT 1000"
        sample_df = pd.read_sql(query, engine)
        
        print(f"✅ Loaded {len(sample_df)} sample rows")
        
        # Check for NULL in primary keys
        if primary_keys:
            for pk in primary_keys:
                if pk in sample_df.columns and sample_df[pk].isnull().any():
                    print(f"❌ Found NULL values in primary key column '{pk}'")
        
        # Run general data quality checks
        check_completeness(sample_df)
        
        # Get table statistics
        count_query = f"SELECT COUNT(*) AS row_count FROM {table_name}"
        row_count = pd.read_sql(count_query, engine).iloc[0]['row_count']
        
        print(f"✅ Total rows in table: {row_count}")
        
        # Check for duplicate records
        if primary_keys:
            dupes_query = f"""
            SELECT {', '.join(primary_keys)}, COUNT(*) as count
            FROM {table_name}
            GROUP BY {', '.join(primary_keys)}
            HAVING COUNT(*) > 1
            LIMIT 5
            """
            
            dupes = pd.read_sql(dupes_query, engine)
            if len(dupes) > 0:
                print(f"❌ Found {len(dupes)} duplicate primary key sets")
                print(dupes)
        
        print("==============================================")
        return True
    
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        return False

# Example usage
# test_database_data_quality('sqlite:///customers.db', 'customers')
```

## 📝 Best Practices for Data Quality

1. **🔄 Start Early**: Implement quality checks from the beginning of a project

2. **🤖 Automate**: Build automated quality checks into your pipelines

3. **📈 Monitor Trends**: Track quality metrics over time to spot degradation

4. **🎯 Focus on Impact**: Prioritize quality issues based on business impact

5. **📝 Document Rules**: Create clear documentation of data quality rules

6. **📊 Use Visualization**: Visualize quality metrics for easier understanding

7. **🔍 Test at Multiple Levels**: Check quality at various stages in the data lifecycle

8. **👥 Involve Stakeholders**: Get business users involved in defining quality requirements

9. **🔍 Profile First**: Always profile new data before diving into detailed quality checks

10. **🧪 Test Incrementally**: Start with basic checks, then add more sophisticated ones

## 🛠️ Data Quality Tools

Here are some popular tools for data quality management:

1. **Great Expectations**: Powerful data validation framework
   ```bash
   pip install great-expectations
   ```

2. **Deequ**: Data quality tool for big data built on Apache Spark
   ```bash
   # For PySpark environment
   pip install pydeequ
   ```

3. **Pandas Profiling**: Automated exploratory data analysis
   ```bash
   pip install pandas-profiling
   ```

4. **dbt**: Data transformation tool with built-in testing
   ```bash
   pip install dbt-core
   ```

5. **Soda Core**: Testing framework for multiple data platforms
   ```bash
   pip install soda-core
   ```

6. **marbles**: Python testing library with enhanced assertions
   ```bash
   pip install marbles
   ```

## 📚 Simple Data Quality Project Example

Here's a complete example of a small data quality project:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime

class DataQualityChecker:
    """A class to check data quality and generate reports"""
    
    def __init__(self, df, name="Dataset"):
        """Initialize with a dataframe"""
        self.df = df
        self.name = name
        self.report = {}
        
    def run_all_checks(self):
        """Run all data quality checks"""
        print(f"🔍 Running data quality checks on {self.name}...")
        
        # Basic info
        self.report['basic_info'] = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'memory_usage': self.df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Run specific checks
        self.check_missing_values()
        self.check_duplicates()
        self.check_data_types()
        self.check_outliers()
        
        # Print summary
        self.print_summary()
        
        return self.report
    
    def check_missing_values(self):
        """Check for missing values"""
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        
        missing_stats = pd.DataFrame({
            'missing_count': missing,
            'missing_percent': missing_percent
        })
        
        # Only include columns with missing values
        missing_stats = missing_stats[missing_stats['missing_count'] > 0]
        
        self.report['missing_values'] = {
            'total_missing': missing.sum(),
            'columns_with_missing': len(missing_stats),
            'missing_stats': missing_stats.to_dict()
        }
    
    def check_duplicates(self):
        """Check for duplicate rows"""
        # Check for exact duplicates
        exact_dupes = self.df.duplicated().sum()
        
        # If primary key columns are provided, check for duplicates by primary key
        pk_dupes = 0
        if 'id' in self.df.columns:
            pk_dupes = self.df.duplicated(subset=['id']).sum()
        
        self.report['duplicates'] = {
            'exact_duplicates': exact_dupes,
            'pk_duplicates': pk_dupes
        }
    
    def check_data_types(self):
        """Check data types and potential mismatches"""
        dtypes = self.df.dtypes.astype(str)
        
        # Check numeric columns that might be stored as strings
        potential_numeric = []
        
        for col in self.df.select_dtypes(include=['object']):
            # Skip if too many unique values (likely not categorical)
            if self.df[col].nunique() > 100:
                continue
                
            # Sample values (non-null)
            sample = self.df[col].dropna().astype(str).iloc[:100]
            
            # Check if values look numeric
            numeric_pattern = re.compile(r'^-?\d+(\.\d+)?$')
            numeric_match_pct = sum(bool(numeric_pattern.match(str(x))) for x in sample) / len(sample)
            
            if numeric_match_pct > 0.8:  # If >80% of values match numeric pattern
                potential_numeric.append({
                    'column': col,
                    'match_percent': numeric_match_pct * 100
                })
        
        self.report['data_types'] = {
            'dtypes': dtypes.to_dict(),
            'potential_numeric': potential_numeric
        }
    
    def check_outliers(self):
        """Check for outliers in numeric columns"""
        outliers = {}
        
        for col in self.df.select_dtypes(include=['number']):
            # Skip if too many unique values
            if self.df[col].nunique() <= 1:
                continue
                
            # Calculate IQR
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            # Count outliers
            outlier_count = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].shape[0]
            outlier_percent = (outlier_count / len(self.df)) * 100
            
            if outlier_percent > 0:
                outliers[col] = {
                    'outlier_count': outlier_count,
                    'outlier_percent': outlier_percent,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'min': self.df[col].min(),
                    'max': self.df[col].max()
                }
        
        self.report['outliers'] = outliers
    
    def print_summary(self):
        """Print a summary of the data quality report"""
        print("\n====== 📊 DATA QUALITY REPORT ======")
        print(f"Dataset: {self.name}")
        print(f"Rows: {self.report['basic_info']['rows']}, Columns: {self.report['basic_info']['columns']}")
        print(f"Memory usage: {self.report['basic_info']['memory_usage']:.2f} MB")
        print(f"Timestamp: {self.report['basic_info']['timestamp']}")
        
        print("\n----- 🔍 MISSING VALUES -----")
        if self.report['missing_values']['total_missing'] == 0:
            print("✅ No missing values found")
        else:
            print(f"❌ Found {self.report['missing_values']['total_missing']} missing values in {self.report['missing_values']['columns_with_missing']} columns")
            # Show top 5 columns with most missing values
            if self.report['missing_values']['columns_with_missing'] > 0:
                missing_stats = pd.DataFrame(self.report['missing_values']['missing_stats'])
                print("\nTop columns with missing values:")
                print(missing_stats['missing_percent'].sort_values(ascending=False).head())
        
        print("\n----- 🔍 DUPLICATES -----")
        if self.report['duplicates']['exact_duplicates'] == 0:
            print("✅ No exact duplicate rows found")
        else:
            print(f"❌ Found {self.report['duplicates']['exact_duplicates']} exact duplicate rows")
        
        if 'id' in self.df.columns:
            if self.report['duplicates']['pk_duplicates'] == 0:
                print("✅ No duplicate IDs found")
            else:
                print(f"❌ Found {self.report['duplicates']['pk_duplicates']} duplicate IDs")
        
        print("\n----- 🔍 DATA TYPES -----")
        print("Column data types:")
        for col, dtype in list(self.report['data_types']['dtypes'].items())[:5]:
            print(f"  - {col}: {dtype}")
        if len(self.report['data_types']['dtypes']) > 5:
            print(f"  - ... and {len(self.report['data_types']['dtypes']) - 5} more columns")
        
        if self.report['data_types']['potential_numeric']:
            print("\n⚠️ Potential numeric columns stored as strings:")
            for col_info in self.report['data_types']['potential_numeric']:
                print(f"  - {col_info['column']}: {col_info['match_percent']:.1f}% of values look numeric")
        
        print("\n----- 🔍 OUTLIERS -----")
        if not self.report['outliers']:
            print("✅ No significant outliers found in numeric columns")
        else:
            print(f"⚠️ Found outliers in {len(self.report['outliers'])} numeric columns:")
            for col, stats in self.report['outliers'].items():
                print(f"  - {col}: {stats['outlier_count']} outliers ({stats['outlier_percent']:.1f}%)")
                print(f"    Range: [{stats['min']} to {stats['max']}], IQR bounds: [{stats['lower_bound']:.2f} to {stats['upper_bound']:.2f}]")
        
        print("\n==================================")
    
    def plot_missing_values(self):
        """Plot missing values heatmap"""
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.show()
    
    def plot_outliers(self):
        """Plot boxplots for numeric columns to visualize outliers"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            print("No numeric columns to plot")
            return
        
        # Plot at most 10 columns
        plot_cols = numeric_cols[:min(10, len(numeric_cols))]
        
        plt.figure(figsize=(12, len(plot_cols) * 2))
        for i, col in enumerate(plot_cols):
            plt.subplot(len(plot_cols), 1, i+1)
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot of {col}')
            plt.tight_layout()
        
        plt.tight_layout(pad=2)
        plt.show()

# Example usage
# Load sample data
df = pd.read_csv('sample_data.csv')

# Create quality checker and run checks
checker = DataQualityChecker(df, name="Customer Orders")
report = checker.run_all_checks()

# Plot visualizations
checker.plot_missing_values()
checker.plot_outliers()
```

## 📊 Summary

Data quality is essential for reliable analytics and decision-making. By implementing comprehensive quality checks, you can:

- ✅ Identify and fix data issues early
- 🔒 Build trust in your data systems
- 👍 Ensure analytics projects succeed
- 📈 Support better business decisions

Start with basic checks and gradually add more sophisticated validation as your data needs grow. Remember that data quality is an ongoing process, not a one-time project.

By integrating data quality checks into your data pipelines and regularly monitoring quality metrics, you can create a foundation for reliable data analysis and reporting.
