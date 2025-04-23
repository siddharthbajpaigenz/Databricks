# 🧪 Functional Testing for Data

This guide covers functional testing in data engineering, from basic concepts to advanced techniques, focusing on how to verify that your data pipelines and transformations work as expected.

## 🔍 What is Functional Testing in Data?

**Functional testing for data** is the process of verifying that data processing systems work according to specifications. Unlike unit testing (which tests individual components) or integration testing (which tests interactions between components), functional testing validates that the entire data flow meets business requirements.

![Functional Testing Concept](https://i.imgur.com/hZ8j2Wy.png)

### 🎯 Core Concept:
```
Input Data → Process/Transform → Assert Expected Output
```

## ❓ Why is Functional Testing Important for Data?

1. **🛡️ Ensures Data Reliability**: Confirms that processed data meets business requirements
2. **🐛 Prevents Bugs**: Catches issues before they affect downstream systems
3. **📋 Validates Business Logic**: Ensures transformations correctly implement business rules
4. **📄 Documents System Behavior**: Test cases serve as living documentation

## 🔰 Basic Functional Testing Steps

### 1️⃣ Define Test Cases
Start by defining what aspects of your data processing you need to test:
- Does the system handle normal input correctly?
- How does it handle edge cases (empty data, maximum values, etc.)?
- Does it properly implement business rules?

### 2️⃣ Prepare Test Data
Create input data that covers the scenarios you want to test:
- Typical/expected data
- Edge cases
- Invalid data

### 3️⃣ Define Expected Outputs
Specify what results you expect for each test case:
- What should the output look like?
- What metadata should be generated?
- What errors should be raised?

### 4️⃣ Implement Tests
Write code that:
- Sets up the test environment
- Runs the function or process being tested
- Compares actual output to expected output
- Reports success or failure

### 5️⃣ Run Tests
Run your tests to verify that your system behaves as expected.

## 🧪 Basic Functional Testing Example

Here's a simple example testing a function that calculates total revenue:

```python
# Function to test
def calculate_total_revenue(transactions):
    """Calculate total revenue from a list of transaction amounts"""
    return sum(transaction['amount'] for transaction in transactions)

# Test function
def test_calculate_total_revenue():
    # 1. PREPARE TEST DATA
    test_transactions = [
        {'id': 1, 'amount': 100.0},
        {'id': 2, 'amount': 50.5},
        {'id': 3, 'amount': 25.0}
    ]
    
    # 2. DEFINE EXPECTED OUTPUT
    expected_total = 175.5
    
    # 3. RUN THE FUNCTION
    actual_total = calculate_total_revenue(test_transactions)
    
    # 4. ASSERT RESULT MATCHES EXPECTATION
    assert actual_total == expected_total, f"Expected {expected_total}, got {actual_total}"
    print("✅ Total revenue calculation test passed!")

# Run the test
test_calculate_total_revenue()
```

## 🔄 Testing Different Types of Data Transformations

### 📝 Text Data Transformation Test

```python
def clean_names(names_list):
    """Clean and standardize names"""
    return [name.strip().title() for name in names_list if name.strip()]

def test_clean_names():
    # Test data
    input_names = [
        "john doe", 
        "  JANE SMITH  ", 
        "robert BROWN", 
        ""  # Empty name
    ]
    
    # Expected output
    expected_names = [
        "John Doe", 
        "Jane Smith", 
        "Robert Brown"
    ]
    
    # Run function
    actual_names = clean_names(input_names)
    
    # Assert
    assert actual_names == expected_names, f"Expected {expected_names}, got {actual_names}"
    print("✅ Name cleaning test passed!")
```

### 🔢 Numerical Data Transformation Test

```python
def calculate_metrics(values):
    """Calculate various metrics for a list of numbers"""
    if not values:
        return {
            'count': 0,
            'sum': 0,
            'avg': None,
            'min': None,
            'max': None
        }
    
    return {
        'count': len(values),
        'sum': sum(values),
        'avg': sum(values) / len(values),
        'min': min(values),
        'max': max(values)
    }

def test_calculate_metrics():
    # Test cases
    test_cases = [
        {
            'input': [10, 20, 30, 40, 50],
            'expected': {
                'count': 5,
                'sum': 150,
                'avg': 30,
                'min': 10,
                'max': 50
            }
        },
        {
            'input': [],
            'expected': {
                'count': 0,
                'sum': 0,
                'avg': None,
                'min': None,
                'max': None
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        # Run function
        actual = calculate_metrics(test_case['input'])
        
        # Assert
        expected = test_case['expected']
        assert actual == expected, f"Test case {i}: Expected {expected}, got {actual}"
    
    print("✅ Metrics calculation tests passed!")
```

### 📅 Date Data Transformation Test

```python
from datetime import datetime, timedelta

def format_date_range(start_date, end_date):
    """Format a date range as a string"""
    if not start_date or not end_date:
        return "Invalid date range"
    
    if start_date > end_date:
        return "Invalid date range"
    
    if start_date == end_date:
        return start_date.strftime("%b %d, %Y")
    
    if start_date.year == end_date.year:
        if start_date.month == end_date.month:
            return f"{start_date.strftime('%b %d')} - {end_date.strftime('%d, %Y')}"
        else:
            return f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"
    else:
        return f"{start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}"

def test_format_date_range():
    # Test cases
    test_cases = [
        {
            'input': {
                'start_date': datetime(2023, 5, 10),
                'end_date': datetime(2023, 5, 15)
            },
            'expected': "May 10 - 15, 2023"
        },
        {
            'input': {
                'start_date': datetime(2023, 5, 10),
                'end_date': datetime(2023, 6, 15)
            },
            'expected': "May 10 - Jun 15, 2023"
        },
        {
            'input': {
                'start_date': datetime(2023, 5, 10),
                'end_date': datetime(2024, 6, 15)
            },
            'expected': "May 10, 2023 - Jun 15, 2024"
        },
        {
            'input': {
                'start_date': datetime(2023, 5, 10),
                'end_date': datetime(2023, 5, 10)
            },
            'expected': "May 10, 2023"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        # Run function
        actual = format_date_range(**test_case['input'])
        
        # Assert
        expected = test_case['expected']
        assert actual == expected, f"Test case {i}: Expected '{expected}', got '{actual}'"
    
    print("✅ Date formatting tests passed!")
```

## 🧰 Intermediate Functional Testing Techniques

### 🧱 Setting Up Test Fixtures

Test fixtures provide consistent test data:

```python
import pytest
import pandas as pd

@pytest.fixture
def sample_sales_data():
    """Provide sample sales data for testing"""
    return pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'product_id': [101, 102, 101],
        'quantity': [5, 3, 2],
        'price': [10.0, 15.0, 10.0]
    })

def test_revenue_calculation(sample_sales_data):
    """Test that revenue calculation works correctly"""
    # Calculate revenue (quantity * price)
    sample_sales_data['revenue'] = sample_sales_data['quantity'] * sample_sales_data['price']
    
    # Expected total revenue
    expected_total_revenue = 5*10.0 + 3*15.0 + 2*10.0
    
    # Actual total revenue
    actual_total_revenue = sample_sales_data['revenue'].sum()
    
    # Assert they match
    assert actual_total_revenue == expected_total_revenue
    print(f"✅ Revenue calculation test passed! Total: {actual_total_revenue}")
```

### 🔄 Testing DataFrame Transformations

```python
def clean_customer_data(df):
    """Clean customer data by standardizing names and removing duplicates"""
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Standardize names (lowercase, then capitalize)
    cleaned_df['name'] = cleaned_df['name'].str.lower().str.title()
    
    # Remove duplicates based on customer_id
    cleaned_df = cleaned_df.drop_duplicates(subset=['customer_id'])
    
    return cleaned_df

def test_clean_customer_data():
    """Test that the customer data cleaning function works as expected"""
    # Sample input
    input_df = pd.DataFrame({
        'customer_id': [1001, 1002, 1001],  # Note: duplicate ID
        'name': ['JOHN DOE', 'jane smith', 'John Doe']
    })
    
    # Expected output
    expected_df = pd.DataFrame({
        'customer_id': [1001, 1002],
        'name': ['John Doe', 'Jane Smith']
    }).reset_index(drop=True)
    
    # Run transformation
    result_df = clean_customer_data(input_df).reset_index(drop=True)
    
    # Check if dataframes are equal
    pd.testing.assert_frame_equal(result_df, expected_df)
    print("✅ Customer data cleaning test passed!")
```

### 📊 Testing Aggregation Functions

```python
def aggregate_sales_by_month(sales_df):
    """Aggregate sales data by month"""
    # Convert date column to datetime
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    
    # Extract month
    sales_df['month'] = sales_df['date'].dt.strftime('%Y-%m')
    
    # Group by month and aggregate
    monthly_sales = sales_df.groupby('month').agg(
        total_sales=('quantity', 'sum'),
        total_revenue=('revenue', 'sum'),
        average_price=('price', 'mean')
    ).reset_index()
    
    return monthly_sales

def test_aggregate_sales_by_month():
    """Test monthly sales aggregation"""
    # Create test data
    sales_data = pd.DataFrame({
        'date': ['2023-01-15', '2023-01-20', '2023-02-10'],
        'product_id': [101, 102, 101],
        'quantity': [5, 3, 2],
        'price': [10.0, 15.0, 10.0],
        'revenue': [50.0, 45.0, 20.0]
    })
    
    # Run aggregation
    monthly_sales = aggregate_sales_by_month(sales_data)
    
    # Expected results
    expected_months = ['2023-01', '2023-02']
    expected_total_sales = [8, 2]
    expected_total_revenue = [95.0, 20.0]
    
    # Assertions
    assert len(monthly_sales) == 2, f"Expected 2 months, got {len(monthly_sales)}"
    assert monthly_sales['month'].tolist() == expected_months, "Month values don't match"
    assert monthly_sales['total_sales'].tolist() == expected_total_sales, "Total sales don't match"
    assert monthly_sales['total_revenue'].tolist() == expected_total_revenue, "Total revenue doesn't match"
    
    print("✅ Monthly sales aggregation test passed!")
```

## 🚀 Advanced Functional Testing Techniques

### 1️⃣ Parameterized Testing

Test multiple scenarios at once:

```python
import pytest

# Function to test
def categorize_customer(annual_spend):
    """Categorize customers based on annual spend"""
    if annual_spend >= 10000:
        return "Premium"
    elif annual_spend >= 5000:
        return "Gold"
    elif annual_spend >= 1000:
        return "Silver"
    else:
        return "Bronze"

# Parameterized test
@pytest.mark.parametrize("spend,expected_category", [
    (500, "Bronze"),
    (1000, "Silver"),
    (4999, "Silver"),
    (5000, "Gold"),
    (9999, "Gold"),
    (10000, "Premium"),
    (15000, "Premium")
])
def test_customer_categorization(spend, expected_category):
    """Test customer categorization works for different spending levels"""
    actual_category = categorize_customer(spend)
    assert actual_category == expected_category, f"Expected {expected_category}, got {actual_category}"
```

### 2️⃣ Property-Based Testing

Test properties that should always hold true:

```python
from hypothesis import given
import hypothesis.strategies as st

# Function to test
def process_transaction(transaction):
    """Process a transaction by adding fees and taxes"""
    amount = transaction['amount']
    # Add 5% service fee
    fee = amount * 0.05
    # Add 8% tax on total
    subtotal = amount + fee
    tax = subtotal * 0.08
    total = subtotal + tax
    
    return {
        'original_amount': amount,
        'fee': fee,
        'subtotal': subtotal,
        'tax': tax,
        'total': total
    }

# Property-based test
@given(st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False))
def test_transaction_processing_properties(amount):
    """Test that transaction processing maintains key properties"""
    transaction = {'amount': amount}
    result = process_transaction(transaction)
    
    # Property 1: Total should be greater than original amount
    assert result['total'] > amount, "Total should be greater than original amount"
    
    # Property 2: Fees and taxes should be non-negative
    assert result['fee'] >= 0, "Fee should be non-negative"
    assert result['tax'] >= 0, "Tax should be non-negative"
    
    # Property 3: Total should equal original + fee + tax
    expected_total = amount + result['fee'] + result['tax']
    assert abs(result['total'] - expected_total) < 0.001, "Total doesn't match components"
```

### 3️⃣ End-to-End Testing for ETL Pipelines

Test the entire data pipeline:

```python
def test_customer_etl_pipeline():
    """Test the entire customer data ETL pipeline"""
    import os
    import pandas as pd
    
    # 1. SETUP
    # Create test input files
    with open('test_input.csv', 'w') as f:
        f.write("customer_id,name,email,spend\n")
        f.write("1001,JOHN DOE,john@example.com,5240.50\n")
        f.write("1002,Jane Smith,jane@example.com,1050.75\n")
        f.write("1001,John Doe,john@example.com,2500.00\n")  # Duplicate
    
    # 2. EXECUTE
    # Run the ETL pipeline
    from my_etl_module import run_customer_etl
    success = run_customer_etl(
        input_path='test_input.csv',
        output_path='test_output.csv',
        error_path='test_errors.csv'
    )
    
    # 3. VERIFY
    # Check that the pipeline ran successfully
    assert success, "ETL pipeline failed to complete"
    
    # Check the output file has correct data
    output_df = pd.read_csv('test_output.csv')
    
    # Verify record count (should be 2 after deduplication)
    assert len(output_df) == 2, f"Expected 2 records, got {len(output_df)}"
    
    # Verify data cleaning was applied
    assert 'John Doe' in output_df['name'].values, "Name standardization failed"
    
    # Verify customer categorization was applied
    assert 'Gold' in output_df['category'].values, "Customer categorization failed"
    
    # 4. CLEANUP
    for file in ['test_input.csv', 'test_output.csv', 'test_errors.csv']:
        if os.path.exists(file):
            os.remove(file)
            
    print("✅ End-to-end ETL pipeline test passed!")
```

## 🧪 Testing Different Data Sources

### 📊 Testing CSV Loading and Processing

```python
def test_csv_processing():
    """Test loading and processing a CSV file"""
    import pandas as pd
    import os
    
    # 1. Create test CSV
    test_csv = "test_data.csv"
    with open(test_csv, 'w') as f:
        f.write("id,name,value\n")
        f.write("1,Product A,100\n")
        f.write("2,Product B,200\n")
        f.write("3,Product C,300\n")
    
    # 2. Load CSV
    df = pd.read_csv(test_csv)
    
    # 3. Run tests
    # Check row count
    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
    
    # Check column names
    expected_columns = ['id', 'name', 'value']
    assert list(df.columns) == expected_columns, f"Column mismatch: {list(df.columns)}"
    
    # Check data types
    assert df['id'].dtype.kind in 'iu', "ID column should be numeric"
    assert df['name'].dtype == 'object', "Name column should be string/object"
    assert df['value'].dtype.kind in 'if', "Value column should be numeric"
    
    # Test specific values
    assert df.loc[1, 'name'] == 'Product B', f"Expected 'Product B', got {df.loc[1, 'name']}"
    assert df['value'].sum() == 600, f"Sum should be 600, got {df['value'].sum()}"
    
    # 4. Cleanup
    os.remove(test_csv)
    
    print("✅ CSV processing test passed!")
```

### 📑 Testing Excel File Handling

```python
def test_excel_processing():
    """Test loading and processing an Excel file"""
    import pandas as pd
    import os
    import numpy as np
    
    # Skip if openpyxl not installed
    try:
        import openpyxl
    except ImportError:
        print("⚠️ Skipping Excel test as openpyxl is not installed")
        return
    
    # 1. Create test Excel file
    test_excel = "test_data.xlsx"
    
    # Create DataFrames for multiple sheets
    sheet1_df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Product A', 'Product B', 'Product C'],
        'value': [100, 200, 300]
    })
    
    sheet2_df = pd.DataFrame({
        'category': ['Cat1', 'Cat2'],
        'description': ['Category 1', 'Category 2']
    })
    
    # Write to Excel
    with pd.ExcelWriter(test_excel) as writer:
        sheet1_df.to_excel(writer, sheet_name='Products', index=False)
        sheet2_df.to_excel(writer, sheet_name='Categories', index=False)
    
    # 2. Load Excel
    products_df = pd.read_excel(test_excel, sheet_name='Products')
    categories_df = pd.read_excel(test_excel, sheet_name='Categories')
    
    # 3. Run tests
    # Test Products sheet
    assert len(products_df) == 3, f"Expected 3 product rows, got {len(products_df)}"
    assert products_df['value'].sum() == 600, f"Product values sum should be 600"
    
    # Test Categories sheet
    assert len(categories_df) == 2, f"Expected 2 category rows, got {len(categories_df)}"
    assert 'Cat1' in categories_df['category'].values, "Expected 'Cat1' in categories"
    
    # 4. Cleanup
    os.remove(test_excel)
    
    print("✅ Excel processing test passed!")
```

### 🗄️ Testing Database Operations

```python
def test_database_operations():
    """Test database operations with SQLite"""
    import pandas as pd
    import sqlite3
    import os
    
    # 1. Create test database
    db_file = 'test_db.sqlite'
    conn = sqlite3.connect(db_file)
    
    # Create test tables
    conn.execute('''
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT
    )
    ''')
    
    conn.execute('''
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        amount REAL,
        FOREIGN KEY (customer_id) REFERENCES customers (id)
    )
    ''')
    
    # Insert test data
    conn.execute("INSERT INTO customers VALUES (1, 'John Doe', 'john@example.com')")
    conn.execute("INSERT INTO customers VALUES (2, 'Jane Smith', 'jane@example.com')")
    
    conn.execute("INSERT INTO orders VALUES (101, 1, 99.99)")
    conn.execute("INSERT INTO orders VALUES (102, 1, 149.99)")
    conn.execute("INSERT INTO orders VALUES (103, 2, 199.99)")
    
    conn.commit()
    
    # 2. Test SQL query execution
    def get_customer_orders():
        """Function to test: Get all customers with their orders"""
        query = '''
        SELECT 
            c.id AS customer_id,
            c.name AS customer_name,
            COUNT(o.id) AS order_count,
            SUM(o.amount) AS total_spent
        FROM customers c
        LEFT JOIN orders o ON c.id = o.customer_id
        GROUP BY c.id, c.name
        ORDER BY c.id
        '''
        
        return pd.read_sql(query, conn)
    
    # Execute the function
    result_df = get_customer_orders()
    
    # 3. Run tests
    # Check row count
    assert len(result_df) == 2, f"Expected 2 rows, got {len(result_df)}"
    
    # Check column names
    expected_columns = ['customer_id', 'customer_name', 'order_count', 'total_spent']
    assert all(col in result_df.columns for col in expected_columns), "Missing expected columns"
    
    # Check specific values
    assert result_df.loc[0, 'customer_name'] == 'John Doe', "Wrong customer name"
    assert result_df.loc[0, 'order_count'] == 2, "John should have 2 orders"
    assert abs(result_df.loc[0, 'total_spent'] - 249.98) < 0.01, "Wrong total for John"
    
    assert result_df.loc[1, 'customer_name'] == 'Jane Smith', "Wrong customer name"
    assert result_df.loc[1, 'order_count'] == 1, "Jane should have 1 order"
    assert abs(result_df.loc[1, 'total_spent'] - 199.99) < 0.01, "Wrong total for Jane"
    
    # 4. Cleanup
    conn.close()
    os.remove(db_file)
    
    print("✅ Database operations test passed!")
```

### 🌐 Testing JSON Data Processing

```python
def test_json_processing():
    """Test loading and processing JSON data"""
    import json
    import pandas as pd
    import os
    
    # 1. Create test JSON file
    test_json = "test_data.json"
    
    # Test data
    test_data = {
        "users": [
            {
                "id": 1,
                "name": "John Doe",
                "email": "john@example.com",
                "orders": [
                    {"id": 101, "amount": 99.99},
                    {"id": 102, "amount": 149.99}
                ]
            },
            {
                "id": 2,
                "name": "Jane Smith",
                "email": "jane@example.com",
                "orders": [
                    {"id": 103, "amount": 199.99}
                ]
            }
        ]
    }
    
    # Write to JSON file
    with open(test_json, 'w') as f:
        json.dump(test_data, f)
    
    # 2. Function to test: Calculate total order amount per user
    def calculate_user_totals(json_file):
        """Calculate total order amount for each user from JSON data"""
        # Load JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Process data
        user_totals = []
        for user in data.get('users', []):
            # Sum order amounts
            total_amount = sum(order['amount'] for order in user.get('orders', []))
            
            user_totals.append({
                'user_id': user['id'],
                'name': user['name'],
                'order_count': len(user.get('orders', [])),
                'total_amount': total_amount
            })
        
        # Return as DataFrame
        return pd.DataFrame(user_totals)
    
    # 3. Run function
    result_df = calculate_user_totals(test_json)
    
    # 4. Test results
    # Check row count
    assert len(result_df) == 2, f"Expected 2 rows, got {len(result_df)}"
    
    # Check user 1 (John Doe)
    john_row = result_df[result_df['user_id'] == 1].iloc[0]
    assert john_row['name'] == 'John Doe', "Wrong name for user 1"
    assert john_row['order_count'] == 2, "User 1 should have 2 orders"
    assert abs(john_row['total_amount'] - 249.98) < 0.01, "Wrong total amount for user 1"
    
    # Check user 2 (Jane Smith)
    jane_row = result_df[result_df['user_id'] == 2].iloc[0]
    assert jane_row['name'] == 'Jane Smith', "Wrong name for user 2"
    assert jane_row['order_count'] == 1, "User 2 should have 1 order"
    assert abs(jane_row['total_amount'] - 199.99) < 0.01, "Wrong total amount for user 2"
    
    # 5. Cleanup
    os.remove(test_json)
    
    print("✅ JSON processing test passed!")
```

## 🛠️ Best Practices for Data Functional Testing

1. **🎯 Test Business Requirements**: Focus on testing that business rules and requirements are correctly implemented.

2. **📝 Document Test Cases**: Clearly document what each test is checking and why.

3. **🧩 Use Realistic Test Data**: Test with data that represents real-world scenarios.

4. **⚙️ Isolate Test Environments**: Keep testing separate from production.

5. **✅ Test Both Happy Paths and Edge Cases**:
   - Valid, expected inputs
   - Missing or null values
   - Boundary values
   - Malformed inputs

6. **🔄 Automate Where Possible**: Add tests to CI/CD pipelines.

7. **📊 Monitor Test Coverage**: Ensure critical data paths are covered.

8. **🔍 Maintain Test Data Independence**: Tests should not depend on each other.

9. **⏱️ Include Performance Considerations**: Test with various data volumes.

10. **🧹 Clean Up Test Resources**: Always clean up temporary files and test databases.

## 📁 Implementing a Testing Framework

Here's a recommended structure for organizing data functional tests:

```
data_project/
├── src/
│   ├── etl/
│   │   ├── extract.py
│   │   ├── transform.py
│   │   └── load.py
│   └── main.py
├── tests/
│   ├── functional/
│   │   ├── test_extract.py
│   │   ├── test_transform.py
│   │   ├── test_load.py
│   │   └── test_end_to_end.py
│   ├── fixtures/
│   │   ├── sample_data.csv
│   │   └── expected_results.csv
│   └── conftest.py
└── pytest.ini
```

## 🧰 Tools for Data Functional Testing

1. **🔍 pytest**: General-purpose testing framework
   ```bash
   pip install pytest
   ```

2. **✅ Great Expectations**: Data validation framework
   ```bash
   pip install great-expectations
   ```

3. **🔄 Airflow**: For testing workflow DAGs
   ```bash
   pip install apache-airflow
   ```

4. **🔢 dbt**: For testing data transformations in SQL
   ```bash
   pip install dbt-core
   ```

5. **📊 pandas.testing**: For DataFrame comparisons

6. **⚡ pytest-spark**: For testing Spark applications
   ```bash
   pip install pytest-spark
   ```

## 📝 Example Testing Project

Here's a complete example of a small data testing project:

```python
# src/data_transformer.py
class DataTransformer:
    def clean_customer_data(self, df):
        """Clean customer data by standardizing names and removing duplicates"""
        cleaned_df = df.copy()
        
        # Handle missing values
        cleaned_df['name'] = cleaned_df['name'].fillna('')
        cleaned_df['email'] = cleaned_df['email'].fillna('')
        
        # Standardize names (lowercase, then title case)
        cleaned_df['name'] = cleaned_df['name'].str.lower().str.title()
        
        # Standardize emails (lowercase)
        cleaned_df['email'] = cleaned_df['email'].str.lower()
        
        # Remove duplicates based on customer_id
        cleaned_df = cleaned_df.drop_duplicates(subset=['customer_id'])
        
        return cleaned_df
    
    def categorize_customers(self, df):
        """Categorize customers based on total spend"""
        categorized_df = df.copy()
        
        # Define categorization function
        def get_category(spend):
            if pd.isna(spend):
                return 'Unknown'
            if spend >= 10000:
                return 'Premium'
            elif spend >= 5000:
                return 'Gold'
            elif spend >= 1000:
                return 'Silver'
            else:
                return 'Bronze'
        
        # Apply categorization
        categorized_df['category'] = categorized_df['total_spend'].apply(get_category)
        
        return categorized_df
```

```python
# tests/test_data_transformer.py
import pandas as pd
import pytest
from src.data_transformer import DataTransformer

@pytest.fixture
def sample_customer_data():
    """Fixture providing sample customer data"""
    return pd.DataFrame({
        'customer_id': [1001, 1002, 1001, 1003],
        'name': ['JOHN DOE', 'jane smith', 'John Doe', None],
        'email': ['JOHN@EXAMPLE.COM', 'jane@example.com', 'john@example.com', 'sam@example.com'],
        'total_spend': [5500, 750, 2000, None]
    })

def test_clean_customer_data(sample_customer_data):
    """Test customer data cleaning functionality"""
    # Initialize transformer
    transformer = DataTransformer()
    
    # Apply transformation
    cleaned_df = transformer.clean_customer_data(sample_customer_data)
    
    # Assertions
    # 1. Check duplicate removal (should have 3 rows, not 4)
    assert len(cleaned_df) == 3, f"Expected 3 rows after deduplication, got {len(cleaned_df)}"
    
    # 2. Check name standardization
    assert cleaned_df.loc[cleaned_df['customer_id'] == 1002, 'name'].iloc[0] == 'Jane Smith', "Name standardization failed"
    
    # 3. Check email standardization
    assert cleaned_df.loc[cleaned_df['customer_id'] == 1001, 'email'].iloc[0] == 'john@example.com', "Email standardization failed"
    
    # 4. Check null handling
    assert cleaned_df.loc[cleaned_df['customer_id'] == 1003, 'name'].iloc[0] == '', "Null name handling failed"
    
    print("✅ Customer data cleaning test passed!")

def test_customer_categorization(sample_customer_data):
    """Test customer categorization functionality"""
    # Initialize transformer
    transformer = DataTransformer()
    
    # Apply transformation
    categorized_df = transformer.categorize_customers(sample_customer_data)
    
    # Assertions
    # 1. Check all rows have a category
    assert 'category' in categorized_df.columns, "Category column not created"
    assert not categorized_df['category'].isnull().any(), "Some categories are null"
    
    # 2. Check specific categorizations
    customer_1001 = categorized_df.loc[categorized_df['customer_id'] == 1001].iloc[0]
    assert customer_1001['category'] == 'Gold', f"Customer 1001 should be Gold, got {customer_1001['category']}"
    
    customer_1002 = categorized_df.loc[categorized_df['customer_id'] == 1002].iloc[0]
    assert customer_1002['category'] == 'Bronze', f"Customer 1002 should be Bronze, got {customer_1002['category']}"
    
    customer_1003 = categorized_df.loc[categorized_df['customer_id'] == 1003].iloc[0]
    assert customer_1003['category'] == 'Unknown', f"Customer 1003 should be Unknown, got {customer_1003['category']}"
    
    print("✅ Customer categorization test passed!")
```

## 🎯 Summary

Functional testing is essential for ensuring data reliability and system correctness. By implementing a comprehensive testing approach, you can:

- ✅ Catch issues early before they affect downstream systems
- 📝 Document how your data systems should behave
- 💯 Build confidence in your data pipelines
- 🔄 Make changes with confidence knowing tests will catch regressions

Start with simple assertions and grow to more sophisticated test cases to create a solid foundation for data quality and system reliability.

Remember that testing is an investment that pays off by reducing bugs, improving maintainability, and making it easier to enhance your data systems over time.
