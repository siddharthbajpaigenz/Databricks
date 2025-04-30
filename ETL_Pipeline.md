# Automated ETL/Data Testing Pipeline in Azure  
![Azure Pipeline Logo](https://dummyimage.com/1200x400/0078d4/ffffff&text=Automate+Data+Testing+in+Azure)  

*A beginner-friendly guide to automate data validation, testing, and error reporting using Microsoft Azure.*  

---

## Table of Contents  
1. [Why Automate?](#why-automate)  
2. [Tools Needed](#tools)  
3. [Step-by-Step Setup](#setup)  
   - 3.1 [Create a Data Factory Pipeline](#adf)  
   - 3.2 [Write Data Tests](#tests)  
   - 3.3 [Auto-Create Error Tickets](#tickets)  
   - 3.4 [Track Who Changed Data](#tracking)  
4. [Benefits](#benefits)  
5. [Troubleshooting](#troubleshoot)  
6. [Next Steps](#next)  

---

## 1. Why Automate? <a name="why-automate"></a>  
- 🕒 **Save Time**: No more manual checks at midnight!  
- 🚫 **Avoid Mistakes**: Robots find errors humans miss.  
- 📋 **Auto-Reports**: Errors are logged as tickets automatically.  

**Old Process**:  
1. Wait for data to load at 11 PM.  
2. Manually check 100+ tables.  
3. Create error tickets one by one.  
4. Work late to finish.  

**New Process**:  
1. Robots test data at 11 PM.  
2. Error tickets auto-created.  
3. Team sleeps peacefully!  

---

## 2. Tools Needed <a name="tools"></a>  
1. **Azure Account** ([Free Trial](https://azure.microsoft.com/))  
2. **Azure Data Factory (ADF)**: To move data.  
3. **Azure Databricks**: To clean/test data.  
4. **Trackspace**: To track errors (like Jira).  

---

## 3. Step-by-Step Setup <a name="setup"></a>  

### 3.1 Create a Data Factory Pipeline <a name="adf"></a>  
**ADF is your robot scheduler.**  

1. **Create ADF**:  
   - Go to [Azure Portal](https://portal.azure.com/) → **Create Resource** → **Data Factory**.  
   - Name: `MyDataRobot`.  
   - Region: Pick one closest to you.  

2. **Build a Pipeline**:  
   - Open ADF → Click **+** → **Pipeline**.  
   - Add a **Copy Data** activity to move raw data to Azure Data Lake.  

3. **Schedule It**:  
   - Click **Trigger** → **New/Edit** → Set time to 11 PM daily.  

![ADF Pipeline](https://dummyimage.com/800x200/555/fff&text=ADF+Daily+Schedule)  

---

### 3.2 Write Data Tests <a name="tests"></a>  
**Two types of tests to find errors:**  

#### **Test 1: Generic Tests (For All Tables)**  
```python  
# File: generic_tests.py  

def test_no_empty_tables(df):  
    assert df.count() > 0, "ERROR: Table is empty!"  

def test_no_missing_emails(df):  
    missing = df.filter(df["email"].isNull()).count()  
    assert missing == 0, f"ERROR: {missing} emails missing!"  
```  

#### **Test 2: Custom Tests (Example: Loyalty Points)**  
```python  
# File: loyalty_tests.py  

def test_valid_loyalty_points(df):  
    negative_points = df.filter(df["points"] < 0).count()  
    assert negative_points == 0, "ERROR: Negative points found!"  
```  

**Run Tests in Databricks**:  
1. In ADF, add a **Databricks Notebook** activity.  
2. Link it to your test files.  

---

### 3.3 Auto-Create Error Tickets <a name="tickets"></a>  
**When a test fails, create a ticket in Trackspace.**  

1. **Build an Error Report**:  
```python  
# File: defect_report.py  

report = {  
    "Test Name": "test_no_missing_emails",  
    "Error": "500 emails missing!",  
    "Table": "Customers",  
    "Time": "2023-10-01 23:05:00"  
}  
```  

2. **Send to Trackspace**:  
```python  
# File: send_to_trackspace.py  
import requests  

def create_ticket(report):  
    api_key = "your-key-here"  
    response = requests.post(  
        "https://trackspace.com/api/tickets",  
        json=report,  
        headers={"Authorization": f"Bearer {api_key}"}  
    )  
    print("Ticket ID:", response.json()["id"])  
```  

---

### 3.4 Track Who Changed Data <a name="tracking"></a>  
**Know who last updated a table.**  

1. **Add a "Last Updated By" Column**:  
```sql  
ALTER TABLE Customers  
ADD LastUpdatedBy VARCHAR(50);  
```  

2. **Update Automatically**:  
```sql  
UPDATE Customers  
SET LastUpdatedBy = 'alice@company.com'  
WHERE customer_id = 1001;  
```  

3. **Include in Error Reports**:  
```python  
report["Last Updated By"] = "alice@company.com"  
```  

---

## 4. Benefits <a name="benefits"></a>  
- ✅ **No Manual Work**: Robots test 100+ tables in minutes.  
- 📩 **Instant Alerts**: Get error tickets with details.  
- 👤 **Accountability**: See who caused issues.  
- 📅 **No Overtime**: Tests run at 11 PM automatically.  

---

## 5. Troubleshooting <a name="troubleshoot"></a>  

| Problem                  | Solution                                  |  
|--------------------------|-------------------------------------------|  
| Tests not running         | Check if Databricks cluster is **ON**.    |  
| Tickets not created       | Verify Trackspace API key permissions.    |  
| "Last Updated By" missing | Add the column to your SQL table.         |  

---

## 6. Next Steps <a name="next"></a>  
1. **Start Small**: Test one table first.  
2. **Customize**: Add tests for your data rules.  
3. **Learn More**: [Microsoft Azure Docs](https://learn.microsoft.com/azure/data-factory/)  

--- 

**Happy Automating!** 🚀  
``` 

---

**How to Use This Guide**:  
1. Replace placeholder values (e.g., `your-key-here`).  
2. Add your own test cases.  
3. Customize the Trackspace API endpoint.  

This README covers everything from setup to error tracking in simple English. For advanced features, refer to the [Azure documentation](https://learn.microsoft.com).
