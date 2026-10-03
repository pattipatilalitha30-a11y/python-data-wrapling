# Customer Sales Data Cleaning and Analysis using Python

**Week 3: Python Basics, Pandas Data Wrangling, and Data Visualization**  
**Author:** Lalitha Patipatti  

---

## 📌 Project Overview & Objective

In real-world data science, raw business data is almost never clean. It often contains missing entries, duplicate transactions, inconsistent naming conventions, malformed date formats, and incorrect data types.

The objective of this project is to build an end-to-end Python and Pandas data wrangling pipeline that:
1. Ingests a realistic, messy e-commerce customer sales dataset.
2. Identifies data anomalies, missing values, duplicates, and type mismatches.
3. Cleans, validates, transforms, and enriches the data using Pandas.
4. Performs business exploratory data analysis (EDA) to calculate key performance indicators (KPIs).
5. Generates high-resolution, publication-ready visualizations using Matplotlib and Seaborn.
6. Exports cleaned datasets, analysis reports, and charts for stakeholders.

---

## 🎯 Week 3 Requirements Covered

This project demonstrates all Week 3 competencies:

### 1. Python Basics
- **Data Structures:** Extensive use of Python lists, dictionaries, tuples, and sets for configuration, aggregations, and metric tracking.
- **Modular Functions:** Clean, reusable functions with comprehensive docstrings and type annotations (`typing.Dict`, `typing.Tuple`, `typing.Any`).
- **Python Scripting:** Professional command-line execution structure using `if __name__ == "__main__": main()`.

### 2. Pandas Operations
- **Data Ingestion:** `pd.read_csv()` to load raw sales transactions and supplementary category reference tables.
- **Dataset Inspection:** `df.shape`, `df.info()`, `df.describe()`, `df.dtypes`, and `df.isna().sum()`.
- **Handling Missing Values:** Targeted imputation using `.fillna()`, median imputation for numerical features (`Age`, `Rating`), and mode imputation for categorical attributes (`City`, `Payment_Method`).
- **Duplicate Removal:** `df.duplicated().sum()` detection and `df.drop_duplicates()` removal.
- **Row Filtering & Validation:** Boolean masking to eliminate corrupted records (negative/zero quantities, negative prices, out-of-bounds ratings).
- **Type Conversion:** Converting currency strings (`$`) to `float64`, integer-safe columns to `int64`, and parsing heterogeneous date strings using `pd.to_datetime(format='mixed', errors='coerce')`.
- **Text Standardization:** Removing whitespace using `.str.strip()`, normalizing casing with `.str.title()`, and dictionary mapping for categorical integrity (`Gender`, `Payment_Method`, `Category`).
- **Feature Engineering:** Creating derived columns:
  - `Total_Sales = Quantity * Unit_Price`
  - `Customer_Segment` (`High`, `Medium`, `Low` tiers)
  - `Order_Month`, `Order_Year`, and `Day_of_Week` from `Order_Date`
- **Merge & Combine:** `pd.merge(df, category_metadata, on="Category", how="left")` to enrich transactions with departmental metadata, target profit margins, and calculate `Estimated_Profit`.

### 3. Data Visualization
- **Matplotlib & Seaborn:** Polished figure layouts, custom color palettes (`Blues_r`, `viridis`, `magma`, `crest`), high DPI rendering (300 DPI), annotated data values, and formatted axis tick labels.

---

## 🗂 Project Structure

```text
week3-python-data-wrangling/
│
├── data/
│   ├── messy_sales_data.csv       # Raw, messy dataset (125 records with duplicates & missing values)
│   ├── category_metadata.csv      # Category reference table for data merging demonstration
│   └── cleaned_sales_data.csv     # Final cleaned, validated, and enriched dataset (105 records)
│
├── results/
│   ├── sales_by_category.png      # Chart 1: Revenue breakdown across product categories
│   ├── top_products.png           # Chart 2: Top 10 products by sales revenue
│   ├── monthly_sales.png          # Chart 3: Monthly revenue trend line chart (2023)
│   ├── sales_by_city.png          # Chart 4: Sales distribution across major metropolitan cities
│   ├── rating_distribution.png    # Chart 5: Customer satisfaction rating frequency
│   └── analysis_results.txt       # Comprehensive text report of all calculated KPIs
│
├── data_wrangling.py              # Main Python script orchestrating the wrangling and analysis
├── generate_messy_data.py         # Deterministic generator for the realistic messy dataset
├── requirements.txt               # Project library dependencies
└── README.md                      # Project documentation and findings
```

---

## 📊 Dataset Description

The project utilizes an e-commerce sales dataset containing customer transaction records:

| Column Name | Raw Characteristics & Issues | Cleaned Characteristics |
| :--- | :--- | :--- |
| `Customer_ID` | Inconsistent casing (`cust004`), trailing whitespace, missing entries | Uppercase standardized (`CUST004`), verified non-null |
| `Customer_Name` | All-caps names (`MARGARET JACKSON`), leading spaces, blanks | Title-cased (`Margaret Jackson`), missing imputed with "Valued Customer" |
| `Age` | String values, negative values (`-5`), extreme outliers (`145`), nulls | Integer values strictly bounded between 18–90; missing imputed with median (44) |
| `Gender` | Inconsistent entries (`M`, `male`, `F`, `female`, empty) | Standardized to `Male`, `Female`, `Other`, or `Not Specified` |
| `City` | Inconsistent casing (`san francisco`, `NEW YORK `), nulls | Title-cased and whitespace stripped; missing imputed with mode |
| `Product` | Leading/trailing whitespace | Cleaned, title-cased product names |
| `Category` | Mixed casing (`ELECTRONICS`, `home & kitchen`) | Unified into 5 canonical retail categories |
| `Quantity` | Float strings (`2.0`), negative values (`-1`), zero values, nulls | Cleaned `int64` strictly greater than 0 |
| `Unit_Price` | Currency symbols (`$45.50`), comma separators, negative values | Cleaned `float64` strictly greater than 0.00 |
| `Order_Date` | Mixed formats (`YYYY-MM-DD`, `MM/DD/YYYY`, `YYYY.MM.DD`, `DD-MM-YYYY`, invalid) | Normalized `datetime64[ns]` timestamp |
| `Payment_Method` | Inconsistent naming (`CREDIT_CARD`, `COD`, `cash`, `PAYPAL`) | Mapped to `Credit Card`, `Debit Card`, `PayPal`, or `Cash on Delivery` |
| `Rating` | Out-of-bounds ratings (`-1.0`, `6.0`), strings, missing entries | Validated `float64` between 1.0–5.0; missing imputed with median (4.0) |
| `Sales` | Unreliable / pre-calculated strings with missing values | Replaced by engineered `Total_Sales` (`Quantity * Unit_Price`) |

---

## 🧹 Data Cleaning Pipeline

The data wrangling process in `data_wrangling.py` follows seven sequential stages:

1. **Ingestion & Initial Inspection:**
   - Raw records: **125**
   - Initial missing values: **48 cells (2.95%)** across 10 columns.
   - Initial exact duplicates: **10 rows**.
2. **Deduplication:**
   - Exact duplicate rows are detected and pruned via `df.drop_duplicates()`.
3. **Text Standardization:**
   - Whitespace stripping and proper casing across `Customer_ID`, `Customer_Name`, `City`, `Product`, `Category`, `Gender`, and `Payment_Method`.
4. **Data Type Casting:**
   - Currency symbols (`$`) and formatting artifacts removed from `Unit_Price`.
   - String quantities converted to numeric.
   - Mixed-format date strings converted via `pd.to_datetime(..., format="mixed")`.
5. **Validation & Filtering:**
   - Corrupted rows with unparseable dates, missing customer identifiers, or non-positive quantities and prices were removed (**10 invalid rows filtered**).
   - Outlier ages (< 18 or > 90) and ratings (< 1.0 or > 5.0) were imputed with robust sample medians.
6. **Feature Engineering:**
   - `Total_Sales`: `Quantity * Unit_Price` rounded to 2 decimal places.
   - `Customer_Segment`: Segmented into `High` ($\ge \$250$), `Medium` ($\$100 - \$249.99$), and `Low` ($< \$100$).
   - `Order_Month` (`YYYY-MM`), `Order_Year`, and `Day_of_Week`.
7. **Relational Enrichment (Merge):**
   - Merged with `category_metadata.csv` to attach `Department`, `Target_Profit_Margin`, and `Category_Manager`.
   - Computed `Estimated_Profit` based on department profit margins.

**Final Cleaned Dataset Size:** **105 validated records** across 21 structured columns with **0 duplicates** and **0 missing values**.

---

## 📈 Visualizations Created

All charts are saved at 300 DPI in the `results/` folder:

| Chart File | Visualization Type | Key Business Question Answered |
| :--- | :--- | :--- |
| `results/sales_by_category.png` | Bar Chart (Sequential Blues) | Which retail department drives the greatest gross revenue? |
| `results/top_products.png` | Horizontal Bar Chart (`viridis`) | Which individual products represent our top 10 revenue drivers? |
| `results/monthly_sales.png` | Line Chart with Area Fill | How did monthly revenue fluctuate across calendar year 2023? |
| `results/sales_by_city.png` | Bar Chart (`magma`) | Which metropolitan regions generate the highest sales volume? |
| `results/rating_distribution.png` | Frequency Distribution Bar Chart (`crest`) | How satisfied are customers with their purchases? |

---

## 💡 Key Business Findings & Insights

Based on the automated exploratory analysis of the 105 cleaned transactions:

1. **Overall Performance:**
   - **Total Sales Revenue:** **\$25,068.07**
   - **Average Order Value (AOV):** **\$238.74** (Median: \$165.00)
   - **Total Units Sold:** **304 units**
   - **Unique Customers:** **42 customers**
   - **Average Customer Rating:** **4.04 / 5.00**

2. **Top Product Category:**
   - **Electronics** is the leading category by a wide margin, generating **\$9,650.61** across 23 orders (Average: \$419.59/order).
   - Followed by **Fashion & Apparel** (\$4,727.00) and **Office Supplies** (\$3,877.96).

3. **Top Selling Products:**
   - **Tablet** is the #1 revenue-generating product (\$5,939.82, 18 units sold).
   - **Desk Chair** ranks second (\$3,420.00, 19 units sold).
   - **Running Shoes** ranks third (\$2,090.00, 19 units sold).

4. **Geographical Performance:**
   - **New York** is the highest grossing city with **\$7,300.90** in sales (24 orders).
   - **Miami** ranked second with **\$4,982.41** (17 orders), followed by **San Francisco** with **\$4,163.43**.

5. **Payment Method Breakdown:**
   - Digital wallets and delivery payment methods lead transaction preference:
     - **PayPal:** \$7,272.44 (29 orders)
     - **Cash on Delivery:** \$7,240.93 (28 orders)
     - **Credit Card:** \$6,206.35 (25 orders)
     - **Debit Card:** \$4,348.35 (23 orders)

6. **Customer Spending Segmentation:**
   - **High-Value Orders ($\ge \$250$):** Account for **66.4%** of total revenue (\$16,645.77 across 32 orders).
   - **Medium-Value Orders (\$100–\$250):** \$6,231.86 across 36 orders.
   - **Low-Value Orders ($< \$100$):** \$2,190.44 across 37 orders.

---

## ⚙️ How to Install Dependencies and Run

### 1. Prerequisites
Ensure Python 3.10+ is installed on your computer.

### 2. Install Required Libraries
Open your terminal in the project directory and install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-generate Raw Messy Data
To re-generate the raw messy dataset from scratch:
```bash
python generate_messy_data.py
```

### 4. Run the Data Wrangling Pipeline
Execute the main script:
```bash
python data_wrangling.py
```

### 5. Verify Outputs
Upon completion, verify that the following files are produced:
- `data/cleaned_sales_data.csv`
- `results/sales_by_category.png`
- `results/top_products.png`
- `results/monthly_sales.png`
- `results/sales_by_city.png`
- `results/rating_distribution.png`
- `results/analysis_results.txt`
