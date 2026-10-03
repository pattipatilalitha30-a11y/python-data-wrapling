"""
Customer Sales Data Cleaning and Analysis using Python
Week 3: Python Basics, Pandas Data Wrangling, and Data Visualization

Author: Lalitha Patipatti
Purpose: Clean a messy realistic e-commerce customer sales dataset,
         perform exploratory data analysis, and produce professional visualizations.
"""

import os
import sys
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set styling for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def get_project_paths() -> Tuple[str, str, str]:
    """
    Resolve base directory, data directory, and results directory.
    Works whether run from workspace root or subdirectories.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return base_dir, data_dir, results_dir


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV data using Pandas.

    Args:
        filepath (str): Path to messy sales CSV file.

    Returns:
        pd.DataFrame: Loaded raw DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    
    print(f"[1/7] Loading raw dataset from: {filepath}")
    df = pd.read_csv(filepath)
    return df


def inspect_dataset(df: pd.DataFrame, label: str = "RAW DATASET") -> Dict[str, Any]:
    """
    Inspect dataset shape, column types, missing values, and duplicates.

    Args:
        df (pd.DataFrame): DataFrame to inspect.
        label (str): Label for output printing.

    Returns:
        Dict[str, Any]: Summary dictionary of inspection metrics.
    """
    print("\n" + "=" * 60)
    print(f" {label} INSPECTION")
    print("=" * 60)
    
    rows, cols = df.shape
    total_cells = rows * cols
    missing_series = df.isna().sum()
    total_missing = missing_series.sum()
    duplicates_count = df.duplicated().sum()

    print(f"Shape: {rows} rows x {cols} columns")
    print(f"Total Duplicates: {duplicates_count}")
    print(f"Total Missing Values: {total_missing} ({total_missing / total_cells * 100:.2f}% of all cells)")
    print("\nMissing values per column:")
    for col, val in missing_series.items():
        if val > 0:
            print(f"  - {col}: {val} missing ({val / rows * 100:.1f}%)")

    print("\nColumn Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col:16s}: {dtype}")

    print("=" * 60)

    return {
        "rows": rows,
        "cols": cols,
        "duplicates": duplicates_count,
        "missing_per_col": missing_series.to_dict(),
        "total_missing": total_missing,
        "dtypes": df.dtypes.astype(str).to_dict()
    }


def clean_sales_data(df_raw: pd.DataFrame, category_meta_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive data cleaning pipeline implementing Week 3 requirements:
    1. Remove duplicate records
    2. Standardize text values & casing
    3. Correct data types (currency to float, dates to datetime, quantities to int)
    4. Validate numerical columns & filter invalid rows (negative/zero qty, out-of-range ratings)
    5. Handle missing values via intelligent imputation
    6. Feature engineering (Total_Sales, Customer_Segment, Month, etc.)
    7. Merge/combine reference category metadata

    Args:
        df_raw (pd.DataFrame): Raw messy dataset.
        category_meta_path (str, optional): Path to reference metadata table.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and cleaning metrics dictionary.
    """
    print("\n[2/7] Starting comprehensive data cleaning pipeline...")
    df = df_raw.copy()
    initial_rows = len(df)

    # 1. Remove duplicate records
    exact_duplicates = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  -> Removed {exact_duplicates} exact duplicate rows.")

    # 2. Standardize text values across categorical columns
    print("  -> Standardizing text casing and stripping extraneous whitespace...")
    
    # Customer_ID
    df["Customer_ID"] = df["Customer_ID"].astype(str).str.strip().str.upper()
    df.loc[df["Customer_ID"].isin(["NONE", "NAN", "", "NULL"]), "Customer_ID"] = np.nan

    # Customer_Name
    df["Customer_Name"] = df["Customer_Name"].astype(str).str.strip().str.title()
    df.loc[df["Customer_Name"].isin(["", "Nan", "None", "Null"]), "Customer_Name"] = "Valued Customer"

    # City standardization
    city_map = {
        "new york": "New York", "new york ": "New York",
        "los angeles": "Los Angeles", "los angeles ": "Los Angeles",
        "chicago": "Chicago", "houston": "Houston",
        "miami": "Miami", "seattle": "Seattle",
        "san francisco": "San Francisco"
    }
    df["City"] = df["City"].astype(str).str.strip().str.title()
    df.loc[df["City"].isin(["", "Nan", "None", "Null"]), "City"] = np.nan
    # Impute missing city with the most frequent city (mode)
    city_mode = df["City"].dropna().mode()[0] if not df["City"].dropna().empty else "Chicago"
    df["City"] = df["City"].fillna(city_mode)

    # Category standardization
    category_map = {
        "ELECTRONICS": "Electronics", "electronics": "Electronics", "Electronics": "Electronics",
        "HOME & KITCHEN": "Home & Kitchen", "home & kitchen": "Home & Kitchen", "Home & Kitchen": "Home & Kitchen",
        "SPORTS & FITNESS": "Sports & Fitness", "sports & fitness": "Sports & Fitness", "Sports & Fitness": "Sports & Fitness",
        "FASHION & APPAREL": "Fashion & Apparel", "fashion & apparel": "Fashion & Apparel", "Fashion & Apparel": "Fashion & Apparel",
        "OFFICE SUPPLIES": "Office Supplies", "office supplies": "Office Supplies", "Office Supplies": "Office Supplies"
    }
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Category"] = df["Category"].map(lambda x: category_map.get(x, x.title()))

    # Product standardization
    df["Product"] = df["Product"].astype(str).str.strip().str.title()

    # Payment Method standardization
    def clean_payment_method(val: Any) -> str:
        if pd.isna(val):
            return "Credit Card"
        v = str(val).strip().upper()
        if "CREDIT" in v:
            return "Credit Card"
        elif "DEBIT" in v:
            return "Debit Card"
        elif "PAYPAL" in v:
            return "PayPal"
        elif "CASH" in v or "COD" in v or "DELIVERY" in v:
            return "Cash on Delivery"
        return "Credit Card"

    df["Payment_Method"] = df["Payment_Method"].apply(clean_payment_method)

    # Gender standardization
    gender_map = {
        "Male": "Male", "male": "Male", "M": "Male",
        "Female": "Female", "female": "Female", "F": "Female",
        "Other": "Other"
    }
    df["Gender"] = df["Gender"].astype(str).str.strip()
    df["Gender"] = df["Gender"].map(lambda x: gender_map.get(x, "Not Specified"))

    # 3. Clean and convert numerical and date columns
    print("  -> Correcting data types and parsing dates...")
    
    # Unit_Price: strip '$' and convert to float
    df["Unit_Price"] = (
        df["Unit_Price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")

    # Quantity: convert to numeric
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    # Age: convert to numeric
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Rating: convert to numeric
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # Order_Date: parse dates with mixed formats
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="mixed", errors="coerce")

    # 4. Filter invalid rows & Validate numerical columns
    print("  -> Validating numerical boundaries and filtering invalid records...")
    valid_mask = (
        df["Customer_ID"].notna() &
        df["Order_Date"].notna() &
        df["Quantity"].notna() & (df["Quantity"] > 0) &
        df["Unit_Price"].notna() & (df["Unit_Price"] > 0)
    )
    invalid_rows_count = len(df) - valid_mask.sum()
    df = df[valid_mask].copy()
    print(f"  -> Filtered out {invalid_rows_count} corrupted rows (missing ID/date or non-positive Qty/Price).")

    # Convert Quantity to integer
    df["Quantity"] = df["Quantity"].astype(int)

    # Validate and impute Age (valid age between 18 and 90)
    invalid_age_mask = (df["Age"] < 18) | (df["Age"] > 90) | df["Age"].isna()
    median_age = int(df.loc[~invalid_age_mask, "Age"].median())
    df.loc[invalid_age_mask, "Age"] = median_age
    df["Age"] = df["Age"].astype(int)

    # Validate and impute Rating (valid rating between 1.0 and 5.0)
    invalid_rating_mask = (df["Rating"] < 1.0) | (df["Rating"] > 5.0) | df["Rating"].isna()
    median_rating = round(df.loc[~invalid_rating_mask, "Rating"].median(), 1)
    df.loc[invalid_rating_mask, "Rating"] = median_rating
    df["Rating"] = df["Rating"].round(1)

    # 5. Create new columns (Feature Engineering)
    print("  -> Creating calculated columns (Total_Sales, Customer_Segment, Month)...")
    # Total_Sales = Quantity * Unit_Price
    df["Total_Sales"] = (df["Quantity"] * df["Unit_Price"]).round(2)

    # Customer_Segment based on spending
    def segment_order(val: float) -> str:
        if val >= 250.0:
            return "High"
        elif val >= 100.0:
            return "Medium"
        else:
            return "Low"

    df["Customer_Segment"] = df["Total_Sales"].apply(segment_order)
    df["Order_Month"] = df["Order_Date"].dt.strftime("%Y-%m")
    df["Order_Year"] = df["Order_Date"].dt.year
    df["Day_of_Week"] = df["Order_Date"].dt.day_name()

    # Drop old/messy raw Sales column if present, since Total_Sales is accurate
    if "Sales" in df.columns:
        df = df.drop(columns=["Sales"])

    # 6. Merge/Combine with reference category metadata
    if category_meta_path and os.path.exists(category_meta_path):
        print(f"  -> Merging with category metadata reference from {category_meta_path}...")
        df_meta = pd.read_csv(category_meta_path)
        df_meta["Category"] = df_meta["Category"].str.strip()
        df = pd.merge(df, df_meta, on="Category", how="left")
        
        # Calculate Estimated_Profit
        if "Target_Profit_Margin" in df.columns:
            df["Estimated_Profit"] = (df["Total_Sales"] * df["Target_Profit_Margin"]).round(2)
        print("  -> Successfully merged category metadata and calculated Estimated_Profit.")

    # Sort records chronologically
    df = df.sort_values(by=["Order_Date", "Customer_ID"]).reset_index(drop=True)
    final_rows = len(df)
    print(f"[2/7 Completed] Cleaned dataset ready with {final_rows} high-quality records.")

    cleaning_metrics = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "duplicates_removed": exact_duplicates,
        "corrupted_rows_filtered": invalid_rows_count,
        "median_age_imputed": median_age,
        "median_rating_imputed": median_rating
    }

    return df, cleaning_metrics


def perform_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform required descriptive and exploratory sales data analysis:
    - Total sales
    - Average sales
    - Total quantity sold
    - Top products
    - Top customers
    - Sales by category
    - Sales by city
    - Sales by payment method
    - Monthly sales
    - Average customer rating

    Args:
        df (pd.DataFrame): Cleaned sales DataFrame.

    Returns:
        Dict[str, Any]: Dictionary containing all aggregated analysis metrics.
    """
    print("\n[3/7] Performing comprehensive business data analysis...")

    total_sales = float(df["Total_Sales"].sum())
    avg_sales = float(df["Total_Sales"].mean())
    median_sales = float(df["Total_Sales"].median())
    total_quantity = int(df["Quantity"].sum())
    avg_rating = float(df["Rating"].mean())
    total_orders = len(df)
    unique_customers = int(df["Customer_ID"].nunique())

    # Top products by sales
    top_products_sales = (
        df.groupby("Product")
        .agg(
            Total_Sales=("Total_Sales", "sum"),
            Total_Quantity=("Quantity", "sum"),
            Avg_Price=("Unit_Price", "mean")
        )
        .sort_values(by="Total_Sales", ascending=False)
        .round(2)
    )

    # Top customers by spending
    top_customers = (
        df.groupby(["Customer_ID", "Customer_Name"])
        .agg(
            Total_Spent=("Total_Sales", "sum"),
            Order_Count=("Customer_ID", "count"),
            Avg_Order_Value=("Total_Sales", "mean")
        )
        .sort_values(by="Total_Spent", ascending=False)
        .round(2)
    )

    # Sales by Category
    sales_by_category = (
        df.groupby("Category")
        .agg(
            Total_Sales=("Total_Sales", "sum"),
            Order_Count=("Category", "count"),
            Avg_Sales=("Total_Sales", "mean")
        )
        .sort_values(by="Total_Sales", ascending=False)
        .round(2)
    )

    # Sales by City
    sales_by_city = (
        df.groupby("City")
        .agg(
            Total_Sales=("Total_Sales", "sum"),
            Order_Count=("City", "count"),
            Avg_Sales=("Total_Sales", "mean")
        )
        .sort_values(by="Total_Sales", ascending=False)
        .round(2)
    )

    # Sales by Payment Method
    sales_by_payment = (
        df.groupby("Payment_Method")
        .agg(
            Total_Sales=("Total_Sales", "sum"),
            Transaction_Count=("Payment_Method", "count")
        )
        .sort_values(by="Total_Sales", ascending=False)
        .round(2)
    )

    # Monthly Sales Trend
    monthly_sales = (
        df.groupby("Order_Month")
        .agg(
            Total_Sales=("Total_Sales", "sum"),
            Order_Count=("Order_Month", "count"),
            Total_Quantity=("Quantity", "sum")
        )
        .sort_index()
        .round(2)
    )

    # Customer Segment Summary
    segment_summary = (
        df.groupby("Customer_Segment")
        .agg(
            Order_Count=("Customer_Segment", "count"),
            Total_Sales=("Total_Sales", "sum"),
            Avg_Sales=("Total_Sales", "mean")
        )
        .loc[["High", "Medium", "Low"]]
        .round(2)
    )

    # Rating Summary
    rating_summary = {
        "mean_rating": round(avg_rating, 2),
        "median_rating": float(df["Rating"].median()),
        "rating_breakdown": df["Rating"].value_counts().sort_index().to_dict()
    }

    results = {
        "total_sales": round(total_sales, 2),
        "avg_sales": round(avg_sales, 2),
        "median_sales": round(median_sales, 2),
        "total_quantity": total_quantity,
        "avg_rating": round(avg_rating, 2),
        "total_orders": total_orders,
        "unique_customers": unique_customers,
        "top_products_sales": top_products_sales,
        "top_customers": top_customers,
        "sales_by_category": sales_by_category,
        "sales_by_city": sales_by_city,
        "sales_by_payment": sales_by_payment,
        "monthly_sales": monthly_sales,
        "segment_summary": segment_summary,
        "rating_summary": rating_summary
    }

    return results


def generate_visualizations(df: pd.DataFrame, analysis_results: Dict[str, Any], results_dir: str) -> Dict[str, str]:
    """
    Generate at least 5 professional visualizations using Matplotlib & Seaborn:
    1. Sales by category (results/sales_by_category.png)
    2. Top 10 products by sales (results/top_products.png)
    3. Monthly sales trend (results/monthly_sales.png)
    4. Sales by city (results/sales_by_city.png)
    5. Customer rating distribution (results/rating_distribution.png)

    Args:
        df (pd.DataFrame): Cleaned sales DataFrame.
        analysis_results (Dict[str, Any]): Aggregated analysis metrics.
        results_dir (str): Directory where charts will be saved.

    Returns:
        Dict[str, str]: Mapping of chart title to saved file path.
    """
    print("\n[4/7] Generating professional data visualizations...")
    saved_charts = {}

    # Chart 1: Sales by Category
    chart1_path = os.path.join(results_dir, "sales_by_category.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_df = analysis_results["sales_by_category"].reset_index()
    palette = sns.color_palette("Blues_r", n_colors=len(cat_df))
    bars = sns.barplot(data=cat_df, x="Category", y="Total_Sales", ax=ax, palette=palette, hue="Category", legend=False)
    ax.set_title("Total Sales Revenue by Product Category", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Product Category", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Total Sales (USD)", fontsize=11, fontweight="semibold")
    plt.xticks(rotation=20, ha="right")
    
    # Add value labels on top of bars
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(f"${height:,.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(chart1_path, dpi=300)
    plt.close(fig)
    saved_charts["sales_by_category"] = chart1_path
    print(f"  -> Saved Chart 1: {chart1_path}")

    # Chart 2: Top 10 Products by Sales
    chart2_path = os.path.join(results_dir, "top_products.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    top_p_df = analysis_results["top_products_sales"].head(10).reset_index()
    palette_top = sns.color_palette("viridis", n_colors=len(top_p_df))
    bars = sns.barplot(data=top_p_df, x="Total_Sales", y="Product", ax=ax, palette=palette_top, hue="Product", legend=False)
    ax.set_title("Top 10 Products by Total Sales Revenue", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Total Sales (USD)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Product Name", fontsize=11, fontweight="semibold")
    
    # Add value labels
    for bar in bars.patches:
        width = bar.get_width()
        ax.annotate(f" ${width:,.2f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(chart2_path, dpi=300)
    plt.close(fig)
    saved_charts["top_products"] = chart2_path
    print(f"  -> Saved Chart 2: {chart2_path}")

    # Chart 3: Monthly Sales Trend
    chart3_path = os.path.join(results_dir, "monthly_sales.png")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    m_df = analysis_results["monthly_sales"].reset_index()
    ax.plot(m_df["Order_Month"], m_df["Total_Sales"], marker="o", color="#1f77b4", linewidth=2.5, markersize=8, label="Monthly Revenue")
    ax.fill_between(m_df["Order_Month"], m_df["Total_Sales"], color="#1f77b4", alpha=0.15)
    ax.set_title("Monthly Sales Revenue Trend (2023)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Month (YYYY-MM)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Total Sales (USD)", fontsize=11, fontweight="semibold")
    plt.xticks(rotation=30)
    
    for _, row in m_df.iterrows():
        ax.annotate(f"${row['Total_Sales']:,.0f}",
                    xy=(row["Order_Month"], row["Total_Sales"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(chart3_path, dpi=300)
    plt.close(fig)
    saved_charts["monthly_sales"] = chart3_path
    print(f"  -> Saved Chart 3: {chart3_path}")

    # Chart 4: Sales by City
    chart4_path = os.path.join(results_dir, "sales_by_city.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    city_df = analysis_results["sales_by_city"].reset_index()
    palette_city = sns.color_palette("magma", n_colors=len(city_df))
    bars = sns.barplot(data=city_df, x="City", y="Total_Sales", ax=ax, palette=palette_city, hue="City", legend=False)
    ax.set_title("Customer Sales Revenue Across Major Cities", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("City", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Total Sales (USD)", fontsize=11, fontweight="semibold")
    plt.xticks(rotation=25, ha="right")
    
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(f"${height:,.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(chart4_path, dpi=300)
    plt.close(fig)
    saved_charts["sales_by_city"] = chart4_path
    print(f"  -> Saved Chart 4: {chart4_path}")

    # Chart 5: Customer Rating Distribution
    chart5_path = os.path.join(results_dir, "rating_distribution.png")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rating_counts = df["Rating"].value_counts().sort_index().reset_index()
    rating_counts.columns = ["Rating", "Count"]
    bars = sns.barplot(data=rating_counts, x="Rating", y="Count", ax=ax, palette="crest", hue="Rating", legend=False)
    ax.set_title(f"Customer Rating Distribution (Mean: {analysis_results['avg_rating']} / 5.0)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Customer Satisfaction Rating (1 to 5 Stars)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Number of Orders", fontsize=11, fontweight="semibold")
    
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(chart5_path, dpi=300)
    plt.close(fig)
    saved_charts["rating_distribution"] = chart5_path
    print(f"  -> Saved Chart 5: {chart5_path}")

    return saved_charts


def save_cleaned_data(df: pd.DataFrame, target_path: str) -> None:
    """
    Save the final cleaned dataset as a CSV file.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        target_path (str): Target CSV file destination.
    """
    print(f"\n[5/7] Saving cleaned dataset to: {target_path}")
    df.to_csv(target_path, index=False)
    print(f"  -> Successfully saved {len(df)} records.")


def save_analysis_report(df: pd.DataFrame, analysis: Dict[str, Any], cleaning: Dict[str, Any], output_path: str) -> None:
    """
    Write a detailed, structured business analysis report to results/analysis_results.txt.

    Args:
        df (pd.DataFrame): Cleaned sales DataFrame.
        analysis (Dict[str, Any]): Dictionary of analysis results.
        cleaning (Dict[str, Any]): Data cleaning metrics.
        output_path (str): Filepath for text report.
    """
    print(f"\n[6/7] Writing detailed analysis report to: {output_path}")

    report_lines = []
    report_lines.append("=" * 75)
    report_lines.append(" CUSTOMER SALES DATA CLEANING AND ANALYSIS REPORT")
    report_lines.append(" Week 3: Python Basics, Pandas Wrangling & Data Visualization")
    report_lines.append("=" * 75)
    report_lines.append("")

    report_lines.append("1. DATA WRANGLING & CLEANING SUMMARY")
    report_lines.append("-" * 50)
    report_lines.append(f"Initial Raw Records:            {cleaning['initial_rows']}")
    report_lines.append(f"Duplicate Rows Removed:         {cleaning['duplicates_removed']}")
    report_lines.append(f"Corrupted/Filtered Rows:        {cleaning['corrupted_rows_filtered']}")
    report_lines.append(f"Final Cleaned Records:          {cleaning['final_rows']}")
    report_lines.append(f"Imputed Missing/Outlier Age:    Median ({cleaning['median_age_imputed']} years)")
    report_lines.append(f"Imputed Missing/Outlier Rating: Median ({cleaning['median_rating_imputed']} / 5.0)")
    report_lines.append("")

    report_lines.append("2. OVERALL SALES & PERFORMANCE KPIS")
    report_lines.append("-" * 50)
    report_lines.append(f"Total Sales Revenue:            ${analysis['total_sales']:,.2f}")
    report_lines.append(f"Average Order Value:            ${analysis['avg_sales']:,.2f}")
    report_lines.append(f"Median Order Value:             ${analysis['median_sales']:,.2f}")
    report_lines.append(f"Total Quantity Sold:            {analysis['total_quantity']:,} units")
    report_lines.append(f"Total Completed Transactions:   {analysis['total_orders']}")
    report_lines.append(f"Unique Customers:               {analysis['unique_customers']}")
    report_lines.append(f"Average Customer Rating:        {analysis['avg_rating']} / 5.00")
    report_lines.append("")

    report_lines.append("3. SALES BY PRODUCT CATEGORY")
    report_lines.append("-" * 50)
    report_lines.append(analysis["sales_by_category"].to_string())
    report_lines.append("")

    report_lines.append("4. TOP 10 PRODUCTS BY REVENUE")
    report_lines.append("-" * 50)
    report_lines.append(analysis["top_products_sales"].head(10).to_string())
    report_lines.append("")

    report_lines.append("5. TOP 10 CUSTOMERS BY TOTAL SPENDING")
    report_lines.append("-" * 50)
    report_lines.append(analysis["top_customers"].head(10).to_string())
    report_lines.append("")

    report_lines.append("6. SALES BY CITY")
    report_lines.append("-" * 50)
    report_lines.append(analysis["sales_by_city"].to_string())
    report_lines.append("")

    report_lines.append("7. SALES BY PAYMENT METHOD")
    report_lines.append("-" * 50)
    report_lines.append(analysis["sales_by_payment"].to_string())
    report_lines.append("")

    report_lines.append("8. MONTHLY SALES REVENUE TREND (2023)")
    report_lines.append("-" * 50)
    report_lines.append(analysis["monthly_sales"].to_string())
    report_lines.append("")

    report_lines.append("9. CUSTOMER SPENDING SEGMENTS")
    report_lines.append("-" * 50)
    report_lines.append(analysis["segment_summary"].to_string())
    report_lines.append("")

    report_lines.append("10. KEY INSIGHTS & TAKEAWAYS")
    report_lines.append("-" * 50)
    top_cat = analysis["sales_by_category"].index[0]
    top_cat_rev = analysis["sales_by_category"].iloc[0]["Total_Sales"]
    top_prod = analysis["top_products_sales"].index[0]
    top_prod_rev = analysis["top_products_sales"].iloc[0]["Total_Sales"]
    top_city = analysis["sales_by_city"].index[0]
    top_city_rev = analysis["sales_by_city"].iloc[0]["Total_Sales"]

    report_lines.append(f"- Highest Performing Category: {top_cat} with ${top_cat_rev:,.2f} in sales.")
    report_lines.append(f"- Best-Selling Product:         {top_prod} generating ${top_prod_rev:,.2f}.")
    report_lines.append(f"- Top Geographical Market:      {top_city} generating ${top_city_rev:,.2f}.")
    report_lines.append(f"- Overall Customer Sentiment:   Positive with an average rating of {analysis['avg_rating']}/5.0.")
    report_lines.append("=" * 75)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"  -> Successfully generated {output_path}")


def print_terminal_summary(analysis: Dict[str, Any], cleaning: Dict[str, Any]) -> None:
    """
    Print an elegant, human-readable summary to standard console output.
    """
    print("\n" + "=" * 65)
    print("           PROJECT EXECUTION SUMMARY & KEY FINDINGS")
    print("=" * 65)
    print(f" Raw Records Ingested:        {cleaning['initial_rows']}")
    print(f" Duplicate Records Removed:   {cleaning['duplicates_removed']}")
    print(f" Corrupted Records Filtered:  {cleaning['corrupted_rows_filtered']}")
    print(f" Clean Records Analyzed:      {cleaning['final_rows']}")
    print("-" * 65)
    print(f" Total Revenue Generated:     ${analysis['total_sales']:,.2f}")
    print(f" Average Order Value:         ${analysis['avg_sales']:,.2f}")
    print(f" Total Units Sold:            {analysis['total_quantity']:,}")
    print(f" Unique Customers:            {analysis['unique_customers']}")
    print(f" Average Rating:              {analysis['avg_rating']} / 5.0")
    print("-" * 65)
    print(f" Top Category:                {analysis['sales_by_category'].index[0]}")
    print(f" Top Product:                 {analysis['top_products_sales'].index[0]}")
    print(f" Top City:                    {analysis['sales_by_city'].index[0]}")
    print(f" Top Payment Method:          {analysis['sales_by_payment'].index[0]}")
    print("=" * 65)
    print(" All required files, dataset, and visualizations generated successfully!\n")


def main() -> None:
    """
    Main orchestration routine for Week 3 project:
    Coordinates ingestion, cleaning, analysis, charting, and reporting.
    """
    print("==================================================================")
    print("  CUSTOMER SALES DATA CLEANING AND ANALYSIS (WEEK 3 PROJECT)     ")
    print("==================================================================")

    # 1. Resolve paths
    base_dir, data_dir, results_dir = get_project_paths()
    raw_csv_path = os.path.join(data_dir, "messy_sales_data.csv")
    cleaned_csv_path = os.path.join(data_dir, "cleaned_sales_data.csv")
    category_meta_path = os.path.join(data_dir, "category_metadata.csv")
    report_txt_path = os.path.join(results_dir, "analysis_results.txt")

    # 2. Ingest raw messy data
    df_raw = load_raw_data(raw_csv_path)

    # 3. Inspect raw data
    inspect_dataset(df_raw, label="RAW MESSY DATASET")

    # 4. Clean data
    df_clean, cleaning_metrics = clean_sales_data(df_raw, category_meta_path)

    # 5. Inspect cleaned data
    inspect_dataset(df_clean, label="CLEANED DATASET")

    # 6. Save cleaned data
    save_cleaned_data(df_clean, cleaned_csv_path)

    # 7. Perform exploratory business analysis
    analysis_results = perform_analysis(df_clean)

    # 8. Generate 5 required visualizations
    generate_visualizations(df_clean, analysis_results, results_dir)

    # 9. Save detailed text analysis report
    save_analysis_report(df_clean, analysis_results, cleaning_metrics, report_txt_path)

    # 10. Print terminal summary
    print_terminal_summary(analysis_results, cleaning_metrics)


if __name__ == "__main__":
    main()
