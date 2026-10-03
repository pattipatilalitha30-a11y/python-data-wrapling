"""
Script to generate realistic messy e-commerce/customer sales dataset
and category reference table for the Week 3 Data Wrangling project.
"""

import os
import random
import pandas as pd
import numpy as np

def generate_datasets():
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Generate category metadata for merge/combine requirement
    category_metadata = pd.DataFrame([
        {"Category": "Electronics", "Department": "Technology", "Target_Profit_Margin": 0.22, "Category_Manager": "Sarah Connor"},
        {"Category": "Home & Kitchen", "Department": "Home Living", "Target_Profit_Margin": 0.28, "Category_Manager": "Gordon Ramsay"},
        {"Category": "Sports & Fitness", "Department": "Athletics", "Target_Profit_Margin": 0.35, "Category_Manager": "Michael Jordan"},
        {"Category": "Fashion & Apparel", "Department": "Lifestyle", "Target_Profit_Margin": 0.40, "Category_Manager": "Miranda Priestly"},
        {"Category": "Office Supplies", "Department": "Business", "Target_Profit_Margin": 0.25, "Category_Manager": "Pam Beesly"}
    ])
    category_metadata_path = os.path.join(data_dir, "category_metadata.csv")
    category_metadata.to_csv(category_metadata_path, index=False)
    print(f"Created category metadata reference table at {category_metadata_path}")

    # 2. Products catalog by category with benchmark unit price
    products_catalog = {
        "Electronics": [
            ("Wireless Headphones", 99.99),
            ("Gaming Mouse", 49.99),
            ("Mechanical Keyboard", 119.50),
            ("Smart Watch", 199.99),
            ("Bluetooth Speaker", 79.00),
            ("Tablet", 329.99),
        ],
        "Home & Kitchen": [
            ("Coffee Maker", 85.00),
            ("Air Fryer", 120.00),
            ("Blender", 45.50),
            ("Chef Knife Set", 65.00),
            ("Toaster", 35.00),
        ],
        "Sports & Fitness": [
            ("Yoga Mat", 25.00),
            ("Running Shoes", 110.00),
            ("Dumbbell Set", 75.00),
            ("Water Bottle", 18.50),
            ("Resistance Bands", 15.00),
        ],
        "Fashion & Apparel": [
            ("Denim Jacket", 68.00),
            ("Sneakers", 95.00),
            ("Hoodie", 42.00),
            ("Backpack", 55.00),
            ("Sunglasses", 30.00),
        ],
        "Office Supplies": [
            ("Desk Chair", 180.00),
            ("Desk Lamp", 38.00),
            ("Monitor Stand", 29.99),
            ("Notebook Pack", 12.50),
            ("Ergonomic Mousepad", 16.00),
        ]
    }

    cities_variations = [
        "New York", "new york", " NEW YORK ", "New York ",
        "Los Angeles", "los angeles", " Los Angeles", "LOS ANGELES",
        "Chicago", "chicago", " CHICAGO ",
        "Houston", "houston", "Houston ",
        "Miami", "miami", " MIAMI",
        "Seattle", "seattle", "Seattle ",
        "San Francisco", "san francisco", "San Francisco "
    ]

    categories_variations = {
        "Electronics": ["Electronics", "electronics", " ELECTRONICS ", "Electronics "],
        "Home & Kitchen": ["Home & Kitchen", "home & kitchen", "HOME & KITCHEN", " Home & Kitchen "],
        "Sports & Fitness": ["Sports & Fitness", "sports & fitness", "SPORTS & FITNESS", "Sports & Fitness "],
        "Fashion & Apparel": ["Fashion & Apparel", "fashion & apparel", "FASHION & APPAREL"],
        "Office Supplies": ["Office Supplies", "office supplies", "OFFICE SUPPLIES", " Office Supplies "]
    }

    payment_variations = {
        "Credit Card": ["Credit Card", "credit card", "CREDIT_CARD", "Credit Card "],
        "PayPal": ["PayPal", "paypal", "PAYPAL", " PayPal "],
        "Debit Card": ["Debit Card", "debit card", "DEBIT_CARD"],
        "Cash on Delivery": ["Cash on Delivery", "COD", "cash on delivery", "cash", "CASH"]
    }

    gender_variations = ["Male", "male", "M", "Female", "female", "F", "Other", None, ""]

    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
                   "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
                   "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
                   "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
                   "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle"]
    
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                  "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
                  "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White"]

    # Pre-create pool of 45 customers
    customers = []
    for i in range(1, 46):
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        cid = f"CUST{i:03d}"
        customers.append((cid, f"{fn} {ln}", random.randint(19, 65)))

    raw_records = []
    
    # Generate 115 initial records
    for i in range(1, 116):
        cust = random.choice(customers)
        cust_id, cust_name, base_age = cust
        
        # Add realistic noise/mess to Customer_ID
        if i % 15 == 0:
            cid_val = cust_id.lower()
        elif i % 25 == 0:
            cid_val = f" {cust_id} "
        elif i == 77:
            cid_val = None
        else:
            cid_val = cust_id

        # Noise to Customer_Name
        if i % 18 == 0:
            cname_val = cust_name.upper()
        elif i % 22 == 0:
            cname_val = f"  {cust_name}  "
        elif i == 88:
            cname_val = ""
        else:
            cname_val = cust_name

        # Age with missing / outlier
        if i in [12, 45, 92]:
            age_val = None
        elif i == 33:
            age_val = -5  # invalid negative age
        elif i == 66:
            age_val = 145 # invalid extreme outlier
        elif i % 10 == 0:
            age_val = str(base_age)
        else:
            age_val = base_age

        # Gender
        gender_val = random.choice(gender_variations)

        # City
        if i in [17, 73]:
            city_val = None
        elif i == 50:
            city_val = "   "
        else:
            city_val = random.choice(cities_variations)

        # Category and Product
        std_cat = random.choice(list(products_catalog.keys()))
        cat_val = random.choice(categories_variations[std_cat])
        prod_tuple = random.choice(products_catalog[std_cat])
        prod_name, base_price = prod_tuple
        
        if i % 8 == 0:
            prod_val = f" {prod_name} "
        else:
            prod_val = prod_name

        # Quantity (some string, some invalid 0 or -1, some float strings)
        if i == 29:
            qty_val = -1 # invalid negative
        elif i == 82:
            qty_val = 0  # invalid zero quantity
        elif i in [14, 58]:
            qty_val = None
        elif i % 7 == 0:
            qty_val = f" {random.randint(1, 4)} "
        elif i % 9 == 0:
            qty_val = f"{random.randint(1, 3)}.0"
        else:
            qty_val = random.randint(1, 5)

        # Unit Price (messy formatting: $, missing, string, etc.)
        if i in [23, 67]:
            price_val = None
        elif i == 41:
            price_val = -20.0 # invalid negative price
        elif i % 3 == 0:
            price_val = f"${base_price:.2f}"
        elif i % 5 == 0:
            price_val = f" {base_price:.2f} "
        else:
            price_val = base_price

        # Order Date (mixed formats)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        year = 2023
        if i == 55:
            date_val = "invalid_date"
        elif i == 99:
            date_val = None
        elif i % 4 == 0:
            date_val = f"{year:04d}-{month:02d}-{day:02d}"
        elif i % 4 == 1:
            date_val = f"{month:02d}/{day:02d}/{year:04d}"
        elif i % 4 == 2:
            date_val = f"{year:04d}.{month:02d}.{day:02d}"
        else:
            date_val = f"{day:02d}-{month:02d}-{year:04d}"

        # Payment Method
        std_pay = random.choice(list(payment_variations.keys()))
        pay_val = random.choice(payment_variations[std_pay])
        if i in [38, 85]:
            pay_val = None

        # Rating (1 to 5, some messy strings, some missing, some invalid)
        if i in [19, 74]:
            rating_val = None
        elif i == 48:
            rating_val = 6.0 # invalid > 5
        elif i == 94:
            rating_val = -1.0 # invalid < 0
        elif i % 6 == 0:
            rating_val = str(random.choice([3, 4, 5]))
        else:
            rating_val = round(random.choice([3.0, 3.5, 4.0, 4.2, 4.5, 4.8, 5.0]), 1)

        # Sales (messy column, missing or slightly inaccurate or string)
        if i in [5, 31, 62, 101]:
            sales_val = None
        elif i % 4 == 0:
            sales_val = f"${(float(base_price) * 2):.2f}"
        else:
            try:
                sales_val = round(float(str(price_val).replace('$', '').strip()) * 2, 2)
            except Exception:
                sales_val = 150.0

        record = {
            "Customer_ID": cid_val,
            "Customer_Name": cname_val,
            "Age": age_val,
            "Gender": gender_val,
            "City": city_val,
            "Product": prod_val,
            "Category": cat_val,
            "Quantity": qty_val,
            "Unit_Price": price_val,
            "Order_Date": date_val,
            "Payment_Method": pay_val,
            "Rating": rating_val,
            "Sales": sales_val
        }
        raw_records.append(record)

    # 3. Add 10 duplicate records (duplicating existing indices)
    duplicate_indices = [3, 11, 24, 35, 47, 60, 71, 83, 90, 105]
    for idx in duplicate_indices:
        raw_records.append(dict(raw_records[idx]))

    # Convert to DataFrame
    df_messy = pd.DataFrame(raw_records)
    messy_path = os.path.join(data_dir, "messy_sales_data.csv")
    df_messy.to_csv(messy_path, index=False)
    print(f"Generated messy sales dataset with {len(df_messy)} records at {messy_path}")

if __name__ == "__main__":
    generate_datasets()
