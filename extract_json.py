import pandas as pd
import json
import re

# File path to your JSON
json_file_path = "data/synthetic_data_inappropriate.json"  # Replace with your actual file path

# Step 1: Load JSON with robust error handling
try:
    with open(json_file_path, 'r', encoding='utf-8-sig') as file:
        raw_content = file.read().strip()
        data = json.loads(raw_content)
    
    # Filter out invalid entries (strings) and keep only dictionaries
    valid_data = [item for item in data if isinstance(item, dict)]
    invalid_count = len(data) - len(valid_data)
    if invalid_count > 0:
        print(f"Warning: Removed {invalid_count} invalid entries (non-dictionary items).")
    
    # Create DataFrame
    df = pd.DataFrame(valid_data)
    
except json.JSONDecodeError as e:
    print(f"JSON parsing error: {e}")
    print("Sample of raw content:", raw_content[:200])
    df = pd.DataFrame()
except Exception as e:
    print(f"Error loading JSON: {e}")
    df = pd.DataFrame()

# Step 2: Validate expected columns
expected_columns = ['business_description', 'domain_name']
if not df.empty and not all(col in df.columns for col in expected_columns):
    print(f"Error: JSON missing expected columns. Found: {df.columns.tolist()}")
    df = pd.DataFrame()
else:
    # Drop rows with missing required columns
    df = df.dropna(subset=expected_columns)
    
    # Handle extra columns (e.g., 'products')
    extra_columns = [col for col in df.columns if col not in expected_columns]
    if extra_columns:
        print(f"Warning: Found extra columns {extra_columns}. Keeping only {expected_columns}.")
        df = df[expected_columns]
    
    # Clean and standardize domain_name
    def clean_domain(domain):
        domain = str(domain).lower()  # Normalize to lowercase
        match = re.search(r'^[a-z0-9-]+\.(com|org|net|app|co|io|me|shop)$', domain)
        #return match.group(0) + "<|eot_id|>" if match else None
        return match.group(0) if match else None
    
    df['domain_name'] = df['domain_name'].apply(clean_domain)
    invalid_domains = df['domain_name'].isna().sum()
    if invalid_domains > 0:
        print(f"Warning: Removed {invalid_domains} rows with invalid domain names.")
    df = df.dropna(subset=['domain_name'])
    
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=['business_description', 'domain_name'])
    print(f"Removed {initial_len - len(df)} duplicate rows.")
    
    # Create formatted text for fine-tuning
    df['text'] = df.apply(lambda row: f"Business: {row['business_description'].strip()}\nDomain: {row['domain_name']}", axis=1)
    
    # Save to CSV
    df.to_csv("./data/synthetic_data_inappropriate.csv", index=False)
    
    # Print for verification
    print("\nDataFrame Info:")
    print(df.info())
    print("\nDataFrame Head:")
    print(df.head())
    print("\nSample Formatted Text:")
    print(df['text'].head())

# Step 3: Debug invalid entries (optional)
if invalid_count > 0:
    print("\nSample of Invalid Entries:")
    invalid_entries = [item for item in data if not isinstance(item, dict)]
    print(invalid_entries[:5])