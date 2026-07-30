import pandas as pd
import os

DATA_DIR = r"D:\forex_bot_data"

def clean_csv(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    print(f"Cleaning {file_path}...")
    
    # Read with error_bad_lines=False to skip malformed rows
    # on_bad_lines='skip' is the newer pandas equivalent
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True, on_bad_lines='skip')
    except Exception as e:
        print(f"  Standard read failed: {e}")
        print("  Attempting line-by-line clean...")
        lines = []
        with open(file_path, 'r') as f:
            header = f.readline()
            lines.append(header)
            for line in f:
                if line.strip() and line.count(',') == 6: # Ensure 7 columns (6 commas)
                    lines.append(line)
        
        with open(file_path, 'w') as f:
            f.writelines(lines)
            
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
    # Drop any rows with NaT index
    df = df[~df.index.isna()]
    
    # Sort by index to ensure chronological order
    df.sort_index(inplace=True)
    
    # Drop duplicates if any
    df = df[~df.index.duplicated(keep='first')]
    
    # Save back
    df.to_csv(file_path)
    print(f"  Cleaned. Total valid rows: {len(df)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

# Clean both files
clean_csv(os.path.join(DATA_DIR, "eurusd_h1_real.csv"))
clean_csv(os.path.join(DATA_DIR, "gbpusd_h1_real.csv"))

print("\nSUCCESS: CSV files cleaned and saved.")
