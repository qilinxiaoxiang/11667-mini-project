import pandas as pd
from datasets import Dataset

def analyze_structure():
    print("Loading Arrow file...")
    try:
        # Load the dataset using the datasets library (handles formatting best)
        ds = Dataset.from_file("../data/processed/hierarchical_dataset_clean/data-00000-of-00001.arrow")
        df = ds.to_pandas()
        
        print(f"✅ Successfully loaded {len(df)} rows.")
        print(f"   Columns found: {df.columns.tolist()}")
        print("="*60)

        # Columns to inspect
        columns_to_check = ['Doctor', '_original_doctor']
        
        for col in columns_to_check:
            if col not in df.columns:
                print(f"❌ Column '{col}' NOT FOUND in dataset.")
                continue

            print(f"🔍 INSPECTING COLUMN: {col}")
            
            # Get the first valid text entry
            sample_text = df[col].dropna().iloc[0]
            
            # Calculate metrics
            newline_count = sample_text.count('\n')
            
            print(f"   > Newlines in first sample: {newline_count}")
            if newline_count == 0:
                print("   ⚠️  WARNING: This text looks FLAT (No structure found).")
            else:
                print("   ✅  SUCCESS: This text has STRUCTURE (Newlines found).")
                
            print(f"\n   --- PREVIEW (First 300 chars) ---")
            print(sample_text[:300])
            print(f"   ---------------------------------\n")
            print("="*60)

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    analyze_structure()
