import pandas as pd

# Step 1: Define industry codes for each PERMNO
def create_permno_criteria():
    permno_criteria_data = {
        'PERMNO': [10000, 10001, 10002],  # Example PERMNOs
        'IndustryCode': [200, 300, 400]  # Corresponding industry codes
    }
    
    # Create DataFrame from criteria data
    permno_criteria_df = pd.DataFrame(permno_criteria_data)
    
    # Save the criteria DataFrame as a Parquet file
    permno_criteria_df.to_parquet('permno_criteria.parquet', index=False)

# Call the function to create the criteria file
create_permno_criteria()

# Step 2: Define the filter function
def filter_dataset_by_permno(full_rseqs_file, permno_criteria_file, criteria_column, filter_logic):
    # Load the full dataset
    full_rseqs_df = pd.read_parquet(full_rseqs_file)
    
    # Load the criteria dataset
    permno_criteria_df = pd.read_parquet(permno_criteria_file)

    # Debug: Print summary of the DataFrames
    print("Full Rseqs DataFrame Info:")
    print(full_rseqs_df.info())
    print("\nFull Rseqs DataFrame Head:")
    print(full_rseqs_df.head())

    print("\nPermno Criteria DataFrame Info:")
    print(permno_criteria_df.info())
    print("\nPermno Criteria DataFrame Head:")
    print(permno_criteria_df.head())

    # Merge datasets on PERMNO
    merged_df = pd.merge(full_rseqs_df, permno_criteria_df, left_on='permno', right_on='PERMNO', how='inner')
    
    # Apply filtering logic
    filtered_df = filter_logic(merged_df, criteria_column)

    return filtered_df

# Example filter logic: Filter by specific industry codes
def industry_filter_logic(df, criteria_column):
    industry_codes_to_keep = [200]  # Modify as needed to specify which industry codes to keep
    return df[df[criteria_column].isin(industry_codes_to_keep)]

# Step 3: Usage example
if __name__ == "__main__":
    # Create the criteria file if it doesn't exist
    create_permno_criteria()

    # Call the filter function
    filtered_data = filter_dataset_by_permno('full_rseqs.parquet', 'permno_criteria.parquet', 'IndustryCode', industry_filter_logic)

    # Print filtered data
    print("\nFiltered Data:")
    print(filtered_data)
