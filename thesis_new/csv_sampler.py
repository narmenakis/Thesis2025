import pandas as pd

def filter_first_ten_entries_and_columns(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Print the total number of entries in the dataset
    total_entries = len(df)
    # print(f"Total number of entries in the dataset: {total_entries}")

    filtered_df = df.head(20)

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered data saved to {output_csv}")

# Example usage
filter_first_ten_entries_and_columns("kathimerini.gr.csv", "output.csv")
