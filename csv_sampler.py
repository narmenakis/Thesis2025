import pandas as pd

def filter_first_ten_entries_and_columns(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Select only the first 2 rows and the columns 'author', 'link', 'title', and 'text'
    # filtered_df = df[['author', 'link', 'title', 'text']].head(2)
    filtered_df = df.head(2)

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)

# Example usage
filter_first_ten_entries_and_columns("/Users/narmen/Downloads/kathimerini.gr.csv", "/Users/narmen/CEID/Διπλωματική/thesis/source/output_2.csv")
