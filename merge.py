import pandas as pd

# Input and output file paths
input_file = "PCTES_ATENDIDOS_2024.xls"  # Replace with your Excel file
output_file = "PCTES_ATENDIDOS_2024.csv"  # Replace with your desired output CSV file

# Load the Excel file
excel_data = pd.ExcelFile(input_file)

# List to store dataframes
dfs = []

# Iterate through each sheet
for sheet_name in excel_data.sheet_names:
    print(f"Processing sheet: {sheet_name}")
    df = pd.read_excel(excel_data, sheet_name=sheet_name)
    dfs.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Save to a single CSV file
combined_df.to_csv(output_file, index=False)

print(f"All sheets have been combined and saved to {output_file}")
