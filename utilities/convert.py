import json
import csv
import sys
import pandas as pd

def convert_json_to_csv(file_prefix):
    json_path = f'/home/aoneill/train_inat/{file_prefix}.json'
    csv_path = f'/home/aoneill/train_inat/{file_prefix}.csv'

    # Load JSON data
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Targeting the 'images' list
    images = data['images']

    # Create CSV file
    with open(csv_path, 'w', newline='') as file:
        csv_file = csv.writer(file)
        # Write CSV headers (keys from the first image object)
        csv_file.writerow(images[0].keys())
        # Write CSV data
        for image in images:
            csv_file.writerow(image.values())

    return csv_path  # Return the path of the created CSV file

def format_genus(file_name):
    taxonomy_with_id = file_name.split('/')[1]  # Get the taxonomy part
    taxonomy = '_'.join(taxonomy_with_id.split('_')[1:])  # Remove the ID and rejoin
    return taxonomy_with_id

def extract_internal_id(file_name):
    # Extracts the UUID from the filename, which is assumed to be the last element before the file extension
    return file_name.split('/')[-1].split('.')[0]

def clean_data(df):
    # Replace NaN values in 'longitude' and 'latitude' with 0
    df['longitude'] = df['longitude'].fillna(0)
    df['latitude'] = df['latitude'].fillna(0)
    return df

# Example usage: python convert.py train_mini
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert.py [file_prefix]")
        sys.exit(1)

    file_prefix = sys.argv[1]
    csv_path = convert_json_to_csv(file_prefix)  # Get the path of the created CSV

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Clean data for NaN values in longitude and latitude
    df = clean_data(df)

    # Apply functions to the 'file_name' column and create new columns
    df['tree/genus/genus'] = df['file_name'].apply(format_genus)
    df['internal_id'] = df['file_name'].apply(extract_internal_id)

    # Save the modified DataFrame to a new CSV file
    modified_csv_path = csv_path.replace('.csv', '_modified.csv')
    df.to_csv(modified_csv_path, index=False)

    # Print the first few rows to verify
    print(df[['file_name', 'tree/genus/genus', 'internal_id', 'longitude', 'latitude']].head())
