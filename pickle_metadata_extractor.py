import os
import pickle
import json
import pandas as pd
from pathlib import Path



# "scans-20250225T092603Z-001/scans/" directory with a naming pattern of "metadata_[ID][number][index].p".


# Configuration
input_folder = "scans-20250225T092603Z-001/scans"
output_folder = "metadata_extracts"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Lists to store all metadata
all_metadata = []
file_paths = []

# Function to recursively find all .p files
def find_pickle_files(directory):
    pickle_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".p") and file.startswith("metadata_"):
                pickle_files.append(os.path.join(root, file))
    return pickle_files

# Find all pickle files
pickle_files = find_pickle_files(input_folder)
print(f"Found {len(pickle_files)} pickle files")

# Process each pickle file
for file_path in pickle_files:
    try:
        # Extract file information
        file_name = os.path.basename(file_path)
        parent_folder = os.path.basename(os.path.dirname(file_path))
        
        # Load the pickle file
        with open(file_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Add file path information to metadata
        metadata_with_source = metadata.copy()
        metadata_with_source['source_file'] = file_name
        metadata_with_source['parent_folder'] = parent_folder
        
        # Store the metadata
        all_metadata.append(metadata_with_source)
        file_paths.append(file_path)
        
        # Save individual JSON file
        output_filename = os.path.splitext(file_name)[0] + ".json"
        output_path = os.path.join(output_folder, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
            
        print(f"Processed: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Save consolidated metadata to CSV and Excel
if all_metadata:
    try:
        # Create a dataframe from all metadata
        df = pd.DataFrame(all_metadata)
        
        # Save to CSV
        csv_path = os.path.join(output_folder, "all_metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved consolidated metadata to {csv_path}")
        
        # Save to Excel
        excel_path = os.path.join(output_folder, "all_metadata.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"Saved consolidated metadata to {excel_path}")
        
        # Save summary information
        summary = {
            "total_files_processed": len(all_metadata),
            "file_paths": file_paths
        }
        
        with open(os.path.join(output_folder, "processing_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4, default=str)
            
    except Exception as e:
        print(f"Error creating consolidated files: {e}")

print("Processing complete! All metadata extracted.")