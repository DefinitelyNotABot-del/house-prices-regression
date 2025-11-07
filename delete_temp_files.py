import os
import sys

# Define the folders and the files to delete within them
files_to_delete_map = {
    'saved_models': [
        'train_medians.joblib',
        'feature_names.joblib',
        'scaler.joblib',
        'lasso_model.joblib',
        'ridge_model.joblib',
        'elasticnet_model.joblib'
    ],
    'submissions': [
        'submission_ols.csv',
        'submission_rlm.csv',
        'submission_glm.csv',
        'submission_lasso.csv',
        'submission_ridge.csv',
        'submission_elasticnet.csv'
    ]
}

print("Attempting to delete files from specified folders...")
print("The folders 'saved_models' and 'submissions' will NOT be deleted, only their contents.")

deleted_count = 0
not_found_count = 0
error_count = 0

# Get the directory where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Loop through each folder
for folder_name, file_list in files_to_delete_map.items():
    folder_path = os.path.join(current_directory, folder_name)
    print(f"\n--- Checking in folder: '{folder_name}/' ---")

    # Check if the folder itself exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: '{folder_name}'. Skipping...")
        not_found_count += len(file_list)
        continue

    # Loop through each file in the list for that folder
    for filename in file_list:
        file_path = os.path.join(folder_path, filename) # Construct full path
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ Successfully deleted: {filename}")
                deleted_count += 1
            else:
                print(f"❌ File not found (already deleted?): {filename}")
                not_found_count += 1
        except OSError as e:
            print(f"⚠️ Error deleting file {filename}: {e}")
            error_count += 1
        except Exception as e:
            print(f"🚨 An unexpected error occurred with {filename}: {e}")
            error_count += 1

print("\n--- Summary ---")
print(f"Files successfully deleted: {deleted_count}")
print(f"Files not found: {not_found_count}")
print(f"Errors encountered: {error_count}")
print("Deletion complete. Folders have been kept.")