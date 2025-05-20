import os
import shutil
import random

# Source folder
src_folder = 'default_setup_transfer_cleaningstock/stock_data'
# Destination folder
dst_folder = 'DiffPrep_transfer_cleaning/data'

# Get all files ending with _historical_data.csv
files = [f for f in os.listdir(src_folder) if f.endswith('_historical_data.csv')]

# Randomly select 1000 files (or all if fewer than 1000)
files = random.sample(files, min(1000, len(files)))

for file_name in files:
    # Extract stock code (before the first underscore)
    stock_code = file_name.split('_')[0]
    
    # Create new folder path
    new_folder = os.path.join(dst_folder, f"{stock_code}_data")
    os.makedirs(new_folder, exist_ok=True)
    
    # Define source file path
    src_file = os.path.join(src_folder, file_name)
    
    # Define destination file path
    dst_file = os.path.join(new_folder, 'data.csv')
    
    # Move and rename the file
    shutil.move(src_file, dst_file)
    
    print(f"Moved {file_name} to {dst_file}")

print("✅ Done moving 1000 random files.")
