"""
Arrange the original dataset for our task:
1) Split the dataset into training, validation (optional) and test sets.
2) Create and save the metadata, to be used in the Dataset class.
"""

import os
import csv
import random

# Define the dataset directory and target instruments
dataset_dir = "all-samples"
target_instruments = {"guitar", "flute", "violin", "clarinet"}
train_csv = "train_metadata.csv"
test_csv = "test_metadata.csv"
split_ratio = 0.7  # 70% training, 30% testing

# Collect metadata
train_metadata = []
test_metadata = []

random.seed(42)  # Ensure reproducibility

for instrument in target_instruments:
    instrument_dir = os.path.join(dataset_dir, instrument)
    if os.path.isdir(instrument_dir):
        files = [os.path.join(instrument_dir, file) for file in os.listdir(instrument_dir) if file.endswith(".mp3")]
        random.shuffle(files)  # Shuffle the files

        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        # Append to metadata lists
        #train_metadata.extend([[file, instrument] for file in train_files])
        train_metadata.extend([[file.replace("\\", "/"), instrument] for file in train_files])
        #test_metadata.extend([[file, instrument] for file in test_files])
        test_metadata.extend([[file.replace("\\", "/"), instrument] for file in test_files])


# Function to write metadata to CSV
def write_metadata(file_path, metadata):
    with open(file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "instrument"])  # Header
        writer.writerows(metadata)

# Save train and test metadata
write_metadata(train_csv, train_metadata)
write_metadata(test_csv, test_metadata)

print(f"Metadata saved to {train_csv} with {len(train_metadata)} entries.")
print(f"Metadata saved to {test_csv} with {len(test_metadata)} entries.")