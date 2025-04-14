# """
# Arrange the original dataset for our task:
# 1) Split the dataset into training, validation (optional) and test sets.
# 2) Create and save the metadata, to be used in the Dataset class.
# """

# import os
# import csv
# import random

# # Define the dataset directory and target instruments
# dataset_dir = "all-samples"
# target_instruments = {"guitar", "flute", "violin", "clarinet", " trumpet", " cello", "saxophone"}
# train_csv = "train_metadata.csv"
# test_csv = "test_metadata.csv"
# split_ratio = 0.7  # 70% training, 30% testing

# # Collect metadata
# train_metadata = []
# test_metadata = []

# random.seed(42)  # Ensure reproducibility

# for instrument in target_instruments:
#     instrument_dir = os.path.join(dataset_dir, instrument)
#     if os.path.isdir(instrument_dir):
#         files = [os.path.join(instrument_dir, file) for file in os.listdir(instrument_dir) if file.endswith(".mp3")]
#         random.shuffle(files)  # Shuffle the files

#         split_idx = int(len(files) * split_ratio)
#         train_files = files[:split_idx]
#         test_files = files[split_idx:]

#         # Append to metadata lists
#         #train_metadata.extend([[file, instrument] for file in train_files])
#         train_metadata.extend([[file.replace("\\", "/"), instrument] for file in train_files])
#         #test_metadata.extend([[file, instrument] for file in test_files])
#         test_metadata.extend([[file.replace("\\", "/"), instrument] for file in test_files])


# # Function to write metadata to CSV
# def write_metadata(file_path, metadata):
#     with open(file_path, mode="w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["audio_path", "instrument"])  # Header
#         writer.writerows(metadata)

# # Save train and test metadata
# write_metadata(train_csv, train_metadata)
# write_metadata(test_csv, test_metadata)

# print(f"Metadata saved to {train_csv} with {len(train_metadata)} entries.")
# print(f"Metadata saved to {test_csv} with {len(test_metadata)} entries.")

import os
import pandas as pd
import random

random.seed(42)  # 保持可复现性

# 假设你原来的结构是 all-samples/instrument_name/*.mp3
base_dir = "all-samples"
instruments = sorted(os.listdir(base_dir))  # 自动识别类别
train_ratio = 0.7

train_data = []
test_data = []

for instrument in instruments:
    folder = os.path.join(base_dir, instrument)
    if not os.path.isdir(folder):
        continue
    files = [f for f in os.listdir(folder) if f.endswith(".mp3")]
    if len(files) < 2:
        continue  # 跳过太少样本的类

    random.shuffle(files)
    split_idx = max(1, int(len(files) * train_ratio))  # 至少 1 个进 test
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    train_data.extend([(os.path.join(folder, f), instrument) for f in train_files])
    test_data.extend([(os.path.join(folder, f), instrument) for f in test_files])

# 保存为 CSV
pd.DataFrame(train_data, columns=["audio_path", "instrument"]).to_csv("train_metadata.csv", index=False)
pd.DataFrame(test_data, columns=["audio_path", "instrument"]).to_csv("test_metadata.csv", index=False)

print("Done! Train size =", len(train_data), ", Test size =", len(test_data))
