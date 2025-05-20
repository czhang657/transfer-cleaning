import os
import json
from collections import Counter

# Define path to result directory
result_dir = "result/stock_runs/diffprep_fix"

# Initialize counters
num_tf_counter = Counter()
num_n_counter = Counter()
unique_pipeline_counter = Counter()
single_column_pair_counter = Counter()  # NEW: count individual (num_tf, normalization) column pairs

# Iterate through each stock result folder
for stock_folder in os.listdir(result_dir):
    pipeline_path = os.path.join(result_dir, stock_folder, "pipeline.json")
    if os.path.exists(pipeline_path):
        with open(pipeline_path, "r") as f:
            data = json.load(f)
            num_tf = data.get("num_tf", [])
            normalization = data.get("normalization", [])

            # Update individual method counters
            num_tf_counter.update(num_tf)
            num_n_counter.update(normalization)

            # Update unique pipeline counter if both lists are valid and equal length
            if num_tf and normalization and len(num_tf) == len(normalization):
                pipeline_signature = tuple(zip(num_tf, normalization))
                unique_pipeline_counter[pipeline_signature] += 1

                # NEW: update individual column-level combination counter
                for pair in zip(num_tf, normalization):
                    single_column_pair_counter[pair] += 1

# Print results
print("num_tf counts:")
for method, count in num_tf_counter.items():
    print(f"{method}: {count}")

print("\nnormalization counts:")
for method, count in num_n_counter.items():
    print(f"{method}: {count}")

print(f"\nNumber of unique pipelines: {len(unique_pipeline_counter)}")

# Print top 5 most frequent pipelines
print("\nTop 5 most frequent full pipelines:")
for pipeline, count in unique_pipeline_counter.most_common(5):
    print(f"Pipeline: {pipeline} | Count: {count}")

# NEW: Print top 5 most frequent (num_tf, normalization) pairs
print("\nTop 5 most frequent (num_tf, normalization) column pairs:")
for pair, count in single_column_pair_counter.most_common(5):
    print(f"Pair: {pair} | Count: {count}")
