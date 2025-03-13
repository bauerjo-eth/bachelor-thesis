import pandas as pd
import numpy as np

# Input and output CSV file paths.
input_csv = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_labels\labels.csv"         # Original CSV with columns: Filename, P1_x, P1_y, P1_z, …, P6_x, P6_y, P6_z
output_csv = r"C:\Dokumente\Studium\6. Semester\Bachelorarbeit\Code\smplx_big_test\smplx_labels\labels_reordered.csv"  # New CSV to be saved

# Read the original CSV.
df = pd.read_csv(input_csv)

# Rename "Filename" to "filename" for consistency.
df.rename(columns={"Filename": "filename"}, inplace=True)

# Assume the six points are labeled as P1_x, P1_y, P1_z, ..., P6_x, P6_y, P6_z.
point_columns = []
for i in range(1, 7):
    point_columns += [f"P{i}_x", f"P{i}_y", f"P{i}_z"]

def reorder_points(row):
    # Extract the 6 points into a (6,3) array.
    points = row[point_columns].values.reshape(6, 3)
    # Sort indices based on x-coordinate.
    sorted_by_x = np.argsort(points[:, 0])
    # Define groups:
    left_indices = sorted_by_x[:2]    # two smallest x values (left)
    right_indices = sorted_by_x[-2:]  # two largest x values (right)
    middle_indices = sorted_by_x[2:4]  # the remaining two (middle)
    
    left_group = points[left_indices]
    right_group = points[right_indices]
    middle_group = points[middle_indices]
    
    # Within each group, sort by z-coordinate (descending: higher z = top)
    left_sorted = left_group[np.argsort(-left_group[:, 2])]
    right_sorted = right_group[np.argsort(-right_group[:, 2])]
    middle_sorted = middle_group[np.argsort(-middle_group[:, 2])]
    
    # Assign anatomical labels based on these sorted groups:
    # Right chest: right group, higher z.
    right_chest = right_sorted[0]
    # Right side of upper-body: right group, lower z.
    right_side = right_sorted[1]
    # Left chest: left group, higher z.
    left_chest = left_sorted[0]
    # Left side of upper-body: left group, lower z.
    left_side = left_sorted[1]
    # Middle above the lungs: middle group, higher z.
    middle_above = middle_sorted[0]
    # Below the belly button: middle group, lower z.
    below_belly = middle_sorted[1]
    
    # Concatenate in the new order:
    new_order = np.concatenate([right_chest, left_chest, right_side, left_side, middle_above, below_belly])
    return new_order

# Apply reordering to each row.
new_points = df.apply(reorder_points, axis=1, result_type='expand')

# Define new column names.
new_columns = [
    "right_chest_x", "right_chest_y", "right_chest_z",
    "left_chest_x", "left_chest_y", "left_chest_z",
    "right_side_x", "right_side_y", "right_side_z",
    "left_side_x", "left_side_y", "left_side_z",
    "middle_above_lungs_x", "middle_above_lungs_y", "middle_above_lungs_z",
    "below_belly_x", "below_belly_y", "below_belly_z"
]
new_points.columns = new_columns

# Create a new DataFrame with filename and the reordered labels.
df_reordered = pd.concat([df[["filename"]], new_points], axis=1)

# Save the new DataFrame to CSV.
df_reordered.to_csv(output_csv, index=False)
print("Reordered labels saved to", output_csv)
