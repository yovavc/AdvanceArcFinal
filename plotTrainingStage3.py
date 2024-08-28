# import matplotlib.pyplot as plt
#
# # Data provided
# start_filters = [1, 2, 4, 8]
# max_test_accuracy = [0.639073149, 0.778736938, 0.906678782, 0.888868696]
#
# # Plotting Max Test Accuracy vs Start Filters
# plt.figure(figsize=(10, 6))
# plt.plot(start_filters, max_test_accuracy, marker='o', linestyle='-', color='blue')
#
# plt.title('Max Test Accuracy vs Start Filters')
# plt.xlabel('Start Filters')
# plt.ylabel('Max Test Accuracy (Max)')
# plt.grid(True)
# plt.show()
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
#
# # Load the CSV file
# file_path = 'wandb_export_2024-08-25T21_09_22.880+03_00.csv'
# data = pd.read_csv(file_path)
#
# # Extract the unique values of freq_mask_param
# unique_freq_mask_param = sorted(data['freq_mask_param'].unique())
#
# # Set up the layout for the plots
# n_cols = 2
# n_rows = len(unique_freq_mask_param) // n_cols + int(len(unique_freq_mask_param) % n_cols > 0)
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, n_rows * 4))
#
# # Loop over each unique freq_mask_param value to create a heatmap
# for i, freq_param in enumerate(unique_freq_mask_param):
#     ax = axes[i // n_cols, i % n_cols]
#
#     # Filter the data for the current freq_mask_param
#     subset_data = data[data['freq_mask_param'] == freq_param]
#
#     # Create a density plot
#     sns.kdeplot(
#         data=subset_data,
#         x='time_mask_param',
#         y='Max Test Accuracy',
#         cmap="viridis",
#         fill=True,
#         ax=ax
#     )
#
#     # Set plot labels and title
#     ax.set_title(f'freq_mask_param = {freq_param}')
#     ax.set_xlabel('time_mask_param')
#     ax.set_ylabel('Max Test Accuracy')
#     ax.grid(True)
#
# # Adjust layout
# plt.tight_layout()
#
# # Show the plot
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Section to read the CSV file
file_path = 'wandb_export_2024-08-25T21_09_22.880+03_00.csv'
data = pd.read_csv(file_path)

# Unique values of time_mask_param
time_mask_values = data['time_mask_param'].unique()

# Set up the figure for subplots in a 2x3 grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Generate density plots
for i, time_mask_param in enumerate(time_mask_values[:6]):  # Adjusted to handle up to 6 plots
    subset_data = data[data['time_mask_param'] == time_mask_param]
    sns.kdeplot(
        data=subset_data,
        x='freq_mask_param',
        y='Max Test Accuracy',
        cmap="coolwarm",
        fill=True,
        ax=axes[i]
    )
    axes[i].set_title(f'Density Plot for time_mask_param = {time_mask_param}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('freq_mask_param', fontsize=12)
    axes[i].set_ylabel('Max Test Accuracy', fontsize=12)
    axes[i].grid(True)
# Remove any unused subplots (if there are fewer than 6 time_mask_param values)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout for better presentation in an article
plt.tight_layout()
plt.show()
