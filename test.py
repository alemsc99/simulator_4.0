import pandas as pd
from tabulate import tabulate
# Read the CSV file into a DataFrame
df = pd.read_csv('./simulation_logs/accuracies_0.csv', index_col=0)

# Convert the DataFrame to a table format with the specified column order
table = df[['Client 0', 'Client 1', 'Client 2', 'Client 3', 'Client 4', 'Client 5', 'Client 6', 'Client 7', 'Client 8', 'Client 9', 'Accuracy Mean']]

# Save the table to a new CSV file
table.to_csv('table.csv', float_format='%.3f')

# Visualize the table
table_str = tabulate(table, headers='keys', tablefmt='grid')

# Save the visualized table to a text file
with open('visualized_table.txt', 'w') as file:
    file.write(table_str)

# Display the visualized table
print(table_str)