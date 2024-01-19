import pandas as pd

input_file = "LISA_que.txt"
csv_file = "queries.csv"

# Read queries from the text file
with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Process the lines to create a list of queries
queries = []
current_index = None
current_query = ""

for line in lines:
    line = line.strip()
    if line and line[0].isdigit():
        if current_query:
            queries.append(current_query.strip("# "))
        current_query = ""
    else:
        current_query += line + " "

# Add the last query
if current_query:
    queries.append(current_query.strip("# "))

# Create a DataFrame using pandas
df = pd.DataFrame({'query': queries})

# Write the DataFrame to a CSV file without an index column
df.to_csv(csv_file, index=False, header=True)
