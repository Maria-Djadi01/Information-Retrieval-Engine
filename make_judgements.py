import pandas as pd
import re

input_file = "LISA_que_ref.txt"
csv_file = "judgment_file.csv"

# Read relevant refs from the text file
with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

# example of test:
# Query 1
# 2 Relevant Refs:
# 3392 3396 -1

# Query 2
# 2 Relevant Refs:
# 2623 4291 -1

list_query = []
list_docs = []
l = 0
for line in lines:
    line = line.strip()
    if line.startswith("Query"):
        query = int(line[-1])
        l = 0
    l += 1
    if l == 3:
        list_que_docs = line.split()[:-1]
        for doc in list_que_docs:
            list_docs.append(int(doc))
            list_query.append(query)

df = pd.DataFrame({"query_number": list_query, "document_number": list_docs})

df.to_csv(csv_file, index=False, header=True)
