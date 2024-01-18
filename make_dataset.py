import re
import os

def separate_documents(input_file):
    with open(input_file, 'r') as file:
        content = file.read()

    # Use regular expression to find and separate documents
    documents = re.split(r'\*{40,}', content)

    # Remove any leading or trailing whitespaces
    documents = [doc.strip() for doc in documents if doc.strip()]

    # Remove "Document" lines from the beginning of each document
    documents = [re.sub(r'^Document\s+\d+\s*', '', doc) for doc in documents]

    # Save each document into a separate file
    for i, doc in enumerate(documents, start=1):
        output_file = os.path.join("documents", f"D{i}.txt")
        with open(output_file, 'w') as output:
            output.write(doc)

if __name__ == "__main__":
    input_file = "LISA.txt"
    separate_documents(input_file)
    

