import os

import nbformat
from nbconvert import PythonExporter

# Get the absolute path of the current script
current_file_path = os.path.abspath(__file__)
# Get the directory of the current script
current_directory = os.path.dirname(current_file_path)


# Specify the notebook file name
notebook_file = current_directory + "/heart_disease_datamining_project.ipynb"
output_file = current_directory + "/main.py"

# Load the notebook file
with open(notebook_file, "r", encoding="utf-8") as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert the notebook to Python script
python_exporter = PythonExporter()
python_code, _ = python_exporter.from_notebook_node(notebook_content)

# Save the Python code to a file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(python_code)

print(f"Converted {notebook_file} to {output_file}")
