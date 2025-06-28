
import nbformat

def extract_code_from_py(content: bytes) -> str:
    return content.decode("utf-8")

def extract_code_from_ipynb(content: bytes) -> str:
    notebook = nbformat.reads(content.decode("utf-8"), as_version=4)
    code_cells = [cell['source'] for cell in notebook.cells if cell['cell_type'] == 'code']
    return "\n\n".join(code_cells)
