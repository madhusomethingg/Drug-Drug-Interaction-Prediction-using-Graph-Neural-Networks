import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

print('starting execution')

in_path = "/Users/madhuu/Desktop/Drug Drug/Drug_example.ipynb"
out_path = "/Users/madhuu/Desktop/Drug Drug/Drug_example.out.ipynb"

nb = nbformat.read(in_path, as_version=4)
client = NotebookClient(nb, timeout=1200, kernel_name="python3", allow_errors=False)
try:
    client.execute()
except CellExecutionError as e:
    print("Execution failed:", e)
    raise

nbformat.write(nb, out_path)
print("done writing", out_path)
