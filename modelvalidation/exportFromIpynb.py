from IPython .nbformat import current as nbformat
from IPython.nbconvert import PythonExporter
import os
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
# Create your views here.
user_name = "user1"
file_path = os.path.join(BASE_DIR, 'static\script_files\\')
filepath = 'ml_validation_atuo_part3.ipynb.ipynb'
export_path = 'ml_validation_atuo_part3.py'


def exporttoPY(request):
    with open(filepath) as fh:
        nb = nbformat.reads_json(fh.read())

    exporter = PythonExporter()

    # source is a tuple of python source code
    # meta contains metadata
    source, meta = exporter.from_notebook_node(nb)

    with open(export_path, 'w+') as fh:
        fh.writelines(source)
