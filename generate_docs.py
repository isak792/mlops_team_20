# generate_docs.py
import os
import sys
import subprocess
from pathlib import Path

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_docs():
    # Asegúrate de que el directorio de documentación existe
    docs_dir = "docs/api"
    ensure_dir(docs_dir)
    
    # Añade el directorio actual al PYTHONPATH
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Comando para generar la documentación
    cmd = [
        "pdoc",
        "--output-dir", docs_dir,
        "--docformat", "google",
        "--template-dir", ".",
        "turkish_music_emotion"
    ]
    
    # Ejecuta pdoc
    try:
        print("Executing command:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("Documentation generated successfully in", docs_dir)
        
    except subprocess.CalledProcessError as e:
        print("Error generating documentation:", e)
        print("Try running the following command directly:")
        print(f"PYTHONPATH={project_root} pdoc --output-dir {docs_dir} --docformat google turkish_music_emotion")

if __name__ == "__main__":
    generate_docs()
