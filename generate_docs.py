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
        "--html",
        "--output-dir", docs_dir,
        "--force",
        "turkish_music_emotion"
    ]
    
    # Ejecuta pdoc
    try:
        subprocess.run(cmd, check=True)
        
        # Convierte los archivos HTML a Markdown
        for html_file in Path(docs_dir).rglob("*.html"):
            # Lee el contenido HTML
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Crea el archivo Markdown correspondiente
            md_file = html_file.with_suffix('.md')
            with open(md_file, 'w', encoding='utf-8') as f:
                # Extrae el contenido relevante y dale formato Markdown
                # Esto es una simplificación - podrías querer usar una biblioteca HTML parser
                content = content.replace('<pre><code>', '```python\n')
                content = content.replace('</code></pre>', '\n```')
                f.write(content)
            
            # Elimina el archivo HTML
            html_file.unlink()
            
        print("Documentation generated successfully in", docs_dir)
        
    except subprocess.CalledProcessError as e:
        print("Error generating documentation:", e)

if __name__ == "__main__":
    generate_docs()
