print("DEBUG: Script upload.py está comenzando a ejecutarse.") # <-- MENSAJE DE PRUEBA

from datasets import Dataset, Audio # Asegúrate de haber hecho pip install datasets
from pathlib import Path
import os
import argparse
from getpass import getpass # Para la entrada oculta del token

print("DEBUG: Importaciones completadas.") # <-- MENSAJE DE PRUEBA

def load_dataset_from_folder(folder_path):
    print(f"DEBUG: load_dataset_from_folder: Cargando desde {folder_path}") # <-- MENSAJE DE PRUEBA
    folder = Path(folder_path)
    entries = []

    for audio_file in sorted(folder.glob("*.wav")):
        txt_file = audio_file.with_suffix(".txt")
        if txt_file.exists():
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            entries.append({"audio": str(audio_file.resolve()), "text": text})
    
    if not entries:
        print(f"DEBUG: load_dataset_from_folder: No se encontraron pares .wav/.txt en {folder_path}") # <-- MENSAJE DE PRUEBA

    dataset = Dataset.from_list(entries)
    dataset = dataset.cast_column("audio", Audio())
    print("DEBUG: load_dataset_from_folder: Dataset creado.") # <-- MENSAJE DE PRUEBA
    return dataset

def push_to_hub(dataset, repo_name, token=None):
    print(f"DEBUG: push_to_hub: Subiendo a {repo_name}") # <-- MENSAJE DE PRUEBA
    try:
        dataset.push_to_hub(repo_name, token=token)
        print(f"✅ Dataset subido: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        print(f"DEBUG: push_to_hub: ERROR al subir - {e}") # <-- MENSAJE DE PRUEBA
        raise # Volver a lanzar la excepción para ver el traceback completo

def main():
    print("DEBUG: Función main() ha comenzado.") # <-- MENSAJE DE PRUEBA
    parser = argparse.ArgumentParser(description="Sube un dataset de audio y texto a Hugging Face Hub.")
    parser.add_argument("folder", help="Carpeta que contiene los pares .mp3 + .txt")
    parser.add_argument("--repo_name", help="Nombre del repo en HuggingFace (e.g. usuario/nombre_dataset). Si no se especifica, se preguntará.", default=None)
    parser.add_argument("--token", help="Token de Hugging Face. Si no se especifica, se preguntará (opcional si ya hiciste 'huggingface-cli login').", default=None)

    print("DEBUG: ArgumentParser configurado. Llamando a parser.parse_args().") # <-- MENSAJE DE PRUEBA
    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(f"DEBUG: parser.parse_args() causó SystemExit: {e}") # <-- MENSAJE DE PRUEBA
        # argparse llama a sys.exit() en caso de error, lo que podría ser "silencioso" en algunos contextos.
        # No relanzamos para ver si podemos imprimir algo después.
        return # Salimos de main si hay error de argumentos

    print(f"DEBUG: Argumentos parseados: folder='{args.folder}', repo_name='{args.repo_name}', token_is_set='{args.token is not None}'") # <-- MENSAJE DE PRUEBA

    repo_name_to_use = args.repo_name
    print(f"DEBUG: repo_name_to_use inicial: {repo_name_to_use}") # <-- MENSAJE DE PRUEBA
    if repo_name_to_use is None:
        print("DEBUG: repo_name_to_use es None. Solicitando input para el nombre del repo.") # <-- MENSAJE DE PRUEBA
        try:
            repo_name_to_use = input("🏷️ Introduce el nombre para el repositorio en HuggingFace (e.g. tu_usuario/nombre_del_dataset): ")
            while not repo_name_to_use:
                print("❌ El nombre del repositorio no puede estar vacío.") # Esto debería aparecer si se da Enter sin texto
                repo_name_to_use = input("🏷️ Introduce el nombre para el repositorio en HuggingFace (e.g. tu_usuario/nombre_del_dataset): ")
        except EOFError:
            print("DEBUG: EOFError al intentar leer el nombre del repo. ¿Se está ejecutando sin una terminal interactiva?") # <-- MENSAJE DE PRUEBA
            return
        except Exception as e:
            print(f"DEBUG: Excepción al solicitar input para repo_name: {e}") # <-- MENSAJE DE PRUEBA
            return

    print(f"DEBUG: repo_name_to_use final: {repo_name_to_use}") # <-- MENSAJE DE PRUEBA

    token_to_use = args.token
    print(f"DEBUG: token_to_use inicial (¿suministrado por argumento?): {'Sí' if token_to_use else 'No'}") # <-- MENSAJE DE PRUEBA
    if token_to_use is None:
        print("DEBUG: token_to_use es None. Solicitando input con getpass para el token.") # <-- MENSAJE DE PRUEBA
        print("🔑 Introduce tu token de Hugging Face. Puedes obtener uno de https://huggingface.co/settings/tokens")
        print("(Si ya has iniciado sesión con 'huggingface-cli login' y tu token tiene permisos de escritura, puedes presionar Enter para omitirlo)")
        try:
            token_to_use = getpass("Token: ") 
            if not token_to_use:
                token_to_use = None
        except EOFError:
            print("DEBUG: EOFError al intentar leer el token. ¿Se está ejecutando sin una terminal interactiva?") # <-- MENSAJE DE PRUEBA
            return
        except Exception as e:
            print(f"DEBUG: Excepción al solicitar input para token con getpass: {e}") # <-- MENSAJE DE PRUEBA
            return
            
    token_length = len(token_to_use) if token_to_use else "No suministrado"
    print(f"DEBUG: token_to_use final (longitud): {token_length}") # <-- MENSAJE DE PRUEBA

    print(f"DEBUG: Llamando a load_dataset_from_folder con: {args.folder}") # <-- MENSAJE DE PRUEBA
    dataset = load_dataset_from_folder(args.folder)
    print("DEBUG: Dataset procesado (puede estar vacío si no se encontraron archivos):") # <-- MENSAJE DE PRUEBA
    print(dataset)

    if not dataset: # Si el dataset está vacío
        print("DEBUG: El dataset está vacío. No se subirá nada a Hugging Face.")
        return

    print(f"DEBUG: Llamando a push_to_hub con repo: {repo_name_to_use}") # <-- MENSAJE DE PRUEBA
    push_to_hub(dataset, repo_name_to_use, token_to_use)
    print("DEBUG: push_to_hub ha finalizado.") # <-- MENSAJE DE PRUEBA


if __name__ == "__main__":
    print("DEBUG: Bloque __main__ alcanzado. Llamando a main().") # <-- MENSAJE DE PRUEBA
    main()
    print("DEBUG: main() ha finalizado y el script va a terminar.") # <-- MENSAJE DE PRUEBA
else:
    # Esto se imprimiría si el script es importado por otro script, no ejecutado directamente
    print(f"DEBUG: Script importado como módulo, __name__ es {__name__}")
