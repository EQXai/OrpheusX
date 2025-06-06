from datasets import Dataset, Audio
from pathlib import Path
import os
import argparse
from getpass import getpass

def load_dataset_from_folder(folder_path):
    folder = Path(folder_path)
    entries = []

    for audio_file in sorted(folder.glob("*.wav")):
        txt_file = audio_file.with_suffix(".txt")
        if txt_file.exists():
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            entries.append({"audio": str(audio_file.resolve()), "text": text})
    
    dataset = Dataset.from_list(entries)
    dataset = dataset.cast_column("audio", Audio())
    return dataset

def push_to_hub(dataset, repo_name, token=None):
    try:
        dataset.push_to_hub(repo_name, token=token)
        print(f"✅ Dataset uploaded: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        raise

def main():
    parser = argparse.ArgumentParser(description="Upload an audio and text dataset to Hugging Face Hub.")
    parser.add_argument("folder", help="Folder containing .wav + .txt pairs")
    parser.add_argument("--repo_name", help="Repository name on HuggingFace (e.g. user/dataset_name). If not specified, it will be prompted.", default=None)
    parser.add_argument("--token", help="Hugging Face token. If not specified, it will be prompted (optional if you have already run 'huggingface-cli login').", default=None)

    args = parser.parse_args()

    repo_name_to_use = args.repo_name
    if repo_name_to_use is None:
        try:
            repo_name_to_use = input("🏷️ Enter the repository name for HuggingFace (e.g. your_username/your_dataset_name): ")
            while not repo_name_to_use:
                print("❌ The repository name cannot be empty.")
                repo_name_to_use = input("🏷️ Enter the repository name for HuggingFace (e.g. your_username/your_dataset_name): ")
        except EOFError:
            return
        except Exception:
            return

    token_to_use = args.token
    if token_to_use is None:
        print("🔑 Enter your Hugging Face token. You can get one from https://huggingface.co/settings/tokens")
        print("(If you have already logged in with 'huggingface-cli login' and your token has write permissions, you can press Enter to skip)")
        try:
            token_to_use = getpass("Token: ") 
            if not token_to_use:
                token_to_use = None
        except EOFError:
            return
        except Exception:
            return
            
    dataset = load_dataset_from_folder(args.folder)

    if not dataset:
        return

    push_to_hub(dataset, repo_name_to_use, token_to_use)


if __name__ == "__main__":
    main()
