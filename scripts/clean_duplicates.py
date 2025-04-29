# scripts/clean_duplicates.py

import shutil
from pathlib import Path

def clean_duplicate_outputs():
    base_output = Path('output')
    
    # Define mappings
    duplicate_folders = [
        {
            'source': base_output / 'models' / 'predictions',
            'target': base_output / 'predictions'
        },
        {
            'source': base_output / 'models' / 'metrics',
            'target': base_output / 'model_metrics'
        }
    ]
    
    for folder_pair in duplicate_folders:
        src, tgt = folder_pair['source'], folder_pair['target']
        if not src.exists():
            print(f"[INFO] Skipping {src} (does not exist)")
            continue
        
        if not tgt.exists():
            tgt.mkdir(parents=True, exist_ok=True)
        
        # Move files if not already in target
        for file_path in src.glob('*'):
            target_path = tgt / file_path.name
            if not target_path.exists():
                shutil.move(str(file_path), str(target_path))
                print(f"[MOVED] {file_path.name} -> {tgt}")
            else:
                print(f"[SKIPPED] {file_path.name} already exists in {tgt}")

        # Remove source folder if empty
        try:
            src.rmdir()
            print(f"[CLEANUP] Removed empty directory {src}")
        except OSError:
            print(f"[INFO] Directory {src} not empty, skipped removal.")

if __name__ == "__main__":
    clean_duplicate_outputs()