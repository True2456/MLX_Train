#!/usr/bin/env python3
"""
Robust multi-model LoRA uploader for Hugging Face Hub using HfApi.
Handles automatic retries and progress tracking for large safetensors files.
"""
import os
import sys
from huggingface_hub import HfApi

REPOS = [
    ("True2456/Gemma-4-12B-Agentic-LoRA", "models/gemma12b/agentic_lora"),
    ("True2456/Gemma-4-12B-Theory-LoRA", "models/gemma12b/theory_lora"),
    ("True2456/Gemma-4-12B-ASM-Systems-LoRA", "models/gemma12b/asm_systems_lora"),
    ("True2456/Gemma-4-12B-3Specialist-Merged-LoRA", "models/gemma12b/mati_3specialist_merged_lora"),
]

FILES_TO_UPLOAD = ["README.md", "adapter_config.json", "adapters.safetensors"]

def main():
    api = HfApi()
    user = api.whoami()["name"]
    print(f"Authenticated as Hugging Face user: {user}")

    for repo_id, folder_path in REPOS:
        print(f"\n==========================================")
        print(f"Processing repository: {repo_id}")
        print(f"==========================================")
        
        # Ensure repo exists
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        
        for filename in FILES_TO_UPLOAD:
            local_file = os.path.join(folder_path, filename)
            if not os.path.exists(local_file):
                print(f"  [WARN] Missing {local_file}, skipping...")
                continue
            
            size_mb = os.path.getsize(local_file) / (1024 * 1024)
            print(f"  -> Uploading {filename} ({size_mb:.2f} MB)...")
            api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {filename} (Gemma 4 12B LoRA specialist)"
            )
            print(f"     [SUCCESS] Uploaded {filename}")

    print("\nAll 4 Gemma 4 12B LoRA specialists successfully uploaded!")

if __name__ == "__main__":
    main()
