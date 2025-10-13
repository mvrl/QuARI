import os
from huggingface_hub import hf_hub_download, HfApi

MODEL_REPO_MAP = {
    "clip-vit-base-patch16": "EricX003/QuARI-clip-vit-base-patch16",
    "clip-vit-large-patch14": "EricX003/QuARI-clip-vit-large-patch14",
    "siglip2-base-patch16-512": "EricX003/QuARI-google-siglip2-base-patch16-512",
    "siglip2-large-patch16-512": "EricX003/QuARI-google-siglip2-large-patch16-512",
}

def download_repo_files(repo_id, local_dir, api):
    """Download all files from a HuggingFace repo maintaining directory structure."""
    try:
        # List all files in the repository
        print(f"  Fetching file list from {repo_id}...")
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")

        print(f"  Found {len(repo_files)} file(s)")

        for file_path in repo_files:
            if file_path.startswith('.git') or file_path.startswith('README.md'):
                continue

            print(f"    Downloading: {file_path}")

            # Download file to cache
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="model"
            )

            # Create target path maintaining directory structure
            target_path = os.path.join(local_dir, file_path)
            target_dir = os.path.dirname(target_path)

            # Create directories if they don't exist
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)

            import shutil
            shutil.copy2(downloaded_path, target_path)

        print(f"Successfully downloaded all files from {repo_id}")
        return True

    except Exception as e:
        print(f"Error downloading from {repo_id}: {e}")
        return False

def main():
    """Download all checkpoints from HuggingFace repos."""
    ckpts_dir = "ckpts/"

    # Create base ckpts directory if it doesn't exist
    os.makedirs(ckpts_dir, exist_ok=True)
    api = HfApi()

    print("="*30)
    print("Downloading checkpoints from HuggingFace")
    print("="*30)

    success_count = 0

    # Iterate through each model mapping
    for model_dir, repo_id in MODEL_REPO_MAP.items():
        local_path = os.path.join(ckpts_dir, model_dir)

        print(f"\n{'='*60}")
        print(f"Downloading {repo_id} -> {local_path}")
        print(f"{'='*60}")

        # Create local directory
        os.makedirs(local_path, exist_ok=True)

        # Download all files from the repo
        if download_repo_files(repo_id, local_path, api):
            success_count += 1
        else:
            print(f"Failed to download {repo_id}")


if __name__ == "__main__":
    main()
