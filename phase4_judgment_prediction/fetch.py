import argparse, os, sys
from pathlib import Path

REPO, PREFIX = "Exploration-Lab/CS-781-Capstone", "Phase_4_CJPE"
FILES = ["Train/train.json", "Train/dev.json", "Test/test.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi, HfFolder, hf_hub_download
    from huggingface_hub.errors import GatedRepoError


    if not (HfFolder.get_token() or os.environ.get("HF_TOKEN")):
        sys.exit("No Hugging Face token. Run `hf auth login`, or set HF_TOKEN.")
    if a.list:
        for f in HfApi().list_repo_files(REPO, repo_type="dataset"):
            if f.startswith(PREFIX):
                print(" ", f)
        return
    import shutil
    for rel in FILES:
        src = f"{PREFIX}/{rel}"
        try:


            p = hf_hub_download(REPO, src, repo_type="dataset", local_dir=a.out)
        except GatedRepoError:
            sys.exit(f"Access to {REPO} is gated; request access on the Hub, then "
                     f"`hf auth login`.")
        dst = Path(p)
        print(f"  {src}  {dst.stat().st_size/1e6:.2f} MB")
    print(f"\nNext: inspect data/Phase_4_CJPE")


if __name__ == "__main__":
    main()
