from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO, PREFIX = "Exploration-Lab/CS-781-Capstone", "Phase_2_LNER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi, HfFolder, hf_hub_download, whoami
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError


    token = HfFolder.get_token() or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "Not logged in to HuggingFace, and this dataset is gated.\n"
            "  1. get a token: https://huggingface.co/settings/tokens\n"
            "  2. hf auth login\n"
            "  3. make sure that account has been granted access to\n"
            f"     https://huggingface.co/datasets/{REPO}\n"
            "(The file listing below would have worked anyway -- gated repos expose\n"
            " metadata publicly and refuse the files, so a successful --list proves\n"
            " nothing about your credentials.)")
    try:
        me = whoami(token=token)
        print(f"[auth] {me.get('name')}")
    except Exception as e:
        sys.exit(f"token present but rejected ({type(e).__name__}). Re-run `hf auth login`.")

    api = HfApi()
    try:
        info = api.repo_info(REPO, repo_type="dataset", files_metadata=True, token=token)
    except GatedRepoError:
        sys.exit(f"Your account ({me.get('name')}) has not been granted access to {REPO}.\n"
                 f"Accept the terms at https://huggingface.co/datasets/{REPO}")
    except RepositoryNotFoundError:
        sys.exit(f"{REPO} not found -- check the name, or your token lacks read scope.")
    files = [(s.rfilename, s.size or 0) for s in info.siblings
             if s.rfilename.startswith(PREFIX)]
    if not files:
        sys.exit(f"nothing under {PREFIX}/ in {REPO}. The phase may not be released yet "
                 f"-- Phase 2 opens 7 October 2026.")

    files.sort(key=lambda x: x[1])
    for f, n in files:
        print(f"  {n/1e6:9.2f} MB  {f}")
    print(f"total: {sum(n for _, n in files)/1e6:.1f} MB")
    if a.list:
        return

    for f, n in files:
        t0 = time.time()
        for attempt in range(5):
            try:
                hf_hub_download(REPO, f, repo_type="dataset",
                                local_dir=a.out, token=token)
                print(f"[ok] {f} ({n/1e6:.2f} MB in {time.time()-t0:.0f}s)")
                break
            except (GatedRepoError, RepositoryNotFoundError) as e:

                sys.exit(f"\n[auth] {type(e).__name__} on {f}\n"
                         f"Your token works but this account cannot read {REPO}.\n"
                         f"Accept the terms at https://huggingface.co/datasets/{REPO}")
            except Exception as e:
                print(f"[retry {attempt+1}/5] {type(e).__name__}: {str(e)[:90]}")
                time.sleep(2 ** attempt)
        else:
            print(f"[FAILED] {f} -- rerun this script")

    print("\nnow run:  python run_train.py --check")
    print("  It round-trips every gold span through the BIO encoder back to character")
    print("  offsets and reports what fraction survives. Under 99% means spans the model")
    print("  can never score, no matter how well it trains -- fix that before any GPU time.")


if __name__ == "__main__":
    main()
