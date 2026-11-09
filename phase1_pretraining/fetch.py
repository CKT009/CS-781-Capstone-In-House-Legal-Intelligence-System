from __future__ import annotations
import argparse, sys, time
from pathlib import Path

REPO, PREFIX = "Exploration-Lab/CS-781-Capstone", "Phase_1_Pretraining"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*", default=None,
                    help="substring filter: test acts supreme_court tribunals high_courts")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().repo_info(REPO, repo_type="dataset", files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings
             if s.rfilename.startswith(PREFIX)]
    if a.only:
        files = [(f, n) for f, n in files if any(k in f for k in a.only)]
    if not files:
        sys.exit(f"nothing matched {a.only}")

    files.sort(key=lambda x: x[1])
    total = sum(n for _, n in files) / 1e9
    for f, n in files:
        print(f"  {n/1e6:9.1f} MB  {f}")
    print(f"total: {total:.2f} GB")
    if a.list:
        return
    if total > 2 and input(f"download {total:.1f} GB? [y/N] ").strip().lower() != "y":
        return

    for f, n in files:
        t0 = time.time()
        for attempt in range(5):
            try:
                p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=a.out)
                print(f"[ok] {f} ({n/1e6:.0f} MB in {time.time()-t0:.0f}s)")
                break
            except Exception as e:
                print(f"[retry {attempt+1}/5] {repr(e)[:110]}")
                time.sleep(2 ** attempt)
        else:
            print(f"[FAILED] {f} -- just rerun this script, it resumes")


if __name__ == "__main__":
    main()
