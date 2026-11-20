"""
Automated Downloader for MVTec Anomaly Detection Dataset
Pharos University in Alexandria - Faculty of Computer Science & AI
Deep Learning AI403 Project
"""

import os
import sys
import tarfile
import urllib.request
import argparse
from tqdm import tqdm

MVTEC_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
BOTTLE_FALLBACK_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract(url: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    archive_path = os.path.join(dest_dir, "dataset.tar.xz")

    print(f"[*] Downloading dataset from: {url}")
    print(f"[*] Destination: {archive_path}")

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(url, filename=archive_path, reporthook=t.update_to)
        print("\n[+] Download completed. Extracting archive...")
        
        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(path=dest_dir)
        print(f"[+] Dataset extracted successfully to: {dest_dir}")
        
        if os.path.exists(archive_path):
            os.remove(archive_path)
            print("[+] Cleaned up temporary archive.")
            
    except Exception as e:
        print(f"[-] Download/Extraction failed: {e}")
        print("[-] Manual instructions: Download the MVTec AD dataset from https://www.mvtec.com/company/research/datasets/mvtec-ad")
        print(f"[-] Place unpacked folders into: {dest_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download MVTec AD dataset")
    parser.add_argument("--category", type=str, default="bottle", help="Category to download ('bottle' or 'all')")
    parser.add_argument("--dest", type=str, default="./data/mvtec_anomaly_detection", help="Target directory")
    args = parser.parse_args()

    url = BOTTLE_FALLBACK_URL if args.category == "bottle" else MVTEC_URL
    download_and_extract(url, args.dest)

if __name__ == "__main__":
    main()
