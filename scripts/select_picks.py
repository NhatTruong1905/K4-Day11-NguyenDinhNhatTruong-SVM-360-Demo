"""Select the locked image subset for the demo (run once; commits assets/picks/*.json).

Day 11 = SVM/360 fisheye. Two real datasets, both with ground truth:
  - FishEye8K (HF, Abeyankar/fisheye8k-mini)  -> Task A object boxes, per camera
  - WoodScape (Kaggle, subarnadasgupta/woodscapes) -> Tasks B/C/D per SVM camera
    (instance_annotations polygons carry free_space / lane_marking+curb / ego_vehicle)

We do NOT re-host raw data (dataset licenses forbid it, slide 44). This script only
records which files to fetch; run.sh downloads them at build time.
"""
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PICKS = ROOT / "assets" / "picks"
PICKS.mkdir(parents=True, exist_ok=True)

N_PER_CAM = 20
WS_CAMS = ["FV", "RV", "MVL", "MVR"]          # WoodScape surround cameras
FE_CAMS = ["camera1", "camera2", "camera3", "camera4"]  # FishEye8K subset cameras

KAGGLE_DS = "subarnadasgupta/woodscapes"
WS_NEED = {"free_space", "lane_marking", "curb", "ego_vehicle"}


def kaggle_headers():
    tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not tok:
        raise SystemExit("KAGGLE_API_TOKEN missing (put it in ../cvat/.env or ./.env).")
    return {"Authorization": f"Bearer {tok}"}


def ws_list_ids(session, max_pages=120):
    """List instance_annotation ids grouped by camera (list only, no download).

    Cap pages: the dataset index is huge; ~120 pages yields plenty of candidates
    per camera to score down to 20.
    """
    url = f"https://www.kaggle.com/api/v1/datasets/list/{KAGGLE_DS}"
    by_cam = defaultdict(list)
    tok = None
    for _ in range(max_pages):
        params = {"pageToken": tok} if tok else {}
        for attempt in range(4):
            try:
                r = session.get(url, params=params, timeout=30)
                break
            except requests.RequestException:
                time.sleep(2)
        else:
            continue
        d = r.json()
        for f in d.get("datasetFiles", []):
            m = re.match(r"Woodscapes/instance_annotations/(\d+)_(FV|RV|MVL|MVR)\.json",
                         f.get("name", ""))
            if m:
                by_cam[m.group(2)].append(int(m.group(1)))
        tok = d.get("nextPageToken")
        if not tok:
            break
        time.sleep(0.2)
    return by_cam


def ws_download_tags(session, idx, cam):
    """Fetch one instance json, return the set of class tags present."""
    url = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DS}"
    p = {"file_name": f"Woodscapes/instance_annotations/{idx:05d}_{cam}.json"}
    for attempt in range(4):
        try:
            r = session.get(url, params=p, timeout=40)
            if r.status_code == 200:
                d = json.loads(r.content)[f"{idx:05d}_{cam}.json"]
                tags = Counter()
                for a in d["annotation"]:
                    for t in a.get("tags", []):
                        tags[t] += 1
                return tags
        except requests.RequestException:
            time.sleep(1.5)
    return Counter()


def pick_woodscape():
    h = kaggle_headers()
    s = requests.Session(); s.headers.update(h)
    print("listing WoodScape ids ...")
    by_cam = ws_list_ids(s)
    print({k: len(v) for k, v in by_cam.items()})
    picks = {}
    for cam in WS_CAMS:
        ids = sorted(by_cam.get(cam, []))[:80]   # cap scoring scan
        chosen = []
        # scan candidates in order; keep images that contain free_space + a line + ego
        for idx in ids:
            tags = ws_download_tags(s, idx, cam)
            has_line = tags.get("lane_marking", 0) or tags.get("curb", 0)
            if tags.get("free_space", 0) and has_line and tags.get("ego_vehicle", 0):
                chosen.append(idx)
                print(f"  {cam} pick {idx:05d}  "
                      f"free={tags['free_space']} lane={tags.get('lane_marking',0)} "
                      f"curb={tags.get('curb',0)} ego={tags.get('ego_vehicle',0)}")
            if len(chosen) >= N_PER_CAM:
                break
            time.sleep(0.15)
        # fallback: relax to free_space + (line OR ego) if too few
        if len(chosen) < N_PER_CAM:
            for idx in ids:
                if idx in chosen:
                    continue
                tags = ws_download_tags(s, idx, cam)
                if tags.get("free_space", 0) and (tags.get("curb", 0) or tags.get("lane_marking", 0)):
                    chosen.append(idx)
                if len(chosen) >= N_PER_CAM:
                    break
                time.sleep(0.15)
        picks[cam] = chosen[:N_PER_CAM]
        print(f"{cam}: {len(picks[cam])} picked")
    (PICKS / "woodscape_picks.json").write_text(json.dumps(picks, indent=2))
    print("wrote", PICKS / "woodscape_picks.json")


def pick_fisheye8k():
    print("fetching FishEye8K-mini samples.json ...")
    url = "https://huggingface.co/datasets/Abeyankar/fisheye8k-mini/resolve/main/samples.json"
    d = requests.get(url, timeout=60).json()
    by_cam = defaultdict(list)
    for s in d["samples"]:
        fp = s["filepath"]                       # data/cameraN_A_M.png
        m = re.match(r"data/(camera\d+)_", fp)
        if not m:
            continue
        # keep only samples that actually have boxes
        dets = s.get("detections", {}).get("detections", [])
        if dets:
            by_cam[m.group(1)].append(Path(fp).name)
    picks = {}
    for cam in FE_CAMS:
        names = sorted(by_cam.get(cam, []))
        picks[cam] = names[:N_PER_CAM]
        print(f"{cam}: {len(picks[cam])} picked of {len(names)}")
    (PICKS / "fisheye8k_picks.json").write_text(json.dumps(picks, indent=2))
    print("wrote", PICKS / "fisheye8k_picks.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["woodscape", "fisheye8k"])
    a = ap.parse_args()
    if a.only in (None, "fisheye8k"):
        pick_fisheye8k()
    if a.only in (None, "woodscape"):
        pick_woodscape()
