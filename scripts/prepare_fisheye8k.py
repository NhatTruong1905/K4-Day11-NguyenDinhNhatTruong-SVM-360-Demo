"""Task A assets — FishEye8K object boxes, per camera, with golden GT (COCO 1.0).

Downloads the locked FishEye8K-mini image subset from HuggingFace (public, no token)
and builds one COCO 1.0 GT file per camera from the dataset's FiftyOne detections
(normalized xywh -> absolute pixel bbox). No re-hosting: images land in data/ only.
"""
import json
import time
from pathlib import Path

import requests

from _common import DATA, FE_CAMS, FE_CLASSES, PICKS

HF = "https://huggingface.co/datasets/Abeyankar/fisheye8k-mini/resolve/main"
OUT = DATA / "fisheye8k"
LABEL_MAP = {c.lower(): c for c in FE_CLASSES}   # tolerate casing


def hget(url, tries=5):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise SystemExit(f"failed to download {url}")


def load_samples():
    import io
    return json.loads(hget(f"{HF}/samples.json"))["samples"]


def main():
    picks = json.loads((PICKS / "fisheye8k_picks.json").read_text(encoding="utf-8"))
    want = {name: cam for cam in FE_CAMS for name in picks.get(cam, [])}
    samples = {Path(s["filepath"]).name: s for s in load_samples()}

    for cam in FE_CAMS:
        names = picks.get(cam, [])
        if not names:
            continue
        img_dir = OUT / cam / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        cats = [{"id": i + 1, "name": c, "supercategory": "road_object"}
                for i, c in enumerate(FE_CLASSES)]
        cat_id = {c["name"]: c["id"] for c in cats}
        images, anns = [], []
        aid = 1
        for k, name in enumerate(names, start=1):
            s = samples[name]
            w = s["metadata"]["width"]; h = s["metadata"]["height"]
            # download image
            dst = img_dir / name
            if not (dst.exists() and dst.stat().st_size > 0):
                dst.write_bytes(hget(f"{HF}/data/{name}"))
            images.append({"id": k, "file_name": name, "width": w, "height": h})
            for det in s.get("detections", {}).get("detections", []):
                lbl = LABEL_MAP.get(det["label"].lower())
                if not lbl:
                    continue
                bx, by, bw, bh = det["bounding_box"]      # normalized xywh
                x, y, ww, hh = bx * w, by * h, bw * w, bh * h
                anns.append({"id": aid, "image_id": k, "category_id": cat_id[lbl],
                             "bbox": [x, y, ww, hh], "area": ww * hh,
                             "iscrowd": 0, "segmentation": []})
                aid += 1
        gt = {"info": {"description": f"Day11 FishEye8K {cam} object GT"},
              "licenses": [], "images": images, "annotations": anns, "categories": cats}
        (OUT / cam / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
        print(f"{cam}: {len(images)} images, {len(anns)} boxes -> {OUT/cam/'gt.json'}")


if __name__ == "__main__":
    main()
