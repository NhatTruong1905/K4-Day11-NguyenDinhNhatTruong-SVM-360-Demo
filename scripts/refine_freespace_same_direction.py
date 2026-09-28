"""Refine Free-space tasks to only include the same-direction (ego) drivable corridor.

Excludes oncoming/opposite traffic lanes:
- FV (Front View): Keeps only the ego vehicle's travel lane (left half of road up to center lane divider).
- RV (Rear View): Keeps only the ego vehicle's travel lane (right half of road from center lane divider).
- MVR (Right View): Keeps the ego lane adjacent to the vehicle, excluding opposite lane.
- MVL (Left View): Curbside / ego vehicle side.

Updates both Ground Truth jobs and regular Annotation jobs on CVAT.

Usage:
    python scripts/refine_freespace_same_direction.py
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvat_sdk import make_client
from _common import DATA, cvat_conn, load_env

WS = DATA / "woodscape"

CAM_TASKS = {
    "front": ("FV", 20),
    "rear": ("RV", 23),
    "left": ("MVL", 26),
    "right": ("MVR", 29),
}


def get_divider_x(cam, name):
    lines_xml = WS / cam / "lines.xml"
    if not lines_xml.exists():
        return 640.0
    tree = ET.parse(lines_xml)
    img_node = tree.getroot().find(f".//image[@name='{name}']")
    if img_node is None:
        return 640.0
    
    candidates = []
    for poly in img_node.findall('polyline'):
        ltype_elem = poly.find(".//attribute[@name='line_type']")
        ltype = ltype_elem.text if ltype_elem is not None else ""
        if ltype == 'lane_marking':
            pts = np.array([[float(v) for v in p.split(',')] for p in poly.get('points').split(';')])
            mx = pts[:, 0].mean()
            if 420 <= mx <= 780:
                candidates.append(mx)
    if candidates:
        return min(candidates, key=lambda x: abs(x - 640))
    return 640.0


def poly_area(pts):
    x = pts[:, 0]; y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def coco_poly_ann(aid, img_id, cat_id, pts, attrs):
    xs, ys = pts[:, 0], pts[:, 1]
    x, y = float(xs.min()), float(ys.min())
    w, h = float(xs.max() - x), float(ys.max() - y)
    return {"id": aid, "image_id": img_id, "category_id": cat_id,
            "segmentation": [pts.reshape(-1).tolist()],
            "bbox": [x, y, w, h], "area": float(poly_area(pts)),
            "iscrowd": 0, "attributes": attrs}


def refine_camera_freespace(cam):
    raw_path = WS / cam / "free_space.json"
    if not raw_path.exists():
        print(f"[SKIP] {raw_path} not found.")
        return False
    
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    refined_anns = []
    aid = 1

    img_by_id = {im['id']: im for im in data['images']}

    for ann in data['annotations']:
        img_info = img_by_id.get(ann['image_id'])
        if not img_info:
            continue
        name = img_info['file_name']
        div_x = get_divider_x(cam, name)

        pts = np.array(ann['segmentation'][0]).reshape(-1, 2)
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid or poly.is_empty:
                continue

        # Define the ego-lane corridor
        if cam == "FV":
            # Driving on left in UK -> ego lane is x <= divider_x
            clip = box(0, 0, div_x, 966)
        elif cam == "RV":
            # Rear view -> ego lane is x >= divider_x
            clip = box(div_x, 0, 1280, 966)
        elif cam == "MVR":
            # Right mirror -> corridor near vehicle
            clip = box(0, 0, 680, 966)
        else: # MVL
            clip = box(0, 0, 1280, 966)

        clipped = poly.intersection(clip)
        if clipped.is_empty:
            continue

        geoms = [clipped] if isinstance(clipped, Polygon) else (clipped.geoms if isinstance(clipped, MultiPolygon) else [])
        for g in geoms:
            if g.is_empty or g.area < 500:
                continue
            coords = np.array(g.exterior.coords)[:-1]
            if len(coords) >= 3:
                refined_anns.append(coco_poly_ann(
                    aid, ann['image_id'], ann['category_id'],
                    coords, ann['attributes']
                ))
                aid += 1

    data['annotations'] = refined_anns
    raw_path.write_text(json.dumps(data), encoding="utf-8")
    print(f"Refined {cam} Free-space: {len(refined_anns)} same-direction polygons.")
    return True


def update_cvat():
    load_env()
    url, u, pw = cvat_conn()
    print(f"\nConnecting to CVAT at {url} as {u}...")

    with make_client(host=url, credentials=(u, pw)) as c:
        for name, (cam, tid) in CAM_TASKS.items():
            gt_file = WS / cam / "free_space.json"
            print(f"\n=== Task #{tid}: Day11 B · Free-space · {name} ({cam}) ===")

            jobs = c.api_client.jobs_api.list(task_id=tid, page_size=10)[0].results
            gt_jobs = [j for j in jobs if str(getattr(j.type, "value", j.type)) == "ground_truth"]
            ann_jobs = [j for j in jobs if str(getattr(j.type, "value", j.type)) == "annotation"]

            if gt_jobs:
                gt_id = gt_jobs[0].id
                c.api_client.jobs_api.destroy_annotations(gt_id)
                c.jobs.retrieve(gt_id).import_annotations("COCO 1.0", filename=str(gt_file))
                shapes = len(c.api_client.jobs_api.retrieve_annotations(gt_id)[0].shapes)
                print(f"  -> GT Job #{gt_id}: {shapes} same-direction polygons imported.")

            if ann_jobs:
                ann_id = ann_jobs[0].id
                c.api_client.jobs_api.destroy_annotations(ann_id)
                c.jobs.retrieve(ann_id).import_annotations("COCO 1.0", filename=str(gt_file))
                shapes = len(c.api_client.jobs_api.retrieve_annotations(ann_id)[0].shapes)
                print(f"  -> Student Annotation Job #{ann_id}: {shapes} same-direction polygons prefilled.")

    print("\n>>> ALL DONE! All Free-space tasks now only cover the same-direction lane!")


def main():
    print(">> Refining Free-space annotations to same-direction (ego) corridor...")
    for name, (cam, tid) in CAM_TASKS.items():
        refine_camera_freespace(cam)
    update_cvat()


if __name__ == "__main__":
    main()
