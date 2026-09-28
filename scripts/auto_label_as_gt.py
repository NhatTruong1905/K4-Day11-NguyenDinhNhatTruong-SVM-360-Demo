"""Auto-label student annotation jobs with Ground Truth annotations.

Imports all Ground Truth labels directly into the regular student annotation jobs
for Day11 tasks, so when you open the task in CVAT, every image is already 100% labeled.

Usage:
    python scripts/auto_label_as_gt.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvat_sdk import make_client
from _common import DATA, cvat_conn, load_env

FE = DATA / "fisheye8k"
WS = DATA / "woodscape"

NAME_TO_CAM = {
    "front": "FV",
    "rear": "RV",
    "left": "MVL",
    "right": "MVR"
}


def find_gt_for_task(task_name: str) -> tuple[str, Path] | None:
    # Task A: Day11 A · Object · camera1 (FishEye8K)
    for cam in ["camera1", "camera2", "camera3", "camera4"]:
        if f"Object · {cam}" in task_name:
            p = FE / cam / "gt.json"
            if p.exists():
                return "COCO 1.0", p

    # Tasks B/C/D: WoodScape
    for word, cam in NAME_TO_CAM.items():
        if f" · {word} " in task_name:
            if "B · Free-space" in task_name:
                p = WS / cam / "free_space.json"
                if p.exists():
                    return "COCO 1.0", p
            elif "C · Parking-Curb" in task_name or "C · Lane-Line" in task_name:
                p = WS / cam / "lines.xml"
                if p.exists():
                    return "CVAT 1.1", p
            elif "D · Ignore" in task_name:
                p = WS / cam / "ignore.json"
                if p.exists():
                    return "COCO 1.0", p
    return None


def main():
    load_env()
    url, u, pw = cvat_conn()
    print(f"Connecting to CVAT at {url} as {u}...")

    with make_client(host=url, credentials=(u, pw)) as c:
        tasks, _ = c.api_client.tasks_api.list(search="Day11", page_size=200)
        if not tasks.results:
            print("No Day11 tasks found on CVAT.")
            return

        print(f"Found {len(tasks.results)} Day11 tasks. Filling annotation jobs with GT...\n")

        for t in sorted(tasks.results, key=lambda x: x.id):
            gt_info = find_gt_for_task(t.name)
            if not gt_info:
                print(f"[SKIP] Task #{t.id} '{t.name}': No matching GT file found in data/.")
                continue

            gt_format, gt_file = gt_info
            jobs = c.api_client.jobs_api.list(task_id=t.id, page_size=50)[0].results
            ann_jobs = [j for j in jobs if str(getattr(j.type, "value", j.type)) == "annotation"]

            if not ann_jobs:
                print(f"[SKIP] Task #{t.id} '{t.name}': No annotation jobs.")
                continue

            for aj in ann_jobs:
                print(f"Importing GT into Task #{t.id} -> Annotation Job #{aj.id} ({gt_format})...")
                # Clear existing and import
                c.api_client.jobs_api.destroy_annotations(aj.id)
                c.jobs.retrieve(aj.id).import_annotations(gt_format, filename=str(gt_file))
                ann = c.api_client.jobs_api.retrieve_annotations(aj.id)[0]
                print(f"  -> Done! Job #{aj.id} now has {len(ann.shapes)} shapes pre-labeled.")

        print("\n>>> ALL DONE! All annotation jobs are now fully labeled as Ground Truth.")


if __name__ == "__main__":
    main()
