"""Update Parking-Curb tasks and labels on CVAT to Lane-Line (lane_line).

1. Updates lines.xml for all 4 cameras (FV, RV, MVL, MVR) with label="lane_line".
2. Updates CVAT Tasks 21, 24, 27, 30:
   - Renames task to 'Day11 C · Lane-Line · <cam_name> (WoodScape)'
   - Renames label 'parking_curb' to 'lane_line'
   - Re-imports annotations to Ground Truth job and Student Annotation job

Usage:
    python scripts/update_lines_to_lane_line.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvat_sdk import make_client
from cvat_sdk.api_client import models
from _common import DATA, WS_CAMS, cvat_conn, load_env

WS = DATA / "woodscape"

CAM_TASKS = {
    "front": ("FV", 21),
    "rear": ("RV", 24),
    "left": ("MVL", 27),
    "right": ("MVR", 30),
}


def update_xml_files():
    print(">> Updating lines.xml files to label='lane_line'...")
    for cam in WS_CAMS:
        p = WS / cam / "lines.xml"
        if p.exists():
            content = p.read_text(encoding="utf-8")
            if 'label="parking_curb"' in content:
                content = content.replace('label="parking_curb"', 'label="lane_line"')
                p.write_text(content, encoding="utf-8")
                print(f"  Updated {p}")
            else:
                print(f"  Already updated: {p}")


def update_cvat():
    load_env()
    url, u, pw = cvat_conn()
    print(f"\n>> Connecting to CVAT at {url} as {u}...")

    with make_client(host=url, credentials=(u, pw)) as c:
        for name, (cam, tid) in CAM_TASKS.items():
            new_task_name = f"Day11 C · Lane-Line · {name} (WoodScape)"
            print(f"\n=== Task #{tid}: {new_task_name} ===")

            # 1. Rename task
            c.api_client.tasks_api.partial_update(
                tid,
                patched_task_write_request=models.PatchedTaskWriteRequest(name=new_task_name)
            )
            print(f"  Task #{tid} renamed to: {new_task_name}")

            # 2. Rename label 'parking_curb' to 'lane_line'
            labels = c.api_client.labels_api.list(task_id=tid)[0].results
            for lbl in labels:
                if lbl.name == "parking_curb":
                    c.api_client.labels_api.partial_update(
                        lbl.id,
                        patched_label_request=models.PatchedLabelRequest(name="lane_line")
                    )
                    print(f"  Label #{lbl.id} renamed to 'lane_line'")

            # 3. Re-import GT and Annotation jobs
            xml_file = WS / cam / "lines.xml"
            jobs = c.api_client.jobs_api.list(task_id=tid, page_size=10)[0].results
            gt_jobs = [j for j in jobs if str(getattr(j.type, "value", j.type)) == "ground_truth"]
            ann_jobs = [j for j in jobs if str(getattr(j.type, "value", j.type)) == "annotation"]

            if gt_jobs:
                gt_id = gt_jobs[0].id
                c.api_client.jobs_api.destroy_annotations(gt_id)
                c.jobs.retrieve(gt_id).import_annotations("CVAT 1.1", filename=str(xml_file))
                shapes = len(c.api_client.jobs_api.retrieve_annotations(gt_id)[0].shapes)
                print(f"  -> GT Job #{gt_id} imported {shapes} polylines (lane_line)")

            if ann_jobs:
                ann_id = ann_jobs[0].id
                c.api_client.jobs_api.destroy_annotations(ann_id)
                c.jobs.retrieve(ann_id).import_annotations("CVAT 1.1", filename=str(xml_file))
                shapes = len(c.api_client.jobs_api.retrieve_annotations(ann_id)[0].shapes)
                print(f"  -> Annotation Job #{ann_id} prefilled {shapes} polylines (lane_line)")

    print("\n>>> ALL DONE! All 4 line tasks on CVAT are now updated to Lane-Line (lane_line)!")


def main():
    update_xml_files()
    update_cvat()


if __name__ == "__main__":
    main()
