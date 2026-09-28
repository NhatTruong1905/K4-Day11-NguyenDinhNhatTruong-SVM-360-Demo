"""Create the Day-11 SVM/360 demo tasks on local CVAT — split by camera_id.

Task A (object)      : FishEye8K, one task per camera (camera1..4)     -> COCO 1.0 GT
Task B (free_space)  : WoodScape,  one task per SVM camera (front..right)-> COCO 1.0 GT
Task C (parking_curb): WoodScape,  one task per SVM camera             -> CVAT 1.1 GT (polyline)
Task D (ignore)      : WoodScape,  one task per SVM camera             -> COCO 1.0 GT

Every task gets a Ground Truth job over all frames with the golden GT imported, so
CVAT's quality report scores per-object IoU (+ per-camera slicing, the lecture's rule).
Run AFTER prepare_fisheye8k.py and prepare_woodscape.py.
"""
import argparse
import time

from cvat_sdk import make_client
from cvat_sdk.api_client import models
from cvat_sdk.core.proxies.tasks import ResourceType

from _common import (DATA, FE_CAMS, WS_CAM_NAME, WS_CAMS, cvat_conn,
                     freespace_labels, ignore_labels, line_labels, load_env,
                     object_labels)

FE = DATA / "fisheye8k"
WS = DATA / "woodscape"
PREFIX = "Day11"


def delete_by_prefix(client, prefix):
    tasks, _ = client.api_client.tasks_api.list(search=prefix, page_size=200)
    for t in tasks.results:
        if t.name.startswith(prefix):
            print(f"  deleting existing task #{t.id} '{t.name}'")
            client.api_client.tasks_api.destroy(t.id)


def wait_ready(client, tid, timeout=300):
    for _ in range(timeout):
        t = client.tasks.retrieve(tid)
        if getattr(t, "size", 0):
            return t
        time.sleep(1)
    raise TimeoutError(f"task {tid} never became ready")


def create_task(client, name, labels, images, gt_format, gt_file):
    print(f"\n=== {name} ===")
    task = client.tasks.create_from_data(
        spec={"name": name, "labels": labels},
        resource_type=ResourceType.LOCAL, resources=[str(p) for p in images])
    task = wait_ready(client, task.id)
    n = task.size
    gt_job, _ = client.api_client.jobs_api.create(models.JobWriteRequest(
        task_id=task.id, type=models.JobType("ground_truth"),
        frame_selection_method=models.FrameSelectionMethod("manual"),
        frames=list(range(n))))
    job = client.jobs.retrieve(gt_job.id)
    job.import_annotations(gt_format, filename=str(gt_file))
    print(f"  task #{task.id} ({n} frames) + GT job #{gt_job.id} [{gt_format}] imported")
    return task, gt_job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--only", choices=["object", "freespace", "lines", "ignore"])
    args = ap.parse_args()
    want = lambda k: args.only in (None, k)

    load_env()
    url, user, pw = cvat_conn()
    with make_client(host=url, credentials=(user, pw)) as client:
        print("connected as", client.api_client.users_api.retrieve_self()[0].username)
        if args.recreate:
            delete_by_prefix(client, PREFIX)
        existing_tasks = {}
        if not args.recreate:
            existing_tasks = {t.name: t.id for t in client.api_client.tasks_api.list(search=PREFIX, page_size=200)[0].results}

        results = []
        # Task A — FishEye8K object, per camera
        if want("object"):
            for cam in FE_CAMS:
                tname = f"{PREFIX} A · Object · {cam} (FishEye8K)"
                if tname in existing_tasks:
                    print(f"skip {tname}: already exists (task #{existing_tasks[tname]})")
                    continue
                imgs = sorted((FE / cam / "images").glob("*.png"))
                if not imgs:
                    print(f"skip object {cam}: no images (run prepare_fisheye8k.py)"); continue
                results.append(create_task(
                    client, tname,
                    object_labels(cam), imgs, "COCO 1.0", FE / cam / "gt.json"))

        # Tasks B/C/D — WoodScape, per SVM camera
        for cam in WS_CAMS:
            nm = WS_CAM_NAME[cam]
            imgs = sorted((WS / cam / "images").glob("*.png"))
            if not imgs:
                print(f"skip WoodScape {cam}: no images (run prepare_woodscape.py)"); continue
            if want("freespace"):
                tname = f"{PREFIX} B · Free-space · {nm} (WoodScape)"
                if tname in existing_tasks:
                    print(f"skip {tname}: already exists (task #{existing_tasks[tname]})")
                else:
                    results.append(create_task(
                        client, tname,
                        freespace_labels(cam), imgs, "COCO 1.0", WS / cam / "free_space.json"))
            if want("lines"):
                tname = f"{PREFIX} C · Lane-Line · {nm} (WoodScape)"
                if tname in existing_tasks:
                    print(f"skip {tname}: already exists (task #{existing_tasks[tname]})")
                else:
                    results.append(create_task(
                        client, tname,
                        line_labels(cam), imgs, "CVAT 1.1", WS / cam / "lines.xml"))
            if want("ignore"):
                tname = f"{PREFIX} D · Ignore · {nm} (WoodScape)"
                if tname in existing_tasks:
                    print(f"skip {tname}: already exists (task #{existing_tasks[tname]})")
                else:
                    results.append(create_task(
                        client, tname,
                        ignore_labels(cam), imgs, "COCO 1.0", WS / cam / "ignore.json"))

        print(f"\nDONE. {len(results)} tasks. Open:")
        for task, gt in results:
            print(f"  {url}/tasks/{task.id}   (GT job #{gt.id})")


if __name__ == "__main__":
    main()
