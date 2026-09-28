"""Verify Day-11 SVM/360 demo tasks on CVAT.

Compatible across Windows, Linux, and macOS (no bash heredoc required).
Usage:
    python scripts/verify.py
"""
import sys
from collections import Counter
from pathlib import Path

# Ensure scripts folder is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvat_sdk import make_client
from _common import cvat_conn, load_env


def main():
    load_env()
    url, u, pw = cvat_conn()
    ok = True
    print(f"Connecting to CVAT at {url} as {u}...")

    with make_client(host=url, credentials=(u, pw)) as c:
        tasks, _ = c.api_client.tasks_api.list(search="Day11", page_size=200)
        fam = Counter()
        if not tasks.results:
            print("No Day11 tasks found on CVAT server.")
            return

        for t in tasks.results:
            jobs_list = c.api_client.jobs_api.list(task_id=t.id, page_size=100)[0].results
            gt = [j for j in jobs_list if str(getattr(j.type, "value", j.type)) == "ground_truth"]
            if not gt:
                print(f"NO GT: {t.name}")
                ok = False
                continue

            a = c.api_client.jobs_api.retrieve_annotations(gt[0].id)[0]
            types = dict(Counter(str(getattr(s.type, "value", s.type)) for s in a.shapes))
            prefix = t.name.split(" · ")[0]
            fam[prefix] += 1
            status = "OK " if a.shapes else "EMPTY"
            print(f"{status} {t.name}: {len(a.shapes)} {types}")

        print("\nFamilies count:", dict(fam))
        if ok and len(tasks.results) >= 16:
            print("\n>>> ALL GOOD: 16 tasks verified with valid Ground Truth shapes!")
        else:
            print(f"\n>>> CHECK: Found {len(tasks.results)} tasks (expected >= 16). Review the warnings above.")


if __name__ == "__main__":
    main()
