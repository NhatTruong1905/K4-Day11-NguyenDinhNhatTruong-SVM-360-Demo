# Day 11 — SVM / 360° Fisheye Demo (CVAT, local)

Practice tasks for the **Day 11 "SVM / 360° External Cameras"** lecture (fisheye,
surround-view, camera-specific QA). Built from `svm360-day11-theory.pdf`.

The lecture's core rule — **gold set / tasks split by `camera_id`, not just by
dataset** — is baked in: every label family is a separate CVAT task **per camera**,
each with a golden ground-truth job so student work is scored per-object IoU and
sliced per camera.

Two real fisheye datasets with ground truth, ~20 images per task:

- **FishEye8K** (HuggingFace mirror) → **Task A** near-object boxes, per camera.
- **WoodScape** (Kaggle) → **Tasks B/C/D** (free-space, parking/curb, ignore-region),
  per SVM camera (front/rear/left/right), derived from its instance-polygon GT.

> Datasets are **not** re-hosted here (their licenses forbid it — slide 44). The repo
> stores only the *locked list of file ids* it uses; `run.sh` downloads that subset
> at build time.

---

## What this builds (16 tasks)

| Family | Geometry | Source | Cameras | Tasks | Golden GT |
|---|---|---|---|---|---|
| **A · Object** | Rectangle | FishEye8K | camera1–4 | 4 | boxes (COCO 1.0), 5 classes |
| **B · Free-space** | Polygon | WoodScape | front/rear/left/right | 4 | `free_space` polygons (COCO 1.0) |
| **C · Parking/Curb** | Polyline | WoodScape | front/rear/left/right | 4 | `lane_marking`+`curb` centerlines (CVAT 1.1) |
| **D · Ignore-region** | Polygon | WoodScape | front/rear/left/right | 4 | `ego_vehicle` polygons (COCO 1.0) |

Task names: `Day11 A · Object · camera1 (FishEye8K)`, `Day11 B · Free-space · front
(WoodScape)`, etc. Task/job **ids change on every `--recreate`** — read the URLs
`run.sh` prints, or `http://localhost:8080/tasks`.

**Honest scope** (derived golden GT — teaching-grade, not the dataset's own eval labels):
- **A** boxes are FishEye8K's authoritative detections (5 classes: Bus, Bike, Car,
  Pedestrian, Truck).
- **B** free-space = WoodScape `free_space` polygons (drivable-near-vehicle intent).
- **C** golden = the **centerline** of each WoodScape `lane_marking`/`curb` polygon
  (PCA axis + binned mean), so a student's polyline is comparable. Real lane/curb
  geometry, reduced to a line.
- **D** ignore = WoodScape `ego_vehicle` polygons (`reason=ego_body`). Lens-border /
  seam / unreadable are in the schema for students to add, but not in the golden.

---

## Prerequisites

1. **Local CVAT + admin.** Course machine: `cd ../cvat && ./start-cvat.sh up && ./start-cvat.sh superuser`.
   Own machine: install official CVAT (`git clone https://github.com/cvat-ai/cvat && cd cvat && docker compose up -d`, then create a superuser) — serves at http://localhost:8080.
2. **Python deps**: `pip install -r requirements.txt` (course machine uses conda env `ai-lab`).
3. **Credentials**: `cp .env.example .env`, set `CVAT_ADMIN_USER`/`CVAT_ADMIN_PASSWORD`
   and a **Kaggle token** (`KAGGLE_API_TOKEN=KGAT_...`, free at
   https://www.kaggle.com/settings/api) — needed for the WoodScape download (B/C/D).
   FishEye8K (A) needs no token.

---

## Build it

**Windows (PowerShell):**
```powershell
.\run.ps1                 # download locked subset + build all 16 tasks + golden GT
.\run.ps1 -Recreate       # delete existing Day11 tasks first
.\run.ps1 -Only object    # one family {object,freespace,lines,ignore}
```

**Windows (CMD):**
```cmd
run.bat
run.bat --recreate
run.bat --only object
```

**Linux / macOS (Bash):**
```bash
./run.sh                 # download locked subset + build all 16 tasks + golden GT
./run.sh --recreate      # delete existing Day11 tasks first
./run.sh --only object   # one family {object,freespace,lines,ignore}
# non-default python: PY=$(which python) ./run.sh
```

Steps (all re-runnable; downloads land in git-ignored `data/`):
1. `prepare_fisheye8k.py` — download the FishEye8K image subset, build per-camera COCO GT.
2. `prepare_woodscape.py` — download the WoodScape rgb + instance-annotation subset,
   derive per-camera free-space/curb-line/ignore golden GT.
3. `setup_cvat.py` — create the 16 tasks + a golden GT job for each.

**AI agents:** follow [AGENTS.md](AGENTS.md).

---

## Label schema (lecture slides 13 + 24)

- **A road_object** (rectangle) — 5 class labels; attrs `occluded`, `truncated`,
  `edge_zone`, `ignore_reason`, `camera_id`. Rule: box the **visible extent**, not an
  imagined un-distorted shape; `edge_zone`/`truncated` are attributes, not a reason to
  draw wider (slide 17).
- **B free_space** (polygon) — attrs `surface`, `confidence`, `blocked_by`, `camera_id`.
  Drivable **by the ego at low speed**, not "all road texture" (slide 28).
- **C parking_curb** (polyline) — attrs `line_type` {lane_marking, curb}, `visibility`,
  `edge_zone`, `camera_id`. Follow the curve in the image plane; don't straighten it
  (slide 29).
- **D ignore_region** (polygon) — attr `reason` {ego_body, lens_border, stitch_seam,
  unreadable, privacy_or_policy}. Ignore regions are part of the label, not "skip"
  (slide 23).

---

## The golden-GT quality loop (per camera)

1. A student annotates a task's annotation job and saves.
2. Task **Actions → Quality → refresh** (or the API below).
3. CVAT scores each shape vs the golden GT: per-object IoU, Missing/Extra, wrong label.
   Because tasks are already split by camera, the report **is** the per-camera slice the
   lecture asks for (slide 35). Compare front vs rear vs left vs right to see each
   camera's error profile.

```bash
~/miniconda3/envs/ai-lab/bin/python - <<'PY'
import sys; sys.path.insert(0, "scripts")
from cvat_sdk import make_client
from _common import cvat_conn, load_env
load_env(); url,u,pw=cvat_conn()
with make_client(host=url, credentials=(u,pw)) as c:
    tasks,_ = c.api_client.tasks_api.list(search="Day11", page_size=200)
    for t in tasks.results:
        rq = c.api_client.quality_api.create_report(quality_report_create_request={"task_id":t.id})
    print("queued quality reports for", len(tasks.results), "Day11 tasks")
PY
```

---

## Regenerate the picks (maintainers)

`assets/picks/*.json` lock which files the build downloads. To re-select:

```bash
python scripts/select_picks.py                 # both datasets
python scripts/select_picks.py --only woodscape # slower (scores instance tags)
```

WoodScape picks favour frames that contain free_space + a line + ego body; FishEye8K
picks take the first 20 annotated frames per camera. Locked → reproducible builds.

---

## File map

```
day11-svm360-demo/
  README.md               this playbook
  AGENTS.md               deterministic runbook for an AI agent
  run.sh                  build (download subset -> derive GT -> create tasks)
  requirements.txt        deps
  .env.example            copy to ./.env; set CVAT admin + Kaggle token
  assets/picks/           COMMITTED locked file lists (no raw data)
    fisheye8k_picks.json  20 image names per camera1..4
    woodscape_picks.json  20 image ids per FV/RV/MVL/MVR
  scripts/
    _common.py            schemas, camera lists, tag maps, env/conn
    select_picks.py       (maintainer) choose the locked subset
    prepare_fisheye8k.py  download A subset + build COCO GT
    prepare_woodscape.py  download B/C/D subset + derive golden GT
    setup_cvat.py         create the 16 camera-split tasks + GT jobs
  data/                   (git-ignored) downloaded images + derived GT
```

## Datasets & licenses

- **FishEye8K** — Gochoo et al.; 8,000 fisheye images, 157K boxes, 5 classes.
  Mirror: `Abeyankar/fisheye8k-mini` (HuggingFace). Paper: arXiv:2305.17449.
- **WoodScape** — Valeo; multi-camera surround-view fisheye, multi-task.
  Mirror: `subarnadasgupta/woodscapes` (Kaggle). Paper: arXiv:1905.01489.

Use for teaching only; obtain data from official sources under their terms. Do **not**
re-host raw data (slide 44). `data/` is git-ignored for this reason.
