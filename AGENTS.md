# AGENTS.md — runbook for an AI agent

Deterministic steps to recreate the Day-11 SVM/360 fisheye tasks on a local CVAT.
Follow in order; verify at the gate.

## Goal

Create **16 CVAT tasks** (split by `camera_id`), each with a golden GT job:

| Family | Geometry | Source | Tasks | Notes |
|---|---|---|---|---|
| A object | rectangle | FishEye8K | 4 (camera1–4) | 5 classes; COCO 1.0 GT |
| B free-space | polygon | WoodScape | 4 (front/rear/left/right) | COCO 1.0 GT |
| C parking/curb | polyline | WoodScape | 4 | CVAT 1.1 GT (centerlines) |
| D ignore | polygon | WoodScape | 4 | COCO 1.0 GT (ego_body) |

Datasets are **not** vendored (license). The build downloads only the locked subset
in `assets/picks/*.json` into `data/` (git-ignored).

## Preconditions

1. Working dir = this folder. `assets/picks/fisheye8k_picks.json` and
   `assets/picks/woodscape_picks.json` exist.
2. Python 3.10+ with deps: `"$PY" -m pip install -r requirements.txt`;
   verify `"$PY" -c "import cvat_sdk,numpy,PIL,requests,dotenv; print('ok')"`.
3. CVAT reachable: `curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/api/server/about` → `200`.
4. `./.env` (or `../cvat/.env`) has `CVAT_URL`, `CVAT_ADMIN_USER`, `CVAT_ADMIN_PASSWORD`,
   and `KAGGLE_API_TOKEN` (WoodScape download). FishEye8K needs no token.

## Build

```bash
./run.sh --recreate           # PY=/path/to/python ./run.sh --recreate  if needed
```

Expected tail: `DONE. 16 tasks. Open:` followed by 16 task URLs. **Read those ids;
they change every `--recreate`. Never hard-code an id.**

## Verify (gate)

```bash
"$PY" - <<'PY'
import sys; sys.path.insert(0,"scripts")
from collections import Counter
from cvat_sdk import make_client
from _common import cvat_conn, load_env
load_env(); url,u,pw=cvat_conn()
ok=True
with make_client(host=url,credentials=(u,pw)) as c:
    tasks,_=c.api_client.tasks_api.list(search="Day11",page_size=200)
    fam=Counter()
    for t in tasks.results:
        gt=[j for j in c.api_client.jobs_api.list(task_id=t.id,page_size=100)[0].results
            if str(getattr(j.type,'value',j.type))=="ground_truth"]
        if not gt: print("NO GT:",t.name); ok=False; continue
        a=c.api_client.jobs_api.retrieve_annotations(gt[0].id)[0]
        types=dict(Counter(str(getattr(s.type,'value',s.type)) for s in a.shapes))
        fam[t.name.split(" · ")[0]]+=1
        print(f"{'OK ' if a.shapes else 'EMPTY'} {t.name}: {len(a.shapes)} {types}")
    print("families:",dict(fam))
    print("ALL GOOD" if ok and len(tasks.results)>=16 else "CHECK")
PY
```

Pass = 16 tasks, each GT job non-empty, geometry types match the family
(A rectangle, B polygon, C polyline, D polygon), final `ALL GOOD`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `KAGGLE_API_TOKEN missing` | add it to `./.env` (WoodScape download). |
| `CVAT_ADMIN_PASSWORD missing` | fill `./.env` from `.env.example`. |
| HTTP != 200 on server/about | start CVAT, wait for `cvat_server`. |
| `failed to download Woodscapes/...` | Kaggle rate-limit/token; re-run (idempotent, resumes cached files). |
| CVAT rejects polyline import (C) | label `parking_curb` must be type `polyline` (`_common.py`); rebuild. |
| picks file missing | `python scripts/select_picks.py` (maintainer; needs Kaggle token). |

## Do NOT

- Do not commit `data/` (downloaded raw data — license forbids re-hosting).
- Do not vendor dataset images into `assets/`; only the picks lists belong there.
- Do not hard-code task/job ids.
