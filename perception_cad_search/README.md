# `move_pillbottle_pad` CAD-ranked perception experiment

This directory is a successor to the reported 35.7% baseline.  It does **not**
overwrite `cup_target_focus.py` or `cup_known_model.py`; those files remain the
reproducible baseline.  The `move_pillbottle_pad` registry entry now selects
this worker, so the normal `run_ula_task.sh --task move_pillbottle_pad` command
uses exact-CAD candidate search as well as the isolated wrapper below.

## Changed decision boundary

Baseline:

```text
DINO -> SAM -> orange hard gate -> cup aspect hard gate -> FoundationPose
```

Experiment:

```text
specific DINO aliases + SAM
             | if exact CAD does not resolve a unique target
             v
bounded unlabelled RGB-D connected regions + SAM
             v
exact current model CAD (base1..base5)
             v
FoundationPose hypotheses
             v
point-to-CAD agreement + rendered-depth agreement + scene-depth check
             v
absolute acceptance and first/second CAD cost-gap check
```

Important properties:

- Orange is not a hard gate. White and brown models remain eligible.
- The cup-specific `0.65..1.8` aspect gate is not used.
- The exact per-episode `model_id` is mandatory in every request.
- DINO remains a fast proposal source, not the final identity authority.
- Search is bounded. It does not run FoundationPose on every automatic SAM mask.
- No simulator object pose, actor mask, contact or success signal is consumed.

## Files

- `pillbottle_cad_search.py`: query/config builder, vision implementation and worker.
- `perceive_pillbottle_cad_search.py`: persistent Python 3.10 vision process.
- `run_move_pillbottle_cad_search.py`: process-local integration with the existing ULA entry.
- `run_move_pillbottle_cad_search.sh`: command wrapper.

## First validation

Localization plus traditional pre-grasp only (no VLA, trial lift or placement):

```bash
cd /data1/xuhuixin/ula

GPU=<free_gpu> ULA_TASK_CONFIG=demo_randomized \
bash perception_cad_search/run_move_pillbottle_cad_search.sh \
  --mode approach-only --grasp-handoff stable-lift \
  --episodes 5 --seed 0 --instruction-from-demo \
  --output results/pillbottle_cad_search_approach5
```

`--grasp-handoff stable-lift` is required here only because the common runner
rejects its registered `visual-trial-lift` default in `approach-only` mode.  No
lift is executed in this mode.

Run a small randomized localization/full-pipeline pilot in a new output directory:

```bash
cd /data1/xuhuixin/ula

GPU=<free_gpu> ULA_TASK_CONFIG=demo_randomized \
bash perception_cad_search/run_move_pillbottle_cad_search.sh \
  --mode hybrid \
  --episodes 5 --seed 0 \
  --instruction-from-demo \
  --output results/pillbottle_cad_search_pilot5
```

Do not compare this pilot directly with the reported 35.7% until both methods
use the same seed list and denominator. First inspect each episode's
`observation_audit_json`, rejected CAD hypotheses, latency, and selected mask.

After the pilot, run a frozen 50-scene localization comparison, then a 50-episode
full-task evaluation. Cup regression must use the same candidate-search backend
with the cup's own exact CAD; it is not established by this implementation.

## Status

Implemented and unit-tested at the configuration/protocol level.  A five-scene
`demo_randomized` pre-grasp pilot (seeds 0--4) was also run at:

```text
tra_pipe/RoboTwin/results/pillbottle_cad_search_approach5_seed0_v2
```

Results:

- exact-CAD object plus support localization: **5/5**;
- pre-grasp reached: **4/5**;
- per-scene models: `base1`, `base2`, `base3`, `base5`, `base3`;
- both white models in this sample (`base2` and `base5`) localized successfully;
- seed 3 localized `base5` successfully, then failed in the downstream approach
  action because a move action carried `target_pose=None`.  It is therefore not
  counted as a perception failure;
- perception compute times were 11.84, 18.30, 8.66, 9.33 and 3.98 seconds
  (median 9.33 seconds).  Loading a previously unseen CAD model dominates the
  slower requests; the repeated `base3` request was the fastest.

One earlier saved real RGB-D frame using `base1` also passed the exact-CAD
pipeline in 11.38 seconds after model initialization.  These are integration
checks, not a statistically adequate success-rate comparison.  No 50-scene or
full-task success-rate claim is made yet, and the brown `base4` model still
needs explicit coverage in the next frozen localization evaluation.

### 2026-11-15 integration check

- The regular unified entry was smoke-tested on randomized seed 4 and reported
  `source=bounded_semantic_plus_RGBD_regions_exact_CAD_search`, model `base3`,
  with a 2.37 mm pre-grasp arrival error.
- A complete no-trial-lift seed-4 run passed RoboTwin's official success check.
  The final object/support XY error was 1.59 cm.  Vertical withdrawal was
  unreachable, so the generic post-release withdrawal selected a feasible
  reverse-transport direction.
- The corresponding result is
  `tra_pipe/RoboTwin/results/pillbottle_retreat_fix_no_trial_seed4_20261115`.
- With visual trial lift enabled, the same seed was rejected before placement:
  post-grasp segmentation/registration produced a false 51 cm displacement for
  a 5 cm hand lift.  Therefore the successful no-trial result is not evidence
  that visual grasp verification is solved.
