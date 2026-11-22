#!/usr/bin/env python3
"""Local action-only warm start of WorldLA; never edits the shared trainer.

Reuses the native action loss and checkpoint layout. The existing Router stays
frozen. A split-only store replaces route-target loading because this action
fine tune does not generate or train against any new WAM route labels.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, default=ROOT/"checkpoints/worldla_100k")
    parser.add_argument("--asset-checkpoint", type=Path, default=ROOT/"checkpoints/worldla_assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000, help="Additional optimization steps, not total source step")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--max-validation-batches", type=int, help="Debug only; omit for full held-out validation")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.fsdp_devices, args.validation_interval) < 1:
        parser.error("Positive training sizes required")
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Refusing to overwrite a nonempty fine-tuned model")
    source = args.source_model.resolve()
    meta = json.loads((source/"experiment.json").read_text())
    if meta["condition"] != "wam_stage_dynamic" or meta["stage"] != "action":
        raise ValueError("Expected existing WorldLA action checkpoint")
    data = json.loads((args.dataset.resolve()/"dataset.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    package = ROOT/"external/wam_vla/experiments/stage_relation_router_multi_task4"
    pi_root = ROOT/"external/pi05"
    sys.path[:0] = [str(package), str(ROOT/"external/wam_vla"), str(pi_root), str(pi_root/"src")]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["OPENPI_RUNTIME_CACHE_DIR"] = str(ROOT/".runtime_cache")
    spec = importlib.util.spec_from_file_location("ula_pickup_native_trainer", package/"train_stage_router.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    import stage_route_data
    original_store = stage_route_data.StageRouteTargetStore
    original_val = stage_route_data.StageRouteOpenPiDataset
    class SplitOnlyStore(original_store):
        def __init__(self, manifest, **kwargs):
            super().__init__(manifest, load_routes=False)
    def full_frame_validation(transformed, store, *, split, condition, feature_store=None):
        return stage_route_data.FullEpisodeActionDataset(transformed, store, split=split)
    stage_route_data.StageRouteTargetStore = SplitOnlyStore
    stage_route_data.StageRouteOpenPiDataset = full_frame_validation
    trainer.configure_pi_runtime = lambda *unused: ROOT/".runtime_cache"
    start = int(meta["steps"])
    assets = args.asset_checkpoint.resolve()/"assets"
    asset_ids = sorted(p.name for p in assets.iterdir() if p.is_dir())
    if len(asset_ids) != 1 or not (assets/asset_ids[0]/"norm_stats.json").is_file():
        raise ValueError("Expected exactly one existing normalization asset")
    sys.argv = [str(package/"train_stage_router.py"), "--stage", "action", "--condition", "wam_stage_dynamic",
        "--pi05-root", str(pi_root), "--runtime-cache-dir", str(ROOT/".runtime_cache"),
        "--config-name", meta["config_name"], "--lerobot-root", data["lerobot_root"],
        "--dataset-repo-id", data["repo_id"], "--asset-dir", str(assets), "--asset-id", asset_ids[0],
        "--route-manifest", data["manifest"], "--init-params", meta["init_params"],
        "--router-model-dir", str(Path(meta["router_checkpoint_loaded"]).parent.parent),
        "--resume-model-dir", str(source), "--output-dir", str(args.output),
        "--steps", str(start+args.steps), "--batch-size", str(args.batch_size),
        "--fsdp-devices", str(args.fsdp_devices), "--learning-rate", str(args.learning_rate),
        "--validation-interval", str(args.validation_interval), "--log-interval", "25",
        # This wrapper already rejected preexisting contents. The ONLY file
        # now present is its own protocol; native trainer does not delete it.
        "--overwrite"]
    if args.max_validation_batches is not None:
        if args.max_validation_batches < 1:
            raise ValueError("Validation batch cap must be positive")
        sys.argv += ["--max-validation-batches", str(args.max_validation_batches)]
    files = [package/"train_stage_router.py", package/"stage_route_data.py", Path(__file__),
             ROOT/"adapters/handover/pickup_data.py", source/"experiment.json"]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    protocol = dict(schema="ula_pickup_action_finetune_v1", status="running", source_model=str(source),
        dataset=data, additional_steps=args.steps, learning_rate=args.learning_rate, batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices, training_parameters="existing_LoRA_and_stage_message_adapter_only",
        frozen="pi05_base_backbone_and_existing_stage_Router", new_WAM_teacher_targets=False,
        loss="native_pi05_flow_matching_action_loss", train_val_split="entire_episodes_45_train_5_validation",
        shared_source_files_edited=False, source_hashes_before=before,
        max_validation_batches=args.max_validation_batches,
        normalization="reuse_original_checkpoint_assets_to_match_existing_runtime")
    (args.output/"ula_protocol.json").write_text(json.dumps(protocol, indent=2))
    try:
        trainer.main()
        # The inherited trainer's teacher description refers to original pretraining;
        # make this fine-tuning stage's provenance explicit in deployable metadata.
        for name in ("experiment.json", "run_metadata.json"):
            path = args.output/name
            value = json.loads(path.read_text())
            value.update(teacher_content="existing_frozen_WAM_pretrained_Router_no_new_teacher_targets",
                action_objective="native_pi05_flow_matching_on_ULA_visual_geometry_short_demonstrations",
                ula_dataset=str(args.dataset.resolve()), ula_finetuning_source=str(source))
            path.write_text(json.dumps(value, indent=2))
        protocol["status"] = "finished"
    except BaseException as error:
        protocol.update(status="failed_or_interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stage_route_data.StageRouteTargetStore = original_store
        stage_route_data.StageRouteOpenPiDataset = original_val
        protocol["source_hashes_after"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        protocol["sources_unchanged"] = before == protocol["source_hashes_after"]
        (args.output/"ula_protocol.json").write_text(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
