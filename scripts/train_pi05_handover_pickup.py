#!/usr/bin/env python3
"""Pure native pi0.5 LoRA on ULA handover pickup demonstrations, NO WorldLA.

Uses native openpi init/train/checkpoint functions. New run starts at optimizer
step zero from pi05_base, with an episode-disjoint validation set. No stage
Router, message adapter, WAM label or WorldLA checkpoint is imported or loaded.
"""
import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PI_ROOT = ROOT/"external/pi05"
sys.path[:0] = [str(ROOT), str(PI_ROOT), str(PI_ROOT/"src")]


def episode_indices(metadata, split):
    indices, offset, seeds = [], 0, []
    for row in metadata["episodes"]:
        length = int(row["length"])
        if row["split"] == split:
            indices.extend(range(offset, offset+length))
            seeds.append(row["seed"])
        offset += length
    if not indices:
        raise ValueError(f"Empty {split} episode split")
    return indices, seeds


def train_only_statistics(metadata, horizon):
    """Exact normalization of native Aloha delta chunks, never held-out data."""
    import h5py
    from openpi.shared.normalize import NormStats
    states, chunks = [], []
    mask = np.ones(14, dtype=bool)
    mask[[6,13]] = False
    for row in metadata["episodes"]:
        if row["split"] != "train":
            continue
        with h5py.File(row["path"]) as source:
            state = source["observation.state"][()].astype(np.float32)
            action = source["action"][()].astype(np.float32)
        if len(state) != row["length"]:
            raise ValueError("Raw demonstration length changed")
        future = np.minimum(np.arange(len(state))[:,None]+np.arange(horizon)[None,:], len(state)-1)
        delta = action[future].copy()
        delta[...,mask] -= state[:,None,mask]
        states.append(state)
        chunks.append(delta.reshape(-1,14))
    result = {}
    for key, values in [("state", states), ("actions", chunks)]:
        value = np.concatenate(values).astype(np.float64)
        result[key] = NormStats(mean=value.mean(axis=0), std=value.std(axis=0),
            q01=np.quantile(value,.01,axis=0), q99=np.quantile(value,.99,axis=0))
    return result


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-params", type=Path, default=PI_ROOT/"pi05_base/params")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--max-validation-batches", type=int, help="Debug only; omit for full validation")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps,args.batch_size,args.fsdp_devices,args.validation_interval,args.save_interval) < 1:
        parser.error("Training sizes and intervals must be positive")
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Use a fresh output directory; --resume only for this same native pi0.5 run")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.dataset.resolve()/"dataset.json").read_text())
    train_indices, train_seeds = episode_indices(metadata, "train")
    val_indices, val_seeds = episode_indices(metadata, "val")
    if set(train_seeds) & set(val_seeds):
        raise ValueError("Train/validation scene leakage")
    if not args.init_params.resolve().is_dir():
        raise FileNotFoundError(args.init_params)

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import torch
    from openpi.models import model as model_api
    from openpi.shared import nnx_utils, normalize
    from openpi.shared.runtime_dirs import configure_runtime_dirs
    from openpi.training import checkpoints, config as training_config, data_loader, optimizer, sharding

    configure_runtime_dirs(log_prefix="[pi05-pickup]")
    jax.config.update("jax_compilation_cache_dir", str(ROOT/".runtime_cache/jax_pi05_pickup"))
    base = training_config.get_config("multi_task_4")
    asset_id = metadata["repo_id"]
    assets_dir = args.output/"normalization"
    norm = train_only_statistics(metadata, base.model.action_horizon)
    normalize.save(assets_dir/asset_id, norm)
    data_factory = dataclasses.replace(base.data, repo_id=metadata["repo_id"],
        assets=training_config.AssetsConfig(assets_dir=str(assets_dir), asset_id=asset_id),
        base_config=dataclasses.replace(base.data.base_config,
            lerobot_data_root=metadata["lerobot_root"], prompt_from_task=True))
    from openpi.training.weight_loaders import CheckpointWeightLoader
    # Explicit LoRA-only update: native model, no WorldLA message adapter.
    freeze = nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*lora.*")))
    config = dataclasses.replace(base, name="ula_handover_pickup_pi05_lora", exp_name="native",
        data=data_factory, weight_loader=CheckpointWeightLoader(str(args.init_params.resolve())),
        freeze_filter=freeze, ema_decay=None, batch_size=args.batch_size, num_workers=0,
        num_train_steps=args.steps, fsdp_devices=args.fsdp_devices, seed=args.seed,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=min(200,max(1,args.steps//10)),
            peak_lr=args.learning_rate, decay_steps=max(3,args.steps), decay_lr=args.learning_rate*.1),
        wandb_enabled=False, overwrite=False, resume=args.resume)
    if args.batch_size % jax.device_count() or jax.device_count() % args.fsdp_devices:
        raise ValueError("Batch and FSDP sizes must divide visible devices")
    data_config = config.data.create(config.assets_dirs, config.model)
    raw = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    if len(raw) != metadata["frames"]:
        raise ValueError("Dataset frame count differs from collection metadata")
    transformed = data_loader.transform_dataset(raw, data_config)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(transformed, train_indices),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
        collate_fn=data_loader._collate_fn, generator=generator)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(transformed, val_indices),
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0,
        collate_fn=data_loader._collate_fn)
    if not len(train_loader):
        raise ValueError("Training split smaller than one global batch")

    spec = importlib.util.spec_from_file_location("ula_native_pi05_train", PI_ROOT/"scripts/train.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    checkpoint_dir = args.output/"checkpoints"
    manager, resuming = checkpoints.initialize_checkpoint_dir(checkpoint_dir, keep_period=None,
        overwrite=False, resume=args.resume)
    state, state_sharding = native.init_train_state(config, jax.random.key(args.seed), mesh, resume=resuming)
    if resuming:
        state = checkpoints.restore_state(manager, state, None)
    jax.block_until_ready(state)
    count = lambda tree: int(sum(np.prod(x.shape) for x in jax.tree.leaves(tree)))
    train_count = count(state.params.filter(config.trainable_filter))
    paths = [Path(__file__), PI_ROOT/"scripts/train.py", PI_ROOT/"src/openpi/training/config.py",
             ROOT/"adapters/handover/pickup_data.py"]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    dataset_hash = hashlib.sha256((args.dataset.resolve()/"dataset.json").read_bytes()).hexdigest()
    protocol_path = args.output/"protocol.json"
    if args.resume and protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        for key, actual in [("dataset_metadata_sha256",dataset_hash), ("target_steps",args.steps),
                            ("batch_size",args.batch_size), ("learning_rate",args.learning_rate)]:
            if previous[key] != actual:
                raise ValueError(f"Resume configuration mismatch: {key}")
    protocol = dict(schema="ula_native_pi05_pickup_v1", status="running", architecture="native_pi0.5_LoRA_only",
        WAM=False, WorldLA=False, stage_router=False, message_adapter=False,
        init_params=str(args.init_params.resolve()), dataset=str(args.dataset.resolve()),
        dataset_metadata_sha256=dataset_hash, target_steps=args.steps, completed_steps=int(state.step),
        batch_size=args.batch_size, learning_rate=args.learning_rate, fsdp_devices=args.fsdp_devices,
        trainable_parameters=train_count, total_parameters=count(state.params),
        train_episodes=len(train_seeds), validation_episodes=len(val_seeds),
        train_frames=len(train_indices), validation_frames=len(val_indices),
        train_seeds=train_seeds, validation_seeds=val_seeds,
        normalization="exact_train_only_native_Aloha_delta_action_chunks",
        normalization_asset_id=asset_id, instruction=metadata["instruction"],
        max_validation_batches=args.max_validation_batches,
        sources_before=before, checkpoint_layout="checkpoints/<completed_optimizer_steps>/{params,assets,train_state}")
    write_json(protocol_path, protocol)
    print(f"[pi05-pickup] PURE pi0.5 trainable={train_count:,} train={len(train_indices)} val={len(val_indices)} steps={int(state.step)}->{args.steps}", flush=True)
    ptrain = jax.jit(functools.partial(native.train_step,config),
        in_shardings=(replicated,state_sharding,data_sharding), out_shardings=(state_sharding,replicated),
        donate_argnums=(1,))
    def validate(current, rng, obs, actions):
        model = nnx.merge(current.model_def,current.params)
        model.eval()
        values = model.compute_loss(rng,obs,actions,train=False)
        return jnp.mean(values,axis=tuple(range(1,values.ndim)))
    pval = jax.jit(validate,in_shardings=(state_sharding,replicated,data_sharding,data_sharding),
                   out_shardings=replicated)
    def pack(batch):
        placed = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding,np.asarray(x)),batch)
        return model_api.Observation.from_dict(placed),placed["actions"]
    class DataConfigProxy:
        def data_config(self):
            return data_config
    iterator = iter(train_loader)
    started = time.perf_counter()
    try:
        for step in range(int(state.step)+1,args.steps+1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator=iter(train_loader)
                batch=next(iterator)
            with sharding.set_mesh(mesh):
                state, metrics = ptrain(jax.random.key(args.seed),state,pack(batch))
            if step == 1 or step%25 == 0 or step == args.steps:
                host = jax.device_get(metrics)
                row = dict(step=step, elapsed_s=time.perf_counter()-started,
                    **{key:float(value) for key,value in host.items() if np.asarray(value).ndim==0})
                if not np.isfinite(row["loss"]):
                    raise RuntimeError("Nonfinite training loss")
                with (args.output/"train_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row)+"\n")
                protocol.update(completed_steps=step, last_training_loss=row["loss"], elapsed_s=row["elapsed_s"])
                write_json(protocol_path,protocol)
                print(f"[pi05-pickup] step={step}/{args.steps} loss={row['loss']:.6f} elapsed={row['elapsed_s']:.1f}s",flush=True)
            if step%args.validation_interval==0 or step==args.steps:
                total, examples = 0.,0
                for index,val_batch in enumerate(val_loader):
                    if args.max_validation_batches is not None and index>=args.max_validation_batches:
                        break
                    # Pad the final validation batch for device divisibility,
                    # then evaluate it separately by per-example losses below.
                    original_count=len(val_batch["state"])
                    if original_count%jax.device_count():
                        padded=((original_count+jax.device_count()-1)//jax.device_count())*jax.device_count()
                        val_batch=jax.tree.map(lambda x:np.concatenate([x,np.repeat(x[-1:],padded-original_count,axis=0)],axis=0),val_batch)
                    obs,actions=pack(val_batch)
                    with sharding.set_mesh(mesh):
                        losses=np.asarray(jax.device_get(pval(state,jax.random.fold_in(jax.random.key(args.seed+1),index),obs,actions)))
                    loss=float(losses[:original_count].mean())
                    total+=loss*original_count
                    examples+=original_count
                row=dict(step=step,action_loss=total/examples,examples=examples)
                with (args.output/"validation_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row)+"\n")
                protocol["last_validation"]=row
                print(f"[pi05-pickup] validation step={step} loss={row['action_loss']:.6f} examples={examples}",flush=True)
            if step%args.save_interval==0 or step==args.steps:
                checkpoints.save_state(manager,state,DataConfigProxy(),step)
                manager.wait_until_finished()
                protocol["latest_checkpoint"]=str(checkpoint_dir/str(step))
                write_json(protocol_path,protocol)
        protocol.update(status="finished",completed_steps=args.steps)
    except BaseException as error:
        protocol.update(status="failed_or_interrupted",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manager.wait_until_finished()
        manager.close()
        protocol["sources_after"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        protocol["sources_unchanged"]=before==protocol["sources_after"]
        write_json(protocol_path,protocol)


if __name__ == "__main__":
    main()
