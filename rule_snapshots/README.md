# 规则快照

保存**所有版本**的编译规则，不覆盖。

## 布局

```
rule_snapshots/<task>/<variant>.yaml   每个源目录一个文件
rule_snapshots/MANIFEST.json           latest 指针 + 每版 sha256/mtime/来源
```

`variant` 来自源目录名去掉前缀，例如：

```
results/demo_primitives_handover_block_rgbd_v2/tasks/handover_block/rule.yaml
  → rule_snapshots/handover_block/handover_block_rgbd_v2.yaml
```

## 默认版本

`MANIFEST.json` 的 `latest` 字段按 mtime 标注每个 `(task, arm)` 的默认版本，
但**所有历史版本都保留**，便于对比与回退。例：

```json
"handover_block/both": {
  "file": "handover_block/handover_block_rgbd_v2.yaml",
  "variant": "handover_block_rgbd_v2"
}
```

## 已知的多版本任务

| 任务 | 版本数 | latest | 说明 |
|---|---:|---|---|
| `handover_block` | 9 | `..._rgbd_v2` | v2 把 `receiver_precontact` 改到帧 141 |
| `handover_mic` | 2 | `..._raw_ep1_v1` | ep1 更新的那条示范 |
| `place_mouse_pad` | 2 | `..._left_v2_360` | v2_360 才解决朝向歧义（76%）|
| `place_fan` | 4 | `..._left_v4_360` | 同上 |
| `move_can_pot/right` | 2 | `..._right_v2` | |
| `move_pillbottle_pad` | 2 | `..._v1` | |
