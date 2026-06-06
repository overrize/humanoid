# 工作交接文档

## 目标

让 G1 机器人通过观察人类示范视频自主学习动作，**不重新训练模型权重**。
学习发生在推理阶段：示范视频 → 解析关节角 → 仿真执行 → 打分 → 迭代调整。

完整设计见 `DESIGN.md`，包含架构图、参考文献（STORM / Instant Policy / GPC / DAS）。

---

## 已完成的工作

### 1. HMR2.0 实时姿态估计

**文件：** `/home/rexcon/hmr2_realtime.py`

USB 摄像头 → HMR2.0 → SMPL 参数（body_pose 23×3×3，global_orient，betas），RTX 5090 上约 30fps。

```bash
python ~/hmr2_realtime.py          # 按 Q 退出
```

- hmr2 已安装，权重在 `~/.cache/4DHumans/`
- 摄像头需要在 `video` 组：`sudo usermod -aG video $USER`

---

### 2. dance_pipeline NPZ → WBT 格式转换

**文件：** `scripts/tools/dance_npz_to_wbt.py`

将 dance_pipeline 输出的 NPZ（ISL/DFS 关节序，11 个关键 body）转换为 BeyondMimic WBT 格式（BFS 关节序，30 个 body，完整 FK）。

```bash
python scripts/tools/dance_npz_to_wbt.py \
    --input  <dance_pipeline_output.npz> \
    --output <wbt_format.npz>
```

**关节序说明：**

| 名称 | 顺序 | 用途 |
|------|------|------|
| ISL / DFS | MuJoCo/Pinocchio 运动学树 DFS | dance_pipeline 输出、motion_player |
| BFS | Isaac Lab BFS 遍历 | WBT policy 输入、combined_g1_50fps.npz |

ISL = DFS = Pinocchio 输出顺序，三者完全相同。

---

### 3. WBT policy MuJoCo sim2sim（已有，已验证）

**文件：** `scripts/sim2sim/sim2sim_wbt_mujoco.py`
**权重：** `models/wbt_g1_v1/model_29999.pt`

```bash
# 原地站立（稳定参考帧：4557）
python scripts/sim2sim/sim2sim_wbt_mujoco.py \
    --checkpoint models/wbt_g1_v1/model_29999.pt \
    --motion_file /tmp/combined_g1_50fps.npz \
    --standing --frame 4557

# 跟踪轨迹（playback）
python scripts/sim2sim/sim2sim_wbt_mujoco.py \
    --checkpoint models/wbt_g1_v1/model_29999.pt \
    --motion_file <wbt_format.npz> \
    --playback
```

**已知问题：** standing 模式不指定 `--frame` 时会选到后退片段，务必加 `--frame 4557`。

---

### 4. 开环轨迹播放器（含几何误差记录）

**文件：** `sim2sim/motion_player.py`

直接把 dance_pipeline NPZ 的关节角序列注入 MuJoCo（不经过 WBT policy），记录实际执行轨迹和逐帧误差。

```bash
python sim2sim/motion_player.py \
    --traj  <dance_pipeline.npz> \
    --out   result.npz \
    --headless        # 无显示器时加此参数
```

输出 NPZ 包含：`target_dof`，`actual_dof`，`geo_error`（逐帧），`screenshots`。

**注意：** 纯开环，无平衡控制，动态动作会摔倒。目的是记录误差供打分，不是稳定复现。

---

### 5. AMP 起身训练调参（已提交）

- `save_interval` 50 → 500
- `TARGET_BASE_HEIGHT_PHASE3` 0.65 → 0.40
- 新增 `base_height_raw` 连续高度奖励（weight=3.0）
- 双 policy 联动脚本：`scripts/rsl_rl/play_getup_loco.py`（AMP 起身 → WBT 行走）

---

## 还没做的（按优先级）

### P1：demo 采集端到端链路

各段单独验证过，但没有串成一个脚本。完整流程：

```
视频文件 / USB 摄像头
    ↓ WHAM (离线) 或 HMR2.0 (实时)
    ↓ SMPL → SMPLRetargeter → dof_positions (ISL 序)
    ↓ dance_pipeline build_npz → dance NPZ
    ↓ dance_npz_to_wbt.py → WBT NPZ
    ↓ sim2sim_wbt_mujoco.py --playback
```

入口参考：`dance_pipeline/pipeline.py`（视频 → dance NPZ）

### P2：几何打分分析

`motion_player.py` 已输出 `geo_error`，还差：
- 找连续弱片段（sliding window，阈值约 0.1 rad）
- 按关节汇总最差的 top-k
- 输出可读报告

### P3：迭代调整循环

```python
for iteration in range(MAX_ITER):
    actual, geo = motion_player.run(target_traj)
    weak = find_weak_segments(geo)
    if not weak: break
    target_traj = adjust(target_traj, weak)   # 幅度缩放 or 时序拉伸
```

调整策略尚未实现，待决策。

### P4：VLM 语义打分

需要 API key（Qwen-VL / GPT-4V）。
Prompt 模板在 `DESIGN.md` 的"语义层"一节。
本地模型（Qwen2.5-VL-7B）因磁盘空间不足暂未下载。

---

## 关键路径和数据位置

| 资源 | 路径 |
|------|------|
| G1 MuJoCo XML（29 DOF）| `/home/rexcon/unitree_ros/robots/g1_description/g1_29dof.xml` |
| WBT policy 权重 | `models/wbt_g1_v1/model_29999.pt` |
| WBT 训练用 combined NPZ | `/tmp/combined_g1_50fps.npz`（9329 帧）|
| walk 测试 NPZ（WBT 格式）| `/tmp/walk_test.npz`（frames 766~946）|
| HMR2.0 权重 | `~/.cache/4DHumans/logs/.../epoch=35-step=1000000.ckpt` |
| AMP 训练数据 | `source/legged_lab/legged_lab/data/MotionData/g1_29dof/` |

---

## 仓库

`git@github.com:overrize/humanoid.git`，当前在 `main` 分支。
