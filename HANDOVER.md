# 工作交接文档

**项目**：G1 人形机器人倒地自主起身 + 行走双 policy 蒸馏  
**仓库**：`git@github.com:overrize/humanoid.git`（本目录）  
**日期**：2026-06-06

---

## 目标

让 G1 机器人在任意倒地姿态下自主站起，然后无缝切换到正常行走。  
最终形式：**单一蒸馏 policy**，同时具备起身和行走能力。

---

## 现状

### AMP 起身训练（进行中）
- 训练进度：**240000 / 552500 iter**（约 43%）
- 最新 checkpoint：`logs/rsl_rl/g1_amp_get_up/2026-06-04_20-40-27/model_240000.pt`
- 当前效果：约 30~50% 的智能体能成功站起（估计，未在 240k 做系统测试）
- 目标效果：80%+ 成功率再进行蒸馏

### WBT 行走 policy（已完成）
- 模型：`models/wbt_g1_v1/model_29999.pt`
- 训练任务：`Tracking-Flat-G1-v0`，参考数据为行走 MoCap（39 clips）
- 网络：`Linear(160, 512) → ELU → (512,256) → (256,128) → 29`

---

## 技术架构

### AMP 起身 policy
- 任务：`LeggedLab-Isaac-AMP-G1-GET-UP-v0`
- 环境配置：`source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_get_up_env_cfg.py`
- Agent 配置：`source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/agents/rsl_rl_ppo_get_up_cfg.py`
- 观测（570 dim = 114 × 5 步历史）：
  - `base_ang_vel(3) + root_local_rot_tan_norm(6) + joint_pos(29) + joint_vel(29) + actions(29) + key_body_pos_b(18)` × 5
- 关键 body：`left/right_ankle_roll, left/right_wrist_yaw, left/right_shoulder_roll`
- MoCap 数据：`data/MotionData/g1_29dof/amp/get_up/`（→ gitee MoCap 转换而来）

### WBT 行走 policy
- 任务：`Tracking-Flat-G1-v0`（`whole_body_tracking/` 子模块）
- 观测（160 dim）：`ref_jpos(29)+ref_jvel(29)+anchor_pos_b(3)+anchor_ori_b(6)+base_lin_vel(3)+base_ang_vel(3)+jpos_rel(29)+jvel_rel(29)+actions(29)`

### 双 policy 临时 play 脚本（参考用）
- `scripts/rsl_rl/play_getup_loco.py`：height < 0.5m 用 AMP，height ≥ 0.5m 用 WBT
- **效果不好**，仅作为结构参考，不用于生产

---

## 立即要做的事

### 1. 继续 AMP 训练（首要）

```bash
cd /home/rexcon/legged_lab
conda run -n env_isaaclab python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-GET-UP-v0 \
  --num_envs 3072 \
  --headless \
  --resume \
  --load_run 2026-06-04_20-40-27 \
  > /tmp/amp_getup_train.log 2>&1 &
```

- 继续跑到 400k~500k iter
- 每次查看状态：`ls -t logs/rsl_rl/g1_amp_get_up/2026-06-04_20-40-27/model_*.pt | head -1`
- **磁盘会积满**（192GB 盘剩约 3.5GB），每隔一段时间清理旧 checkpoint：
  ```bash
  ls -t logs/rsl_rl/g1_amp_get_up/2026-06-04_20-40-27/model_*.pt | tail -n +4 | xargs rm -f
  ```

### 2. 测试当前 checkpoint 效果

```bash
# 先停训练，再运行（GPU 不够同时跑）
conda run -n env_isaaclab python scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-G1-GET-UP-Play-v0 \
  --num_envs 48 \
  --load_run 2026-06-04_20-40-27 \
  --load_checkpoint model_240000.pt
```

### 3. 蒸馏（AMP 效果满意后）

目标：用 `DistillationRunner`（rsl_rl 已有）训练单一 student policy。

**蒸馏逻辑**：
- 以 AMP get-up env 为基础环境（机器人从倒地状态初始化）
- 每步根据 `root_height`：
  - `height < 0.5m` → AMP teacher 提供动作监督
  - `height ≥ 0.5m` → WBT teacher 提供动作监督
- Student 观测需包含两个 policy 都能用的状态信息
- 参考：`scripts/rsl_rl/play_getup_loco.py` 里的 obs 计算方式

---

## 绝对不要做的事

1. **不要删 `~/.cache/packman/chk/`**  
   这不是缓存目录，是 Isaac Sim 本体的软链接目标。删了之后 Isaac Sim 完全无法启动。  
   恢复方法：`repo.sh pull_extensions -c release`（耗时极长）

2. **不要同时跑训练和 play**  
   训练占 ~14GB GPU，play 占 ~5GB，总计超过 24GB 显存上限会 OOM。  
   测试前必须先 kill 训练进程。

3. **不要用 PKL MoCap 数据直接训练而不 clamp 关节角**  
   原始重定向数据 ankle_pitch 最大 2.644 rad，G1 限位仅 ±0.524 rad。  
   不 clamp 会导致训练约 60k iter 后 `normal expects all elements of std >= 0.0` 崩溃。  
   转换脚本：`scripts/pkl_getup_to_npz.py`（已含 clamp）。

4. **不要用 WBT tracking 训练起身**  
   BeyondMimic WBT 要求逐帧精确跟踪参考轨迹，起身动作接触时机因初始姿态不同而变化，  
   严格 tracking 无法泛化，效果极差。起身必须用 AMP。

5. **不要把 MoCap 数据 commit 到 git**  
   `data/MotionData/` 里的 NPZ 文件体积大（数百 MB），已在 `.gitignore` 中排除。

---

## 目录结构

```
legged_lab/
├── source/legged_lab/legged_lab/tasks/locomotion/
│   ├── amp/                        # AMP 起身任务
│   │   ├── config/g1/
│   │   │   ├── g1_amp_get_up_env_cfg.py   # 环境配置
│   │   │   ├── agents/rsl_rl_ppo_get_up_cfg.py  # agent 配置
│   │   │   └── __init__.py        # 任务注册
│   │   └── mdp/
│   │       ├── rewards.py         # 起身奖励函数
│   │       ├── events.py          # reset_from_ref, apply_force
│   │       ├── observations.py    # root_local_rot_tan_norm 等
│   │       ├── commands.py        # ForceCommand
│   │       └── curriculums.py     # force_level curriculum
│   └── velocity/                  # 速度跟踪任务（不用动）
├── whole_body_tracking/            # BeyondMimic WBT 子模块
├── models/wbt_g1_v1/model_29999.pt # WBT 行走 policy
├── logs/rsl_rl/g1_amp_get_up/     # 训练日志和 checkpoint
│   └── 2026-06-04_20-40-27/       # 当前有效 run（model_240000.pt）
├── scripts/rsl_rl/
│   ├── train.py                   # 通用训练脚本
│   ├── play.py                    # AMP play
│   ├── play_wbt.py                # WBT play
│   └── play_getup_loco.py         # 双 policy play（参考）
├── data/MotionData/g1_29dof/amp/get_up/  # MoCap NPZ（不在 git）
├── docs/getup_development_log.md  # 踩坑记录（必读）
└── DESIGN.md                      # 视频学习架构设计文档（另一条线）
```

---

## 必读文档

- `docs/getup_development_log.md`：详细踩坑记录，包含所有已知问题和解决方案
- `DESIGN.md`：更长远的视频学习方向设计（与当前起身任务平行的另一条线，暂未开发）

---

## 环境

- conda 环境：`env_isaaclab`
- Isaac Lab 路径：`/home/rexcon/isaac-lab/lab/IsaacLab/`
- GPU：RTX 4090 24GB
- 训练时 GPU 占用：~14GB（3072 envs）
