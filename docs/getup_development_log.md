# G1 倒地起身开发记录

## 目标

让 G1 机器人在 BeyondMimic WBT locomotion policy 基础上，增加从倒地状态自主站立的能力。

---

## 技术路线

### 最终方案：AMP get-up + WBT locomotion 双 policy 切换

- **行走**：BeyondMimic WBT（Whole-Body Tracking）policy，已训练完成
- **起身**：AMP（Adversarial Motion Prior）policy，使用真实 MoCap 数据作为 discriminator 训练数据
- **切换**：运行时根据 root height 阈值切换（倒地 → AMP 起身，站稳 → WBT 行走）

> 参考：Gitee [legged_lab](https://gitee.com/chaomingsanhua/legged_lab) 的双蒸馏方案（`LeggedLab-Isaac-AMP-G1-GET-UP-v0`），其两个 expert 均为 AMP，本项目改为 AMP get-up + WBT locomotion。

---

## 踩坑记录

### 坑 1：BeyondMimic WBT 严格 tracking 不适合起身

**现象**：用 WBT tracking 训练起身 policy，机器人完全起不来，远差于开源 AMP 效果。

**原因**：
- WBT tracking 要求 policy 逐帧精确复现参考轨迹
- 起身动作高度依赖接触时机（何时手撑地、何时脚发力），而接触时机因初始姿态不同而变化
- 单条 MoCap 轨迹无法覆盖所有初始条件，严格 tracking 无法泛化

**教训**：起身用 AMP（discriminator 学"什么动作自然"，policy 有自由度自己摸索路径）；行走用 WBT tracking（轨迹误差局部，tracking 稳定）。

---

### 坑 2：MoCap 数据关节角度严重超出 G1 物理限位

**现象**：用 PKL MoCap 数据训练时，约第 64500 iter 报错 `RuntimeError: normal expects all elements of std >= 0.0`，训练崩溃；play 时机器人抽搐。

**原因**：
- Gitee MoCap 数据经人体→G1 重定向后，部分关节角度严重超出物理限位
- 最典型：`right_ankle_pitch_joint` 最大达 2.644 rad，而 G1 限位仅 ±0.524 rad（超出 5 倍）
- `ankle_roll` 最大 1.915 rad，限位仅 ±0.262 rad（超出 7 倍）
- 这些极端角度导致观测值爆炸 → 网络 std 变负 → `normal()` 崩溃

**修复**：在 PKL→NPZ 转换时（`scripts/pkl_getup_to_npz.py`）clamp 所有关节角度到 MuJoCo XML 中的物理限位：

```python
for i, name in enumerate(DFS_NAMES):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
    dof_dfs[:, i] = np.clip(dof_dfs[:, i], lo, hi)
```

**注意**：clamp 会改变脚部参考姿态，对 WBT tracking 影响大（轨迹失真），对 AMP 影响小（只用于 discriminator 风格学习）。

---

### 坑 3：合成轨迹起身数据被行走数据淹没

**现象**：将合成起身轨迹（351 帧）与行走数据（9680 帧）合并训练 WBT，policy 几乎不学起身（3.6% 概率采样到起身帧）。

**修复**：使用纯起身数据（1577 帧，10 条 MoCap clip）单独训练起身 policy，不混入行走数据。

---

### 坑 4：z_offset 计算基于 geom 导致错误

**现象**：用 MuJoCo geom 最低点计算地面高度偏移，ankle 极端角度把 sensor sphere 放到 z=-0.55m，z_offset 补偿过大，导致参考轨迹整体偏高。

**修复**：改为用 `root_pos z` 的最小值作为基准，偏移到 `SUPINE_PELVIS_Z = 0.12m`（机器人仰卧时 pelvis 的期望高度）：

```python
SUPINE_PELVIS_Z = 0.12
min_rp_z = float(rp_r[:, 2].min())
z_offset = SUPINE_PELVIS_Z - min_rp_z
rp_r[:, 2] += z_offset
```

---

### 坑 5：CUDA 状态损坏（训练被 force-kill 后）

**现象**：强制终止训练后重新训练报 `RuntimeError: CUDA unknown error`。

**原因**：`nvidia_uvm` 内核模块状态损坏。

**修复**：
```bash
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
```

---

### 坑 6：多行 bash 命令解析错误

**现象**：用反斜杠续行的多行命令传给 Python `-c` 时，shell 解析异常，出现 `unrecognized arguments`。

**修复**：用独立 `.py` 脚本文件替代 `-c` 参数传入的内联代码；或将命令写成单行。

---

## 数据流

### PKL → NPZ 转换（MoCap 数据）

```
Gitee MoCap PKL (30fps, DFS 关节顺序)
  → 关节角 clamp 到 G1 物理限位
  → 重采样到 50fps (CubicSpline + SLERP)
  → DFS → BFS 关节顺序重排
  → MuJoCo FK 计算 30 个 body 的 pos/quat/lin_vel/ang_vel
  → 保存为 BeyondMimic NPZ 格式
```

脚本：`scripts/pkl_getup_to_npz.py`
输出：`/tmp/getup_mocap_g1_50fps.npz`（10 clips，1577 帧，31.5s @ 50fps）

### 关节顺序对应

- **DFS**（MuJoCo/GMR 顺序）：PKL `dof_pos` 的存储顺序
- **BFS**（Isaac Lab 顺序）：NPZ `joint_pos` 和 AMP discriminator 的顺序
- 转换数组 `BFS_TO_DFS[i]` = BFS 第 i 个关节在 DFS 中的索引

---

## 训练配置

### AMP Get-up（当前）

- Task：`LeggedLab-Isaac-AMP-G1-GET-UP-v0`
- 脚本：`scripts/rsl_rl/train.py`
- Discriminator：使用 10 条 MoCap clip（weight=1.0 的部分）
- 奖励：`target_base_height`（目标 0.75m，phase3 阈值 0.65m）+ `target_orientation` + `ang_vel_xy` + `lin_vel_xy` + regularization
- 终止：仅 `time_out`（不早期终止，让 robot 有机会自行站起）
- `apply_force`：reset 时施加外力，增加鲁棒性
- AMP style_reward_scale=50.0，task_style_lerp=0.5

### WBT Locomotion（已完成）

- Task：`Tracking-Flat-G1-v0`
- 脚本：`scripts/rsl_rl/train_wbt.py`
- 参考数据：`/tmp/getup_mocap_g1_50fps.npz`（后续应换回行走数据）

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `scripts/pkl_getup_to_npz.py` | PKL MoCap → BeyondMimic NPZ 转换，含 joint clamp |
| `scripts/generate_getup_npz.py` | 手工合成起身轨迹（备用，无 MoCap 时使用） |
| `source/.../amp/config/g1/g1_amp_get_up_env_cfg.py` | AMP 起身 env 配置（从 Gitee 移植） |
| `source/.../amp/config/g1/agents/rsl_rl_ppo_get_up_cfg.py` | AMP 起身 agent 配置 |
| `source/.../amp/mdp/events.py` | `reset_from_ref`, `apply_force` 等 event 函数 |
| `source/.../amp/mdp/commands.py` | `ForceCommandCfg` |
| `source/.../amp/mdp/curriculums.py` | `force_level` curriculum |
| `source/.../amp/mdp/rewards.py` | 新增起身奖励函数 |

---

## 参考

- [legged_lab (Gitee)](https://gitee.com/chaomingsanhua/legged_lab)：AMP get-up + 双蒸馏完整实现
- [BeyondMimic / whole_body_tracking](https://github.com/zitongbai/legged_lab)：WBT locomotion 基础
- G1 MuJoCo XML：`/home/rexcon/unitree_ros/robots/g1_description/g1_29dof.xml`
- MoCap PKL 数据：`/tmp/gitee_legged_lab/.../data/MotionData/g1_29dof/amp/get_up/`
