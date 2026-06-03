# 从视频学动作：系统设计文档

## 目标

让 G1 机器人通过观察人类示范视频自主学习动作，不需要重新训练模型权重。
学习发生在推理阶段（Test-time Adaptation），迭代闭环收敛。

---

## 整体架构

```
① 人展示动作（视频 / USB 摄像头实时）
        │
        ├─ HMR2.0 → SMPL → G1 关节角序列        [几何层，逐帧]
        └─ VLM-A  → 动作语义描述                  [语义层，片段级]
                    "深蹲动作，重心偏右，双臂前伸"
        │
        ▼
② G1 在 MuJoCo 仿真中执行目标关节角序列
        │
        └─ 记录：实际关节角轨迹 + 执行截图
        │
        ▼
③ 双重打分
        │
        ├─ 几何分：L2(actual_joints, target_joints)  逐帧，定位薄弱片段
        └─ 语义分：VLM-B 对比两张截图
                    输入：[参考帧, 机器人执行帧]
                    输出："躯干前倾不足，左膝弯曲角度差约15度" + 0-10分
        │
        ▼
④ 定位薄弱片段 → 调整目标序列 → 回到②（迭代直到收敛）
```

---

## 各层组件

### 几何层（本地，已有基础）

| 组件 | 工具 | 状态 |
|------|------|------|
| 视频 → SMPL 参数 | WHAM（离线）/ HMR2.0（实时） | ✅ 已装好 |
| SMPL → G1 关节角 | SMPLRetargeter | ✅ 已有 |
| MuJoCo 执行 | mujoco-python，data.ctrl 注入 | 🔲 待接入 |
| 几何打分 | L2 / MPJPE，逐帧 | 🔲 待实现 |

### 语义层（VLM，API 调用，不占本地空间）

| 组件 | 用途 | 方案 |
|------|------|------|
| VLM-A | 理解示范动作的语义意图 | API（Qwen-VL / GPT-4V） |
| VLM-B | 对比参考帧与执行帧，输出差异描述 + 分数 | 同上，双图输入 |

VLM-B Prompt 模板：
```
左图是人类示范，右图是机器人执行结果。
从手臂、躯干、腿部三个维度各评分（0-10），
指出偏差最大的部位和估计角度差，输出 JSON。
```

---

## 实时 USB 摄像头支持

用于实时采集示范动作，替代预录视频。

```
USB Cam → HMR2.0（本地，RTX 5090 可实时）→ SMPL → G1 关节角
```

- 摄像头：/dev/video0，已验证可打开
- HMR2.0：已装好，单帧推理 ~30fps（已测试）
- 接口：逐帧调用，不依赖序列输入

---

## 迭代控制逻辑（伪代码）

```python
target_traj = hmr2_retarget(demo_video)      # 几何目标序列
semantic_desc = vlm_describe(demo_video)     # 语义描述（一次）

for iteration in range(MAX_ITER):
    actual_traj, frames = mujoco_execute(target_traj)

    geo_score, weak_segments = geometric_score(actual_traj, target_traj)
    sem_score, feedback = vlm_score(demo_frame, robot_frame)

    if geo_score > GEO_THRESH and sem_score > SEM_THRESH:
        break  # 收敛

    target_traj = adjust(target_traj, weak_segments, feedback)
```

---

## 当前进度

- [x] HMR2.0 实时推理验证（USB 摄像头，~30fps）
- [x] 设计文档
- [ ] MuJoCo G1 模型加载 + data.ctrl 接入
- [ ] 几何打分函数
- [ ] VLM API 接入（选定 key 后）
- [ ] 迭代控制循环

---

## 参考文献

本设计与以下 2024-2025 研究方向一致，核心共识：**冻结基础模型 + 推理阶段搜索/规划**。

---

### STORM (2025.12)
**Search-guided generaTive wOrld models for Robotic Manipulation**
arXiv: 2512.18477

用冻结的视频生成模型（世界模型）+ Monte Carlo Tree Search，在推理阶段搜索最优动作序列。
给定当前状态，让世界模型想象不同动作的未来画面，选择视觉结果最好的路径执行。

**与本设计的对应**：MuJoCo 仿真扮演世界模型的角色——不需要学一个神经网络世界模型，直接用物理引擎做前向仿真，用几何分 + VLM 分打分，在目标关节角空间里搜索。

---

### Instant Policy (ICLR 2025 Best Paper at Robot Learning Workshop)
**In-Context Imitation Learning via Graph Diffusion**
arXiv: 2411.12633

只需 1-2 个示范动作即可学会新任务，纯推理阶段 in-context conditioning，零梯度更新。
把示范动作和当前状态建模成图结构，用扩散模型生成对应的执行动作。

**与本设计的对应**：验证了"给几帧示范就能产生执行动作"的可行性。我们的方案是几何层（HMR2.0）而非图扩散，但目标相同——无需重训练，示范即学习。

---

### GPC: Generative Predictive Control (2025.02)
**Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling**
arXiv: 2502.00622

在冻结的扩散策略旁边挂一个动作条件世界模型。推理时，对候选动作做前向预测，比较预测结果和目标，选出最优动作执行。等于在不改权重的情况下给策略装了一个"预判"层。

**与本设计的对应**：我们的迭代循环（执行 → 打分 → 调整）是 GPC 逻辑的直接实现，区别是我们用 MuJoCo + VLM 替代了神经网络世界模型和预测打分。

---

### DAS: Diffusion Alignment as Sampling (ICLR 2025 Spotlight)
**Test-time Alignment of Diffusion Models without Reward Over-optimization**
arXiv: 2501.05803

用 Sequential Monte Carlo 在推理阶段对扩散模型的采样过程做 reward 对齐，不重训练基础模型。
解决了 reward over-optimization 问题（打分高但动作退化），保持生成多样性的同时向目标 reward 对齐。

**与本设计的对应**：我们的 VLM 打分 = reward 函数，迭代调整目标序列 = 在动作空间里做 reward-guided 搜索。DAS 的 SMC 采样是更严谨的数学框架，未来可以替换我们的启发式调整策略。

---

## 待决策

1. **VLM API**：Qwen-VL、GPT-4V 还是其他？
2. **MuJoCo G1 模型来源**：MuJoCo Menagerie / Unitree 官方 MJCF？
3. **调整策略**：薄弱片段用幅度缩放还是时序拉伸优先？
