# Humanoid — G1 AMP Locomotion

基于 [legged_lab](https://github.com/zitongbai/legged_lab) 的 G1-29DOF AMP 步态训练配置，以及 Isaac Lab → MuJoCo 的 sim2sim 验证工具。

## 依赖

| 仓库 | 版本 | 说明 |
|---|---|---|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | 2.3.x | 仿真框架 |
| [legged_lab](https://github.com/zitongbai/legged_lab) | main | 腿式机器人 RL 扩展 |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) | AMP branch | PPO + AMP 算法 |
| MuJoCo | ≥ 3.x | sim2sim 验证 |

conda 环境：`env_isaaclab`

## 文件说明

```
config/g1_amp/
  g1_amp_env_cfg.py     # 奖励函数、观测、命令范围配置
  rsl_rl_ppo_cfg.py     # PPO + AMP 算法超参数

sim2sim/
  sim2sim_g1_amp.py     # Isaac Lab checkpoint → MuJoCo 验证脚本
  scene_g1.xml          # MuJoCo 场景（仅用于调试，G1 MJCF 直接加载）
```

## 安装

将 `config/` 下的文件覆盖到 legged_lab 对应路径：

```bash
LEGGED_LAB=~/legged_lab/source/legged_lab/legged_lab

cp config/g1_amp/g1_amp_env_cfg.py \
   $LEGGED_LAB/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py

cp config/g1_amp/rsl_rl_ppo_cfg.py \
   $LEGGED_LAB/tasks/locomotion/amp/config/g1/agents/rsl_rl_ppo_cfg.py
```

## 训练

```bash
cd ~/legged_lab/scripts/rsl_rl

# 新训练
nohup conda run --no-capture-output -n env_isaaclab bash -c \
  "source ~/anaconda3/envs/env_isaaclab/etc/conda/activate.d/setenv.sh && \
   python train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --num_envs 4096" \
  > /tmp/amp_train.log 2>&1 &

# 从 checkpoint 恢复
nohup conda run --no-capture-output -n env_isaaclab bash -c \
  "source ~/anaconda3/envs/env_isaaclab/etc/conda/activate.d/setenv.sh && \
   python train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --num_envs 4096 \
   --resume --load_run <run_name> --checkpoint <model_N.pt>" \
  > /tmp/amp_train.log 2>&1 &
```

## 关键配置说明

### 奖励改动（相对 legged_lab 默认值）

| 项 | 原始值 | 当前值 | 原因 |
|---|---|---|---|
| `lin_vel_z_l2` | -0.2 | **-1.5** | 减少垂直弹跳 |
| `ang_vel_xy_l2` | -0.05 | **-0.15** | 抑制俯仰摇晃 |
| `dof_acc_l2` | -1e-7 | **-2.5e-7** | 减少关节抖动 |
| `action_rate_l2` | -0.005 | **-0.01** | 平滑动作输出 |
| `joint_deviation_arms` | -0.05 | **-0.2** | 约束手臂自然垂放 |

### AMP 改动

| 参数 | 原始值 | 当前值 | 原因 |
|---|---|---|---|
| `style_reward_scale` | 5.0 | **30.0** | dt=0.02 时 scale=5 只占总奖励 9%，AMP 几乎无效 |
| `disc_obs_buffer_size` | 100 | **200** | 增加 demo 样本多样性 |

## sim2sim 验证

```bash
cd sim2sim

# 可视化运行
python sim2sim_g1_amp.py --checkpoint <path/to/model_N.pt> --cmd_vx 0.5

# 无渲染压测
python sim2sim_g1_amp.py --checkpoint <path/to/model_N.pt> --no_render --steps 10000
```

checkpoint 路径通常在：
`~/legged_lab/scripts/rsl_rl/logs/rsl_rl/g1_amp/<run_name>/`
