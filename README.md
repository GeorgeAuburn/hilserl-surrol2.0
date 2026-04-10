# HIL-SERL SurRoL (Improved)

基于 [HIL-SERL](https://github.com/rail-berkeley/hil-serl) 的人类在环强化学习框架，在 [SurRoL v2](https://github.com/med-air/SurRoL) 仿真环境中训练 Franka Panda 机械臂完成拾取任务。

本仓库基于 [GeorgeAuburn/hilserl-surrol](https://github.com/GeorgeAuburn/hilserl-surrol) 改进，修复了导致训练不收敛的关键配置问题。

## 主要改进

| 改进项 | 原始值 | 改进值 | 影响 |
|--------|--------|--------|------|
| SAC 温度 | 0.01 | **1.0** | 修复探索能力丧失 |
| 折扣因子 | 0.97 | **0.99** | 有效视野 3.3s → 10s |
| 温度学习率 | 0.0003 | **0.001** | 加快自适应调节 |
| Actor 学习率 | 0.0003 | **0.0001** | 稳定策略更新 |
| 梯度裁剪 | 10.0 | **1.0** | 防止梯度爆炸 |
| 动作范围 | [-0.4, 0.4] | **[-1.0, 1.0]** | 匹配实际数据 |
| 硬编码路径 | `/home/zjj/...` | **自动配置** | 可移植部署 |

## 快速开始

```bash
# 克隆（包含子模块）
git clone --recursive https://github.com/LIJianxuanLeo/hilserl-surrol-improved.git
cd hilserl-surrol-improved

# 一键部署
bash setup.sh

# 激活环境
conda activate hilserl
cd lerobot

# 终端1: 启动 Learner
python -m lerobot.rl.learner --config_path train_config_gym_hil_touch.json

# 终端2: 启动 Actor（等 Learner 启动后）
python -m lerobot.rl.actor --config_path train_config_gym_hil_touch.json
```

## 系统要求

- Ubuntu 22.04
- NVIDIA GPU (RTX 3060 12GB 或更高)
- Geomagic Touch 触觉设备 + OpenHaptics SDK
- Conda (Python 3.12)

## 文档

- **[部署与训练指南](docs/DEPLOY_GUIDE.md)** - 完整的中文安装、配置、训练、调参指南

## 项目结构

```
hilserl-surrol-improved/
├── setup.sh                          # 一键部署脚本
├── docs/DEPLOY_GUIDE.md              # 详细部署指南（中文）
├── lerobot/                          # 修改版 LeRobot（SAC + HIL）
│   ├── train_config_gym_hil_touch.json       # 训练配置（已修复）
│   ├── env_config_gym_hil_touch_record.json  # 数据采集配置
│   ├── env_config_gym_hil_il.json            # 键盘采集配置
│   ├── franka_sim_touch_demos/               # 预采集演示数据（30 episodes）
│   └── src/lerobot/
│       ├── rl/                       # Actor-Learner 分布式训练
│       ├── policies/sac/             # SAC 策略（含离散动作修改）
│       ├── teleoperators/touch/      # Touch 触觉设备集成
│       └── processor/hil_processor.py # HIL 干预处理器
└── SurRoL_v2/                        # SurRoL 仿真环境
    ├── surrol/                       # 仿真核心（PyBullet）
    ├── haptic_src/                   # Touch 设备 SWIG 绑定
    └── ext/                          # 子模块（bullet3, pybullet_rendering）
```

## 训练架构

```
[Touch 触觉设备] → [Actor] ←gRPC:50051→ [Learner]
                      ↓                      ↓
              [SurRoL 仿真环境]         [SAC 策略更新]
              [PyBullet + Panda]        [Replay Buffer]
```

- **Button 1**：按住 = 人工接管，松开 = 策略自主
- **Button 2**：切换夹爪开合
- 人工干预 transitions 同时进入 online/offline buffer，batch 50/50 混合采样

## License

Apache-2.0
