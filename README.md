# Bolt

基于 **Isaac Lab** 的人形机器人足球颠球强化学习项目。

项目地址：**https://github.com/MAZHIPENG001/Bolt**

# 1. PPO 训练

训练脚本：**scripts/skrl/train.py**

## 1.1 基本训练
```bash
python scripts/skrl/train.py \
    --task Bolt-Soccer-Teacher-v0
```
```
python scripts/skrl/train.py --help
```
## 1.2 环境:Bolt-Soccer-Teacher-v0
### 1.2.1 单卡训练/继续训练
```bash
python scripts/skrl/train.py \
    --task Bolt-Soccer-Teacher-v0 \
    --max_iterations 10000 \
    --num_envs 4096 \
    --headless \
    --checkpoint logs/skrl/inreal_v2_soccer/2026-07-13_10-59-36_ppo_torch_origin/checkpoints/agent_96000.pt
```
### 1.2.2 多卡训练
```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    scripts/skrl/train.py \
    --task Bolt-Soccer-Teacher-v0 \
    --headless \
    --num_envs 7192 \
    --distributed
```
```bash
NCCL_PROTO=LL \
NCCL_ALGO=Ring \
torchrun \
    --standalone \
    --nproc_per_node=4 \
    scripts/skrl/train.py \
    --task Bolt-Soccer-Teacher-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 10000 \
    --distributed \
    --checkpoint 
```
```bash
NCCL_DEBUG=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
NCCL_PROTO=LL \
NCCL_ALGO=Ring \
torchrun \
    --standalone \
    --nproc_per_node=4 \
    scripts/skrl/train.py \
    --task Bolt-Soccer-Teacher-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 100000 \
    --distributed \
    --checkpoint 
```
日志会出现 `carb.cudainterop.plugin` 警告，严重时会导致非法显存访问。当前服务器要使用
物理卡 0、1、2、3，只需设置 `--nproc_per_node=4`，`LOCAL_RANK` 会自动选择这四张卡。
## 1.3 环境:Bolt-Soccer-Depth-v0
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

NCCL_DEBUG=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
NCCL_PROTO=LL \
NCCL_ALGO=Ring \
torchrun \
    --standalone \
    --nproc_per_node=4 \
    scripts/skrl/train.py \
    --task Bolt-Soccer-Depth-v0 \
    --headless \
    --num_envs 1024 \
    --max_iterations 50000 \
    --distributed \
    --checkpoint 
```
## 1.4 环境:Bolt-Soccer-DepthImage-v0
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

NCCL_DEBUG=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
NCCL_PROTO=LL \
NCCL_ALGO=Ring \
torchrun \
    --standalone \
    --nproc_per_node=4 \
    scripts/skrl/train.py \
    --task Bolt-Soccer-DepthImage-v0 \
    --headless \
    --num_envs 1024 \
    --max_iterations 50000 \
    --distributed \
    --checkpoint 
```
```
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

NCCL_PROTO=LL \
NCCL_ALGO=Ring \
torchrun \
    --standalone \
    --nproc_per_node=4 \
    scripts/skrl/train.py \
    --task Bolt-Soccer-DepthImage-v0 \
    --headless \
    --num_envs 1024 \
    --distributed
```
## 1.4 查看训练信息和曲线

训练启动时，终端会打印任务、设备、并行环境数、观测/动作空间、PPO 批次大小、学习率、日志目录以及各模型的参数量。训练过程中会定期打印 reward、episode、loss、学习率和环境上报的全部标量。

`--log_interval` 的单位是 PPO policy iteration，默认每 100 次迭代打印一次并写入一次 TensorBoard。需要更密集的日志时，例如每 10 次迭代记录一次：

```bash
python scripts/skrl/train.py \
    --task Bolt-Soccer-Teacher-v0 \
    --headless \
    --log_interval 10
```

训练脚本启动时会打印可直接执行的 TensorBoard 命令。也可以在另一个终端手动启动：

```bash
conda activate isaaclab
tensorboard \
    --logdir logs/skrl/inreal_v2_soccer \
    --port 6006
```

然后浏览器访问 `http://localhost:6006`，即可实时查看 reward、policy/value loss、episode 长度、学习率和各奖励项曲线。传入 `--log_interval 0` 可以同时关闭终端指标和 TensorBoard 标量记录。

训练时录制视频会额外消耗：**GPU+显存+渲染时间**,因此大规模训练时建议关闭**--video**。

## 1.5 深度相机策略

`Bolt-Soccer-Depth-v0` 在 `torso_link` 上挂载 96×72 的批量深度相机。Actor 的球位置、
球速度和脚到球向量均由深度点云估计，不读取足球刚体真值。奖励和课程统计也使用同一份
深度球状态；击球、交替击球和飞行周期事件由深度轨迹及机器人正向运动学推断，不使用按足球
对象过滤的仿真接触信号。Critic 和终止条件仍可在训练时使用仿真真值。深度任务使用独立的
PPO 配置：Actor 只接收 `OBSERVATIONS`，Critic 才接收 `STATES`。深度渲染开销较大，建议先从
较小规模开始：

```bash
python scripts/skrl/train.py \
    --task Bolt-Soccer-Depth-v0 \
    --num_envs 32 \
    --headless
```

修复前使用 `STATES` 训练出的 Depth checkpoint 实际依赖 Critic 真值，不能作为深度模型继续训练
或部署；请使用当前配置从头训练。启动训练和回放时会自动校验这一输入约束。

`Bolt-Soccer-DepthImage-v0` 是独立的端到端视觉环境，不会改动上述环境。Actor 直接接收两帧
归一化的 96×72 原始深度图，以及 IMU、关节状态、脚部正向运动学和动作历史；其中没有球检测、
点云处理、球位置/速度估计或球接触推断。训练期的奖励和 Critic 仍可使用仿真真值，但这些输入
不会进入 Actor，也不能带到真机。该环境使用独立 checkpoint，需从头训练：

```bash
python scripts/skrl/train.py \
    --task Bolt-Soccer-DepthImage-v0 \
    --num_envs 32 \
    --headless
```

训练和回放脚本会为该任务自动启用相机。开始训练前，可用校准脚本比较深度估计与仿真真值；
真值只在此诊断脚本中用于计算误差，不进入 Actor：

```bash
python scripts/tools/check_depth_ball.py \
    --num_envs 1 \
    --steps 20 \
    --headless
```

若要在 Isaac Sim 内实时查看深度图，不能使用 `--headless`。下面的命令会打开
`Bolt Depth Camera` 窗口并持续运行，关闭仿真器即可结束：

```bash
python scripts/tools/check_depth_ball.py \
    --num_envs 1 \
    --camera_id 0 \
    --show_depth \
    --steps 0
```

预览图中近处为亮色、远处为暗色、无效或超过 4 m 的像素为黑色。也可以在 Isaac Sim 中打开
`Tools → Robotics → Camera Inspector`，选择 `DepthCamera` 后创建 Viewport；该 Viewport 用于检查
相机姿态和视场，显示的不是深度数值图。

当前检测器使用已知足球半径、深度表面法向和三维 Hough 投票估计球心，并通过关节正向运动学
剔除机器人自身点云。若实际场景还包含尺寸相近的球形物体，应进一步用仿真真值生成标签，训练
深度图球检测网络，再用网络输出替换该几何检测器。


# 2. 模型回放与录制

训练完成后使用：**scripts/skrl/play.py**加载模型。

```bash
python scripts/skrl/play.py \
    --task Bolt-Soccer-Teacher-v0 \
    --num_envs 1 \
    --checkpoint logs/skrl/inreal_v2_soccer/2026-08-18_14-15-06_ppo_torch_origin/checkpoints/agent_72000.pt \
    --video \
    --video_length 800 \
    --headless
```
```bash
python scripts/skrl/play.py \
    --task Bolt-Soccer-Depth-v0 \
    --num_envs 1 \
    --checkpoint logs/skrl/inreal_v2_soccer_depth/2026-08-20_10-35-44_ppo_torch_depth_only/checkpoints/agent_24000.pt \
    --video \
    --video_length 800 \
    --headless
```
模型权重上传与下载:  **logs/README.md**

# 3. 项目简介

当前项目的主要强化学习环境为： **Bolt-Soccer-Teacher-v0**

任务目标是训练 **Inreal V2 人形机器人**使用双脚控制足球，在保持身体平衡的同时完成连续足球颠球动作。

整个环境基于 Isaac Lab： **ManagerBasedRLEnv** 构建。

当前环境主要包含：

* Inreal V2 人形机器人
* 足球刚体
* 地面
* 机器人碰撞传感器
* 左右脚与足球接触传感器
* 5 帧历史观测
* Actor / Critic 非对称观测
* 关节位置控制
* 物理参数随机化
* 初始状态随机化
* 足球颠球奖励函数
* 分阶段课程学习
* PPO 强化学习训练



# 4. 项目目录

```text
Bolt/
├── data/
│   └── assets/
│       └── Inreal_v2/
│           ├── img/
│           ├── mesh/
│           ├── urdf/
│           ├── usd/
│           └── xml/
│
├── scripts/
│   ├── list_envs.py
│   ├── random_agent.py
│   ├── zero_agent.py
│   │
│   ├── skrl/
│   │   ├── train.py
│   │   └── play.py
│   │
│   └── tools/
│       └── fix_inreal_usd_references.py
│
└── source/
    └── Bolt/
        ├── Bolt/
        │   └── tasks/
        │       └── manager_based/
        │           └── bolt/
        │               ├── agents/
        │               │   └── skrl_ppo_cfg.yaml
        │               ├── mdp/
        │               │   ├── curriculum.py
        │               │   ├── events.py
        │               │   ├── juggle_state.py
        │               │   ├── observations.py
        │               │   ├── rewards.py
        │               │   └── terminations.py
        │               ├── assets_cfg.py
        │               └── bolt_env_cfg.py
        │
        ├── config/
        │   └── extension.toml
        └── setup.py
```



# 5. 环境要求

本项目需要在已经正确安装的 **Isaac Lab 2.1.1** 环境中运行。

Isaaclab环境安装:  [IsaacLab/v2.1.1](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/setup/installation/pip_installation.html)
## 5.1 Isaac Lab 安装
### 5.1.1 Installing Isaac Sim
```bash
conda create -n isaaclab python=3.10
conda activate isaaclab
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
# Verifying the Isaac Sim installation
isaacsim
```
### 5.1.2 Installing Isaac Lab
```bash
git clone git@github.com:isaac-sim/IsaacLab.git
sudo apt install cmake build-essential
pip install -e .
./isaaclab.sh --install

# Verifying the Isaac Lab installation
# Option 1: Using the isaaclab.sh executable
# note: this works for both the bundled python and the virtual environment
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

# Option 2: Using python in your virtual environment
python scripts/tutorials/00_sim/create_empty.py
```

## 5.2 克隆项目
使用 HTTPS：

```bash
git clone https://github.com/MAZHIPENG001/Bolt.git
cd Bolt
```

## 5.3 检查 Isaac Lab 环境
```bash
python -c "import isaaclab; print('Isaac Lab OK')"
```
正常输出：**Isaac Lab OK**
## 5.4 安装 Bolt
进入项目根目录：
```bash
cd ~/Bolt
python -m pip install -e source/Bolt
```
检查：
```bash
python -c "import Bolt; print(Bolt.__file__)"
```

# 6 环境测试
## 6.1. 查看已经注册的环境

```bash
python scripts/list_envs.py
```

## 6.2 Zero Agent 测试
```bash
python scripts/zero_agent.py \
    --task Bolt-Soccer-Teacher-v0 \
    --num_envs 1
```
* USD 是否能够正确加载
* Robot articulation 是否正常
* 初始姿态是否正常
* 足球是否能够正常生成
* Contact Sensor 是否正常
* 环境是否能够正常 reset
* 仿真是否稳定

## 6.3 Random Agent 测试

```bash
python scripts/random_agent.py \
    --task Bolt-Soccer-Teacher-v0 \
    --num_envs 1
```



# 7. 详细信息
## 7.1 物理模型
### 7.1.1 机器人模型
#### 7.1.1.1 机器人 USD：
```text
data/assets/Inreal_v2/usd/
└── inreal_v2_entity2_0527_robot_isaaclab.usd
```
机器人配置位于：**source/Bolt/Bolt/tasks/manager_based/bolt/assets_cfg.py**
##### 腿部关节：
```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint

right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
```
##### 上半身关节：
```text
waist_pitch_joint
waist_yaw_joint

left_shoulder_pitch_joint
left_shoulder_roll_joint
left_elbow_joint

right_shoulder_pitch_joint
right_shoulder_roll_joint
right_elbow_joint
```

##### 7.1.1.2 足球模型

足球使用 Isaac Lab： **RigidObjectCfg** 进行定义。

主要参数：
```text
半径：0.11 m
质量：0.43 kg
```
初始位置大约为：
```text
x = 0.4
y = 0
z = 1.0
```
<!-- 足球配置位于：
```text
source/Bolt/Bolt/tasks/manager_based/bolt/assets_cfg.py
``` -->
## 7.2 电机与控制

机器人使用: **DCMotorCfg**

配置不同关节组的：

* stiffness
* damping
* effort limit
* velocity limit
* armature

主要包括：

```text
hip_pitch
hip_roll
hip_yaw
knee
ankle_pitch
ankle_roll
waist
shoulder_pitch
shoulder_roll
elbow
```
机器人当前使用： **Joint Position Action**, 即策略网络输出关节位置控制量。

## 7.3 Observation

环境采用：**HISTORY_LENGTH = 5**,即策略输入包含最近 **5 帧历史状态**。

### 7.3.1 Policy Observation

Policy 可获得的信息主要包括：

```text
机器人角速度
重力方向
关节位置
关节速度
上一时刻动作

足球相对机器人位置
足球相对机器人速度

左右脚位置
左右脚姿态
左右脚与足球之间的位置关系
```

在 `Bolt-Soccer-Teacher-v0` 中，上述足球信息来自仿真真值；在
`Bolt-Soccer-Depth-v0` 中，Actor 的对应信息来自深度图，输入维度和排列保持不变。

并且 Policy Observation 中加入了一定的噪声,目的是提高策略对传感器误差和现实环境扰动的鲁棒性。



### 7.3.2 Critic Observation

Critic 使用更加完整的状态信息。

除了 Policy Observation 之外，还可以包含：

```text
机器人 Root Position
机器人线速度
更多脚部速度信息
下一次应该使用哪只脚
足球完整状态
其他 Privileged State
```

因此整个系统属于：

```text
Asymmetric Actor-Critic
```

即：

```text
Actor
只能看到部署时能够获得的状态
        │
        ▼
      Action

Critic
可以使用更完整的仿真真值
        │
        ▼
     Value
```

这种方法通常有利于进行：**Sim-to-Real**训练。



## 7.4 Domain Randomization

为了提高模型鲁棒性，环境加入了多种 Domain Randomization。

包括：
### 机器人

```text
脚部摩擦系数
Torso 质心
PD stiffness
PD damping
关节 friction
关节 armature
默认关节位置
虚拟脚部坐标系
```

### 足球

```text
摩擦系数
恢复系数
质量
惯量
```

### Reset

每次 Episode Reset 时，还会随机化：

```text
机器人位置
机器人朝向
机器人关节位置
机器人关节速度
足球状态
```

### 7.5 外部扰动

训练过程中还会周期性向机器人施加随机速度扰动。

主要包括：

```text
x
y
z
roll
pitch
yaw
```

可以迫使策略学会在受到扰动后恢复平衡。

## 7.6 Reward 设计

项目的核心任务:**稳定站立+控制足球+交替双脚+持续颠球**,因此奖励函数由多个部分组成。


### 7.6.1 足球相关奖励

例如：

```text
foot_ball_distance
kick
alternating_kick
kick_quality
juggle_streak
flight_cycle
ball_height
ball_approach_vel
ball_upward_vel
```

鼓励机器人：

* 脚靠近足球
* 成功触球
* 将足球向上踢
* 让球真正离脚并完成“上升 -> 最高点 -> 下降”的飞行周期
* 保持足球在合理高度
* 连续颠球



### 7.6.2 双脚交替

项目特别限制机器人长期只使用同一只脚。

相关 Reward / Penalty 包括：

```text
alternating_kick
same_foot_kick_penalty
wrong_foot_proximity_penalty
double_foot_proximity_penalty
double_contact_penalty
```

目标是学习：**左脚->右脚->左脚->右脚->...**形式的连续颠球策略。


### 7.6.3 身体稳定性

例如：

```text
robot_upright
stable_standing
torso_upright
torso_lean_penalty
base_height_penalty
waist_pitch_penalty
undesired_contacts
```

鼓励机器人：**保持站立+避免长期前倾/后仰和深蹲+减少不必要碰撞**

### 7.6.4 动作平滑

包括：

```text
action_rate_l2
action_rate_2nd_l2
arm_action_rate_penalty
leg_action_rate_penalty
joint velocity penalty
```
主要用于避免:**动作突变,关节剧烈抖动,高频震荡**,使策略输出更加平滑。

## 7.7 Curriculum Learning
当前任务使用分阶段课程学习：**Phase 1a --(训练达到一定条件)--> Phase 1b**
综合判断：**双脚交替踢球成功率+左脚踢球情况+右脚踢球情况+左右脚平衡程度+Episode Length+Global Steps**,而不是简单根据固定训练步数切换。
### 7.7.1 Curriculum 意义

训练初期如果直接要求机器人：**连续稳定双脚颠球**任务难度过高。

因此训练逻辑更接近：**首先学会站住->接近足球->正确触球->将足球踢起->使用另一只脚->连续交替->稳定颠球**


## 7.8 Episode 终止条件

Episode 在以下情况下可能结束：

**达到最大时间 || 机器人高度过低 || 机器人姿态倾斜过大 || 上半身发生非法接触 || 足球落地 || 足球距离机器人太远 || 足球高度过高 || 足球跑到机器人后方 || 连续只使用同一只脚 || 双脚长时间夹住足球 || 长时间未形成下一次有效踢球周期**

默认：**episode_length_s = 15.0**,即一个 Episode 最长：**15 秒**

## 7.9 仿真参数

当前主要参数：

```text
num_envs      = 256
env_spacing   = 5.0 m

sim.dt        = 0.005 s
decimation    = 4

episode       = 15 s
```

**Physics Frequency = 1 / 0.005 = 200 Hz**

策略控制周期：**0.005 × 4 = 0.02 s**

因此 Policy Frequency 为：**50 Hz**


## 7.10 PPO 网络

配置文件：**source/Bolt/Bolt/tasks/manager_based/bolt/agents/skrl_ppo_cfg.yaml**

### 主要超参数
```yaml
rollouts: 24
learning_epochs: 5
mini_batches: 4
discount_factor: 0.99
lambda: 0.95
learning_rate: 1.0e-3
ratio_clip: 0.2
value_clip: 0.2
entropy_loss_scale: 0.005
grad_norm_clip: 1.0
```
随机种子：**seed = 42**

## 7.11 日志保存位置

默认训练日志目录：**logs/skrl/inreal_v2_soccer/**

每次训练会创建一个带时间戳的目录。

```text
logs/
└── skrl/
    └── inreal_v2_soccer/
        └── 2026-07-13_09-00-00_ppo_torch_origin/
            ├── checkpoints/
            ├── params/
            │   ├── agent.yaml
            │   ├── env.yaml
            │   ├── agent.pkl
            │   └── env.pkl
            └── ...
```

其中：
* **env.yaml** 保存环境参数;

* **agent.yaml** 保存 PPO 参数。


## 7.12 主要源码

### 7.12.1 环境配置

```text
source/Bolt/Bolt/tasks/manager_based/bolt/bolt_env_cfg.py
```

包含：

```text
Scene
Observation
Action
Event
Reward
Termination
Curriculum
Simulation
```

### 7.12.2 机器人和足球

```text
source/Bolt/Bolt/tasks/manager_based/bolt/assets_cfg.py
```
包含：

```text
Robot USD
Actuator
PD 参数
Robot Initial State
Soccer
```
### 7.12.3 Observation
```text
source/Bolt/Bolt/tasks/manager_based/bolt/mdp/observations.py
```
### 7.12.4 Reward
```text
source/Bolt/Bolt/tasks/manager_based/bolt/mdp/rewards.py
```
### 7.12.5 Event / Domain Randomization
```text
source/Bolt/Bolt/tasks/manager_based/bolt/mdp/events.py
```
### 7.12.6 Curriculum
```text
source/Bolt/Bolt/tasks/manager_based/bolt/mdp/curriculum.py
```
### 7.12.7 Termination
```text
source/Bolt/Bolt/tasks/manager_based/bolt/mdp/terminations.py
```
### 7.12.8 足球颠球状态管理
```text
source/Bolt/Bolt/tasks/manager_based/bolt/mdp/juggle_state.py
```
### 7.12.9 PPO 配置
```text
source/Bolt/Bolt/tasks/manager_based/bolt/agents/skrl_ppo_cfg.yaml
```
# 8. 项目状态

本项目目前仍在持续开发中。

当前主要研究内容包括：

* Inreal V2 在 Isaac Lab 中的仿真
* 人形机器人足球控制
* 足球颠球
* 双脚交替策略
* Privileged Teacher
* Asymmetric Actor-Critic
* Domain Randomization
* Curriculum Learning
* PPO 强化学习
* 大规模并行仿真
* Sim-to-Real
后续环境参数、奖励函数、课程学习策略以及 PPO 超参数可能持续调整。

# 9. 致谢

本项目基于以下开源项目与工具：

* [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
* [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
* [skrl](https://github.com/Toni-SM/skrl)
* [Gymnasium](https://gymnasium.farama.org/)

本项目将 Inreal V2 足球任务适配到 Isaac Lab 的 Manager-Based 强化学习框架中，用于人形机器人足球控制与连续颠球策略研究。
