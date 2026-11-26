# Q1 部署契约

> 一份策略要能在**训练环境、浏览器、机器人**三个地方跑出同样的行为，
> 靠的不是"大家都照着文档写"，是这个目录。
>
> 数据来源：Q1 机器人运控配置目录 `cfg/robot/q1_t2d0/`（2026-09 固件），只读读回。

---

## 为什么要有这个目录

sim2real / sim2sim 出问题，九成不是策略不行，是**观测拼错了**：
顺序差一位、缩放少乘一个数、历史帧对齐差一拍、默认角用了另一套。
这类错误不会报错，只会让机器人"看起来学废了"。

Q1 的情况更糟一点，因为同一份数据在四个地方有四个版本，而且互相矛盾：

| 来源 | 关节限位 | 力矩 | 关节顺序 |
|---|---|---|---|
| `q1_v3.urdf`（官网 2026-09，v2 只差惯性参数） | ✅ 权威 | 36/22 | 无序（XML 顺序） |
| `q1_v3_mc.xml` | ✅ 与 URDF 逐位一致 | actuator ctrlrange 36/22 | MJCF body 树顺序 |
| 机器人 `robot_model.yaml` | ❌ 旧机型残留 | 33/14 | 另一种顺序 |
| SDK `接口说明.md` | ❌ 抄自上一行 | 33/14 | 又一种顺序 |

所以：**这个目录是唯一真源，其他四份都只是它的输入。**

## 文件

| 文件 | 内容 | 谁生成 |
|---|---|---|
| `joints.yaml` | 22 个关节的顺序、限位、力矩/速度上限、armature、下发约束 | `gen_joints.py`，**不要手改** |
| `obs_spec.json` | 两套观测布局的逐字段切片 | 手写，抄自机器人配置 |
| `gains.yaml` | kps / kds / default_dof_pos / action_scale | 手写，抄自机器人配置 |
| `rates.yaml` | 50 Hz 策略、1 kHz 物理、decimation 20、各种超时 | 手写，抄自机器人配置 |
| `gen_joints.py` | 从 URDF + MJCF + robot_model 推导 `joints.yaml` | — |
| `validate.py` | 五项一致性检查，进 CI | — |

```bash
python3 gen_joints.py                       # 重新生成 joints.yaml
python3 validate.py                         # 自洽性检查（CI 跑这个）
python3 validate.py --onnx path/to/p.onnx   # 顺带校验一个导出的策略
```

---

## 1. 关节顺序（锁死，永不排序）

这 22 个名字的**顺序本身就是契约**。抄自机器人
`rl/mimic_opensource_v2_config.yaml` 的 `seq`（与 `action_seq` 相同）。

```
 0 left_hip_pitch        1 left_hip_roll         2 left_hip_yaw
 3 left_knee             4 left_ankle_pitch      5 left_ankle_roll
 6 right_hip_pitch       7 right_hip_roll        8 right_hip_yaw
 9 right_knee           10 right_ankle_pitch    11 right_ankle_roll
12 waist_roll           13 waist_yaw
14 left_shoulder_pitch  15 left_shoulder_roll   16 left_shoulder_yaw   17 left_elbow
18 right_shoulder_pitch 19 right_shoulder_roll  20 right_shoulder_yaw  21 right_elbow
```

注意 **12 是 waist_roll、13 是 waist_yaw**（不是反过来），
而 URDF 里 `config/joint_names_*.yaml` 给的顺序是 `waist_roll, waist_yaw` 但整体排列不同——
不要拿那份当依据。

⚠️ URDF 里还有 `left_wrist_yaw_joint` / `right_wrist_yaw_joint` 两个关节，
**MJCF 里没有，契约里也没有**。动作空间是 22，不是 24。

## 2. 限位：按 URDF，这件事有硬证据

机器人上的 `robot_model.yaml`（以及抄它的 SDK 文档）腿部限位是从 QD1/T1 四足产品线
复制来没改的：三组值 `(-0.4887, 0.4887)` / `(0.0, 4.119)` / `(-2.618, 2.618)` 按 3 循环重复，
同目录下就躺着 `qd1_t0d0` / `qd1_t1d5` / `qd1_t2d0` 三套四足配置。

**决定性证据**：厂商自己行走策略的标称站姿（`gains.yaml` 的 `walk.default_dof_pos`）里，

```
膝       = 0.6982   → robot_model 说限位是 (-0.4887, 0.4887)   ✗ 越界
踝 pitch = -0.3582  → robot_model 说限位是 (0.0, 4.119)        ✗ 越界
```

而机器人是**站着走路**的。所以那份限位不可能被当成位置钳位执行。

`validate.py` 把这件事变成一条常驻断言：
**`default_dof_pos` 必须落在 `joints.yaml` 的限位内**。
这条断言现在两套 gains 都过——因为 `joints.yaml` 用的是 URDF。
哪天有人把限位改回 `robot_model.yaml` 那套，CI 会立刻红。

### 但力矩和速度上限取两者较小值

`robot_model.yaml` 有一样 URDF 完全没有的东西：`max_velocity`。逐关节取 min：

| 关节组 | 力矩 (N·m) | 速度 (rad/s) | 谁更严 |
|---|---:|---:|---|
| 髋 ×3 / 膝 / 腰 ×2 | 33 | 20.94 | robot_model |
| 踝 pitch, roll | 22 | 20.94 | 力矩 URDF，速度 robot_model |
| 肩 pitch, yaw / 肘 | 22 | 20.94 | 力矩 URDF，速度 robot_model |
| 肩 roll | 14 | 27.23 | 都是 robot_model |

## 3. 观测：两套契约，推荐用哪套

### `MIMIC_OPENSOURCE_V2`（推荐）

119 维，**全部有效，没有补零**。这是运行时实际拼出来的布局，**不是**机器人配置文件开头那段注释
（那段说"116 维有效、尾部补零、gvec 在 44、base_ω 在 47"，已经过时）：

```
[  0: 44)  motion_ref_pos(22) + motion_ref_vel(22)   参考轨迹，当前帧
[ 44: 47)  ref_gravity                                参考动作骨盆系下的重力方向（见下）
[ 47: 50)  gvec                                       机器人基座系下的重力方向（IMU）
[ 50: 53)  base_ang_vel                               IMU 角速度
[ 53: 75)  joint_pos_err = q - default_dof_pos
[ 75: 97)  joint_vel                                  原始值，不乘 0.05（配置里写了，运行时不用）
[ 97:119)  prev_action                                上一步网络原始输出
```

**ref_gravity 从哪来**：运行时从这个动作的重力保护 csv（`res/mimic_gravity_guard/<name>.csv`，
每帧一行 `frame_id,gx,gy,gz`）按当前帧读，**而且只有在 `mimic_opensource_v2_config.yaml` 的
`CustomConfigs` 里给这个动作开了 `gravity_guard` 才会读**，否则这 3 维保持旧值。
我们训练时用参考 npz 骨盆姿态算出来的重力，部署时用同一个函数从同一个 npz 生成 csv
（`tools/make_deploy_package.py`），`deploy/robot/install.sh` 负责开重力保护。

**怎么确认的**（2026-10-05，一个按旧布局训练的策略在真机上一触发就剧烈抽搐摔倒之后）：
- 观测按 `motion_pos, motion_vel, ref_gravity, gravity, imu_omega, dof_pos, dof_vel, actions`
  依次拼接，共 119 维、没有补零；`ref_gravity` 来自该动作的重力保护数据；只有 `ang_vel` 一个缩放系数。
- 厂商 ONNX 里内置的观测归一化：44–49 是两组重力样的三元组，50–52 均值约 0，
  116–118 是真实数据；关节速度那段标准差约 1.5 rad/s（乘过 0.05 不可能这么大）。
- 厂商自己的策略在我们的仿真里：旧布局 FLY 4/4 摔、起身起不来；新布局 FLY 关节误差 0.08 rad、
  起身正常。这个对比做成了部署前必过的检查：**`python tools/check_vendor_policies.py`**。

`action_scale = 0.25`，`target_q = default_dof_pos + 0.25 * action`。

**选它的理由**：布局基本是学术界标准的那套（投影重力 + 基座角速度），
而且机器人的 `mc_main` 直接支持——往 `motion_list.yaml` 加一段就能加载，
厂商的 1 kHz PD 下发、防摔 guard、状态机、退出阻尼全都白送。

**代价**：观测前 44 维是参考轨迹，所以它天生是**动作跟踪**类策略，不是纯速度指令行走。

参考轨迹的 npz 格式已经从机器人上的文件解出来了（解 zip 里每个 `.npy` 的头），
三份样本（`fly_0422` T=400、`q1_sit` T=558、`q1_getUp_v3.3` T=309）结构完全一致：

```
fps             (1,)        float64
joint_pos       (T, 22)     float32    joints.yaml 顺序
joint_vel       (T, 22)     float32    joints.yaml 顺序
body_pos_w      (T, 26, 3)  float32    世界系
body_quat_w     (T, 26, 4)  float32    世界系
body_lin_vel_w  (T, 26, 3)  float32    世界系
body_ang_vel_w  (T, 26, 3)  float32    世界系
```

这就是 **mjlab / IsaacLab 的 motion npz 约定**。
26 = MJCF 的 `nbody=27` 去掉 `world`，**顺序就是 MJCF body 树顺序去掉 world**（pelvis 在前）。
2026-10-04 用机器人上全部 16 个厂商 npz 验证过：拿 body 0 的位姿加 `joint_pos` 做正运动学，
较新的 8 个文件和 `body_pos_w` 吻合到 1e-7 m；较早的 8 个只在脚踝差 7 mm，
应该是当时的 URDF 小腿长度不同。四元数是 (w, x, y, z)。

**参考帧时序**：运控每拍依次读观测、推理、写动作，然后才把帧号加一，
所以第 k 拍的 `motion_input` 是第 k 帧，没有往前看一帧。训练环境的约定与此相同。

### `Walk`（不推荐，但它是最好的 locomotion 参考）

71 维 × `frame_stack: 5`，用 **roll/pitch 两维欧拉角**（不是三维投影重力）
外加 **CPG 相位**，和学术界常见布局不一样，训练侧复现麻烦。

**`prev_action` 是 13 维，这是对的**（2026-10-04 读 ONNX 确认）。配置的 `obs_index`
写 `action: 54, cpg: 67`，中间只有 13 维，而 `action_nums: 22`，看着矛盾，其实不矛盾：
`q1_walkRun_upperPosCtrl` v2.1/v3.1/v3.2 的 actor 只输出 13 维
（12 个腿关节加 1 个腰关节，按关节顺序是 waist_roll），
图里再拼 9 个常数 0 凑成 22 维。剩下 waist_yaw 和 8 个手臂关节走位置控制（`upperPosCtrl`）。
观测里放的是策略自己那 13 维输出。ONNX 有两个输入：
`observations` 71 维、`observation_history` 355 维（5×71），
图里自带一个估计器（速度 3、隐变量 16、地形 16）。
`obs_spec.json` 用 `policy_action_dim: 13` 记录这一点，`validate.py` 也已通过。

它的价值在于**参数可以直接抄**：kps/kds、指令范围（vx ∈ [-0.3, 0.4] m/s、
vy ∈ [-0.4, 0.4]、yaw ±0.785 rad/s）、步态周期 0.25–0.7 s、步长 0.6。
这些是甲方在真机上调出来的，省掉我们一轮调参。

## 4. 频率：不是可选项

```
策略        50 Hz     (dt = 0.02)
物理        1000 Hz   (timestep = 0.001, decimation = 20)
关节下发    1000 Hz   (rt_joint_pub, SCHED_RR:80, 绑 CPU 5)
关节回传    500 Hz
```

机器人上 13 份带 `dt:` 的 RL 配置（`q1_t0d0`）**全部**是 `dt: 0.02`。其中 5 份的注释写着 `# 100Hz`，**是错的**（另 8 份写 `# 50Hz`）；
另有两份写对了并且写全了：

> `dt: 0.02  # 50Hz policy, same as sim2sim decimation 20 * 0.001s.`

`validate.py` 断言 `timestep × decimation == 1 / policy_hz`，并核对 MJCF 的 `<option timestep>`。

### 10 ms 超时决定了策略跑在哪

```yaml
data_timeout_guard: { imu_timeout_ms: 10, joint_timeout_ms: 10 }
cmd_guard: { expiration_time_ms: 200 }
```

IMU 和关节数据 10 ms 不到就判超时。跨 WiFi 的 DDS 做不到这个。
**策略必须是机器人本机的进程。**

## 5. 契约的五个消费者

```
                    ┌─ envs/      训练时构造 obs
                    ├─ export/    导出 ONNX 时写 metadata
  contract/  ──────►├─ web/       浏览器里的 JS obs 构造器
                    ├─ deploy/    机器人上的 obs 构造器（路径 C）
                    └─ mc_main    厂商自己的运行时（路径 A，我们控制不了）
```

前四个我们控制，`validate.py` 管住它们。第五个是厂商的——
所以还需要一条**黄金轨迹**（100 步 obs→action 对照表），
用来把我们的 obs 构造器和 `mc_main` 的对拍。一旦对上，路径 A 就是安全的。

> 黄金轨迹还没生成——需要先有一个符合契约的策略。

## 6. 下发侧的四条硬约束

写在 `joints.yaml` 的 `command:` 段，来自 SDK `接口说明.md`（v2–v4 原文一致，2026-10-16 核对 v4）：

| 约束 | 后果 |
|---|---|
| 单帧位置增量 ≤ **0.2 rad** | 超了有减速器崩齿 / 母线过流风险。**训练环境里就要加这条速率限制**，否则真机行为和仿真不一致 |
| 下发 **500–1000 Hz** | 低于 100 Hz 触发 watchdog 和电机啸叫。50 Hz 的策略必须配一个保持转发器 |
| `header.stamp` 机载时钟、严格单调 | 乱序帧**静默丢弃**，不报错 |
| 退出前发一帧 `stiffness=0, damping=5` | 否则不是平稳进入阻尼态，是直接失控 |

## 7. 改这个目录的规矩

1. `joints.yaml` 只能由 `gen_joints.py` 生成。要改限位，改的是 URDF 或生成脚本里的
   `ROBOT_MODEL_CAPS`，不是 yaml。
2. `ORDER` 一旦有策略导出过就**不能再动**——改了它，所有历史 ONNX 全部作废。
3. 任何改动之后跑 `validate.py`，退出码必须是 0。
4. 从机器人上读回新数据时，在提交信息里记下来源和日期。
   这些配置会随 OTA 变，没有日期的数字三个月后没人敢用。
