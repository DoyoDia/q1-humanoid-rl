# Retargeting：人体/其他机器人动作 → Q1 参考动作

两套流程，输出同一种 motion npz（契约关节顺序、50 Hz、落地），训练和真机都直接读：

- **robot_retargeter**（[4o4lab/robot_retargeter](https://github.com/4o4lab/robot_retargeter)）：源是另一台机器人（G1）的动作，本目录根下的文件。
- **GMR**（[4o4lab/GMR](https://github.com/4o4lab/GMR)）：源是人体动捕（LAFAN1 BVH），`gmr/` 子目录。

两者都只在本仓库放 Q1 专用的部分，上游仓库克隆到 `_ref/`（不进 git）。

```bash
git clone --depth 1 https://github.com/4o4lab/robot_retargeter _ref/robot_retargeter
pip install mink daqp scipy tqdm            # 另需 mujoco、numpy
python robots/q1/build_model.py --keep-meshes --out robots/q1/q1_visual.xml

# G1 的 LAFAN1 动作（retargeter 自带 dataset/lafan1_g1/）→ runs/motions/<name>.npz
PY=python retarget/run.sh walk1_subject1 dance1_subject2

# 看一眼再训
python tools/render_motion.py runs/motions/walk1_subject1.npz -o walk.png --end 12
python train.py --motion npz --npz runs/motions/walk1_subject1.npz
```

**源是 T800**（bvh_tool / Isaac 格式的 motion npz：`joint_pos` 按 Isaac 关节顺序、`body_pos_w`/`body_quat_w` 取 0 号 body 为 LINK_BASE、四元数 wxyz）：
retargeter 自带 `config/robot/t800.yaml` 和模型，`t800_to_csv.py` 把 npz 转成它回放用的 CSV，并用正运动学和 npz 里的 body 位姿对一遍
（除了虚拟脚底点 LINK_FOOT_* 两个模型差 64.5 mm 以外全部 0 误差）。

```bash
SOURCE=t800 SOURCE_FPS=50 START=0.12 PY=python retarget/run.sh path/to/clip_t800.npz   # -> runs/motions/clip_t800.npz
```

`START=0.12`：IK 从默认姿态起步，头几帧没收敛（nldx 第 0→1 帧左髋 yaw 跳 0.68 rad，折合 34 rad/s），跳过 6 帧后最大关节速度 7.9 rad/s。

| 文件 | 作用 |
|---|---|
| `t800_to_csv.py` | T800 motion npz → retargeter 的机器人动作 CSV（基座位置、xyzw 四元数、按模型 qpos 顺序的关节） |
| `make_model.py` | 生成 retargeter 用的 Q1 模型（视觉模型 + 7 个标记点 body），并安装 `q1.yaml` |
| `q1.yaml` | Q1 的 retargeter 配置：连杆定义、IK 权重、关键帧坐标映射 |
| `to_motion.py` | retargeter 输出的 CSV → 契约顺序、50 Hz、落地的 motion npz（同时打印超限/穿地/速度统计） |
| `run.sh` | 上面三步串起来 |
| `balance.py` | `to_motion.py --balance`：把重心挪到着地的脚上方（下面单独说） |

## 配置里两处和 G1 不同，都有实测依据

- **髋 roll 坐标系偏置 10°**：G1 的 `hip_roll_link` 零位时俯仰 -10°，Q1 没有。不补的话腿部姿态目标会让大腿多倾 10°。
- **手部姿态权重 0**（G1 是 1）：Q1 没有腕关节，对不上 G1 手掌朝向。walk1 上手腕误差 14 mm（权重 0）对 17 mm（权重 1）。

`key_frame_config` 是从 G1 配置**整块照抄**的——它按关键点分别定义（手臂和腿不一样），抄错一项手臂就会扭到限位上
（第一版就是这样：肩 roll 96% 的帧顶在限位，肘位置误差 5 cm；照抄后 0%、6 mm）。

## 已知限制

- LAFAN1 片段开头有 T 字校准姿势，训练时用 `to_motion.py --start` 跳过。
- 舞蹈类片段的关节速度会贴近电机限速（dance1 最高 21.0 rad/s，限速 20.9）。
- 穿地：约 1–2% 的帧有超过 5 mm 的穿地，最深 11–14 mm。

## 重心修正（`--balance`）

人跳舞的平衡换到比例不同、脚小得多的身体上就不成立了（Q1 脚掌 14.5 × 4.6 cm）。dance1 原样重定向后，
单脚站时重心在支撑脚外面（中位 6.4 cm），策略一抬脚就倒，于是学成两脚一直不离地，空中的点地、勾脚全丢了。

```bash
python retarget/to_motion.py OUT.csv --fps 30 --start 7 --end 30 --balance -o runs/motions/dance1_s2_7_30_bal.npz
```

1. **着地判断**：不看绝对高度（dance1 整段悬空 1.2–1.8 cm），哪只脚比另一只低于 1 cm 以内就算着地；
   基本放平（倾斜 <10°）的脚按整个脚掌支撑，踮脚/翘脚只算贴地的那条边。
2. **重心轨迹**：整段一个二次规划，让 ZMP（小车-桌子模型）落在支撑区内，同时尽量贴近原轨迹、并且本身平滑
   （只约束"修正量"平滑不够：原轨迹有抖动，规划会用加速度去抵消抖动，重心照样在脚外）。
3. **逐帧 IK**（mink，多进程）：着地的脚是硬约束，腾空的脚是软约束可以跟着骨盆挪一点，手臂和腰保持角度，
   只动骨盆和腿。腿够不到规划值的帧，把重心限制在够得到的范围里重新规划，迭代几轮。
4. **去跳变**：踝关节顶限位时相邻帧可能落到不同的 IK 解，对修正量做时间滤波。

dance1（7–30 s）：ZMP 超出脚掌 >1 cm 的帧 58% → 24%，单脚支撑时重心在脚外 6.4 → 2.1 cm（中位）；
骨盆平均挪 4.8 cm，着地脚 99% 的帧动 <12 mm，关节速度最高 17.6 rad/s。剩下的主要卡在踝关节限位
（俯仰 −45°～+25°、侧倾 ±20°）。需要 `mink`、`clarabel`（`pip install mink clarabel`）。

## GMR 版本（`gmr/`）

```bash
git clone --depth 1 https://github.com/4o4lab/GMR _ref/GMR
pip install rich loop-rate-limiters natsort psutil opencv-python-headless   # GMR 的 import 依赖
# LAFAN1 BVH（Ubisoft，CC BY-NC-ND 4.0，非商用）解压到 _ref/lafan1/
PY=python retarget/gmr/run.sh walk1_subject1 dance1_subject2     # -> runs/motions/gmr_<name>.npz
```

| 文件 | 作用 |
|---|---|
| `make_config.py` | 从 GMR 的 `bvh_lafan1_to_g1.json` 推导 Q1 配置：缩放 = G1 系数 × Q1/G1 实测肢长比（腿 0.585、臂 0.750），姿态偏移 = G1 偏移 × 零位坐标系差（只有髋 yaw 差 10°），手臂姿态权重下调 |
| `bvh_lafan1_to_q1.json` | 生成的配置（提交在库里，改了 `make_config.py` 要重新生成） |
| `bvh_to_q1.py` | 无窗口驱动 GMR，运行时把 Q1 注册进 GMR 的表，不改上游代码；输出和 robot_retargeter 同格式的 CSV |

**手臂姿态权重为什么要降**：Q1 肩 yaw 只有 ±90°（G1 ±150°）、没有腕关节。照搬 G1 的权重（上臂 100、手 10），
IK 一直在追 Q1 做不到的扭转，在两组手臂解之间来回跳：dance1 有 590 帧某个关节一步（20 ms）跳 >0.3 rad、2.5% 的帧超电机限速。
手 →≈0、上臂/前臂 →1 之后：180 帧、0.1%；walk1 不受影响。GMR 自带的 `use_velocity_limit` 帮不上：它写死了 G1 的电机名，
而且只限制单次 IK 迭代，不限制帧间变化。

## 两套流程对比（`tools/motion_quality.py`）

| 片段 | 跳变帧 | 超限速 | 穿地 >5 mm | 最深穿地 | 悬空中位数 |
|---|---|---|---|---|---|
| walk1 · robot_retargeter（源：G1） | 2 | 0.0% | 0.7% | 10 mm | 16 mm |
| walk1 · GMR（源：人体 BVH） | 4 | 0.0% | 1.2% | 18 mm | 20 mm |
| dance1 · robot_retargeter | 57 | 0.0% | 1.1% | 13 mm | 12 mm |
| dance1 · GMR | 180 | 0.1% | 2.2% | 15 mm | 24 mm |

- 走路两者都能用，GMR 的腿更直、更像人（它直接从人体动捕来，没经过 G1 那一层）。
- 舞蹈 robot_retargeter 更干净：源是已经 retarget 到 G1 的动作，G1 那一步已经把人体做不到机器人上的部分消化掉了。
- 两者都有约 2 cm 的整体悬空：落地按最低点的第 5 百分位对齐，而重定向出来的脚高本身有 ±1.5 cm 起伏。
  训练时环境复位会把机器人放回地面，影响不大，但想更贴地可以调 `from_root_trajectory(ground_percentile=...)`。

## 从视频（PromptHMR → SMPL-X → GMR）

第 1 步用任意能输出 SMPL-X 序列的视频动捕流程，我们用的是 [PromptHMR](https://github.com/yufu-wang/PromptHMR)
（环境：torch 2.4.0+cu121 / xformers 0.0.27.post2 / detectron2 源码编译 / sam2 源码 `--no-deps`）。下面的路径是我们本地的布局，按需替换。

```bash
# 1. 视频 -> SMPL-X（固定机位用 --static-camera；4.5 分钟 24 fps 约 20 分钟，单卡）
cd _ref/video2robot && PYTHONPATH=$PWD python scripts/extract_pose.py --project data/bt --static-camera
# 2. SMPL-X -> Q1 CSV（GMR，配置 gmr/smplx_to_q1.json 由 make_config.py --source smplx 生成）
python retarget/gmr/smplx_to_q1.py _ref/video2robot/data/bt/smplx.npz -o runs/bt/bt_q1.csv \
    --body-models _ref/video2robot/third_party/PromptHMR/data/body_models
# 3. CSV -> 参考动作：落脚放平 + 重心修正 + 按 2 s 滑窗贴地
python retarget/to_motion.py runs/bt/bt_q1.csv --fps 24 --plant-feet --flight 207-248 --balance -o runs/motions/bt_v3.npz
```

PromptHMR 本地要补的东西（都在 `_ref` 里，不进库）：

- 权重：Google Drive 常超配额，用 ModelScope 镜像 `jijibaojiji/prompt_hmr`；所有 pickle 先过 `tools/scan_pickles.py`。
- `data/body_models/` 的三个补充文件：`J_regressor_h36m.npy` 来自补充文件夹；`smplx2smpl.pkl`、`smplx2smpl_joints.npy`
  直接从 `phmr/checkpoint.ckpt` 的 buffer `smplx2smpl`（6890×10475，与 GVHMR 的 `smplx2smpl_sparse.pt` 逐位相同）和
  `smplx.smpl_joints`（33×10475）取出。
- `pipeline/camera/__init__.py`：DROID-SLAM 改成延迟导入（固定机位用不到，且要编 CUDA 扩展）。
- `pipeline/pipeline.py`：每个阶段结束存 `results_partial.pkl`，出错后重跑从断点继续（原版只在最后存一次，任何一步出错检测跟踪全部重来）。
- sam2 官方版缺 `load_video_frames_from_np`，照 PromptHMR 的调用补上。

视频源的两个特点，`to_motion.py` 已经处理：
- **高度漂移**：同一个人静止站着，开头悬空 31 mm，两分钟后 12 mm（单目深度随时间漂）。整段统一平移会被个别穿地帧顶高；
  `--ground-window 2.0` 在 2 s 滑窗里取最低点第 5 百分位逐帧贴地。广播体操：悬空中位数 23 → 9.5 mm，最深穿地 42 → 7 mm，跳跃的腾空保留。
- **脚掌不平**（`--plant-feet`，`balance.plant_feet`）：估计出来的着地脚倾斜中位 13°（静止站立也有 7–10°），两只脚高差 1–3 cm，
  只有 13% 的帧双脚着地；广播体操脚掌几乎一直平踩。按“离地 < 8 cm 且脚踝速度 < 0.2 m/s”判着地（站立的脚 < 0.1 m/s，
  抬腿 0.3–0.7；高度门槛放宽是因为双脚站立时有一只脚被估在 5.5 cm 高），每段着地取脚的水平位置、朝向中位数，放到地面、
  脚掌水平，首尾 6 帧和源姿态过渡（不过渡膝关节一帧跳十几度），IK 让腿去够。结果：着地脚倾角 11.5° → 0°，双脚着地 13% → 65%
  （其余是提膝、踢腿、侧点地、跳跃），再做重心修正后 ZMP 超出脚底 >1 cm 的帧 8%。
  **始终一只脚平贴地面**：两脚都没判为着地的空档里，让较低的那只着地；过渡只在另一只脚踩实时进行。唯一例外是
  `--flight` 给出的时段里两脚同时离地 1.2–7 cm 的帧（开合跳；提膝跳是一只脚 10–19 cm、另一只在地上）。
  广播体操 `--flight 207-248`：除开合跳腾空的 5% 帧外，每帧都有一只脚离地 < 5 mm 且倾角 < 2°（之前 31% 的帧没有）。
  落脚后的高度就以它的地面估计为准，不再按百分位贴地（那样会被摆动脚踩低的帧把整段顶高 1–3 cm）。
- **快动作**：广播体操里手臂常在 2–3 帧内甩下 0.3–0.6 rad（15–25 rad/s），是真实动作不是 IK 翻转，保留。

训练时环境的跌倒判定是相对参考的（倾角偏离参考 60°、骨盆低于参考 0.15 m），弯腰摸脚尖（骨盆前倾 92°）、全蹲（骨盆 0.22 m）不会被误判。
