# 实测基准 —— 全部在同一台机器、同一个 Q1 模型上跑出来的

机器：Intel i9-14900K（24 核 / 32 线程）+ 2× RTX A6000 48G，Ubuntu 6.8。
模型：`q1_v2/xml/q1_v2_mc.xml`（厂商给的那份带 actuator/sensor/floor 的训练版）

> 2026-10-16 起模型换成 `q1_v3`（只差惯性参数，`_ref/q1_v2` 已删），下面的数字是当时用 v2 测的。
nq=29 nv=28 **nu=22** nbody=27 ngeom=71 nmesh=24，整机质量 16.38 kg，
timestep=1 ms，solver=NEWTON(100 it)，integrator=EULER。

单位统一为「物理子步/秒」(sim-substeps/s)，即 `nworld × nstep / wall`。

| 后端 | 设备 | 并行环境数 | 吞吐 | 备注 |
|---|---|---:|---:|---|
| `mujoco` 朴素 `mj_step` 循环 | 1 核 CPU | 1 | 53,584 | 基准线 |
| **mjbatch** | CPU 1 线程 | 4096 | 52,183 | 1.0× |
| **mjbatch** | CPU 4 线程 | 4096 | 206,667 | 3.9× |
| **mjbatch** | CPU 8 线程 | 4096 | 377,911 | 7.1×（≈笔记本上限） |
| **mjbatch** | CPU 16 线程 | 4096 | 588,437 | 11.0× |
| **mjbatch** | CPU 24 线程 | 4096 | 761,602 | 14.2× |
| **mjbatch** | CPU 32 线程 | 4096 | **812,207** | 15.2× |
| **mujoco_warp** | 1× RTX A6000 | 4096 | **2,485,667** | +82.8 s JIT 编译 |
| **mujoco_warp** | CPU (warp cpu device) | 64 | 6,988 | 比 mjbatch 慢 ~116× |
| **MuJoCo WASM** (`@mujoco/mujoco` 3.13.0) | Chrome 153, 单线程 | 1 | 28,798 | = 28.8× 实时 |

复现命令：

```bash
# mjbatch（CPU）
uv venv .venv && uv pip install --python .venv/bin/python mjbatch
.venv/bin/python mjbatch/benchmarks/scaling.py q1_v3/xml/q1_v3_mc.xml \
    --num-sims 4096 --threads 1,4,8,16,24,32

# mujoco_warp（GPU）
uv pip install "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp.git"
mjwarp-testspeed q1_v3/xml/q1_v3_mc.xml --nworld 4096 --nstep 200

# 浏览器：基准页 web_poc/ 已于 2026-10-16 删除，要复现先取回（git show 5b79856:web_poc/index.html），
# 放到 web/ 下并把其中的 ./node_modules 和 ../robots 路径对上，再用 bench/cdp.js 跑
```

## 浏览器闭环实测（当时的基准页 `web_poc/`，已删除；Chrome 153 headless，单线程，无渲染）

```
mujoco wasm boot   125 ms     nq=29 nv=28 nu=22 ngeom=44 dt=0.001
physics          28,798 steps/s = 28.8x 实时
policy            0.032 ms/次   (onnxruntime-web, wasm EP, 198k 参数 MLP 60→512→256→128→22)
sim2sim 闭环      25.43x 实时   (10 s 仿真用了 0.39 s；50 Hz 策略 + 1 kHz 物理)
```

2026-10-16 重测（基准页改为加载网页实际用的 `robots/q1/q1_web.xml`：q1_v3、位置执行器、碰撞 mesh 换成包围盒，
从站姿关键帧开始，动作按契约 `默认角 + 0.25 × action`；Chrome 155 headless）：

```
mujoco wasm boot   124 ms     nq=29 nv=28 nu=22 ngeom=47 dt=0.001
physics          38,782 steps/s = 38.8x 实时
policy            0.029 ms/次
sim2sim 闭环      36.54x 实时   (10 s 仿真用了 0.27 s)
```

即：**浏览器里跑满环仿真有 25 倍以上的实时余量**（上面两次的模型和浏览器版本都不同，不是同条件对比）。就算手机比这台机器慢 10 倍，
仍有 2.5× 实时。加上 three.js 渲染后按 60 fps 每帧 20 个物理步算，占用远低于一帧预算。

## 网页资源体积

| 文件 | 体积 |
|---|---|
| `mujoco.wasm`（官方单线程构建） | 9.8 MB |
| `q1_web.xml`（浏览器物理模型，无外部资源） | 26 KB |
| `policy.onnx`（198k 参数） | 0.79 MB |
| 原始 STL mesh 总量（**不能直接进浏览器**） | 24 MB |

`q1_v2_web.xml` 由 `q1_v2_mc.xml` 删掉全部 `<mesh>` asset 和引用 mesh 的 geom 得到
（那 27 个 geom 全是 `contype=0 conaffinity=0 density=0` 的纯视觉件）。
删完后 500 步随机力矩 rollout 的 `qpos` 差为 0.0——**但这只说明站立时等价，不说明物理等价**。
（2026-09-24 更正）被删的 mesh 里有 3 个是碰撞几何（双脚和头，`contype=1`）。
站立时机器人踩在每只脚 4 个 2 mm 的触点小球上，脚底 mesh 碰不到地，所以站立测不出来；
翻滚时就会偏离。现在训练用的 `robots/q1/q1_physics.xml` 保留这 3 个碰撞 mesh，
浏览器用的 `q1_web.xml` 把它们换成包围盒，代价用 `tools/check_mesh_substitution.py` 量。
早期 PoC 用的 `web_poc/q1_v2_web.xml` 删掉了碰撞 mesh，只用于第一次吞吐测量；2026-10-16 已删除，整个 `web_poc/` 随后也删了。
渲染要好看的话，mesh 走 three.js 的 glb（需要抽面），和物理彻底解耦。

---

## 追加：`robots/q1/q1_physics.xml`（框架实际使用的模型，2026-09-19）

`build_model.py` 产出的模型比厂商原始 MJCF 快约 25%——换了 `implicitfast` 积分器、
用 position 执行器把 PD 放进 MuJoCo（于是一个策略步可以一次 `step(nstep=20)`
而不是 20 次 Python 往返）、并且没有 mesh 要加载。

| 后端 | 设备 | 吞吐（物理子步/秒） | 对应策略步/秒（decimation 20） |
|---|---|---:|---:|
| 朴素 `mj_step` | 1 核 | 71,782 | 3,589 |
| mjbatch | 8 线程 | 494,593 | 24,730 |
| mjbatch | 32 线程 | **1,015,551** | **50,778** |

这是**纯物理的上限**。（2026-09-24 更正：此前在这里直接折算成"2 亿步 32 线程 1.1 小时、
8 线程笔记本 2.2 小时"，漏掉了观测、奖励和学习器的开销。）
实测端到端（PPO，纯 CPU learner，2048 环境，`train.py --device cpu`）：

| 配置 | 端到端策略步/秒 | 2 亿步 |
|---|---:|---:|
| 32 线程（i9-14900K 全开） | 19,400 | **2.9 小时** |
| 8 线程（`taskset -c 0-7` = 4 个 5.7 GHz 性能核 × 超线程） | 8,800 | **6.3 小时** |

8 线程这一行是笔记本的**乐观**近似：这台机器的性能核比笔记本快得多。
另外"2 亿步"本身是 locomotion 文献里的常见量级，不是我们测出来的；
实际需要多少步因任务而异，见后面动作跟踪的实测。
（原始 `q1_v2_mc.xml` 是 1.4 / 2.9 小时。）

```bash
make bench
```

## 追加：契约跨语言对拍（2026-09-19，Chrome 153 headless）

```
contract.json  hash a2bb47b7052f98d3  joints 22
golden_trace   contract=mimic_opensource_v2  steps=100  tol=1e-6
loaded         obs 116->119  50 Hz  decimation 20

max |obs_js - obs_py|       5.821e-11
max |target_js - target_py| 5.003e-13
PASS
```

最大差 5.8e-11 是 float32 的存储精度，比 1e-6 的容差低五个数量级。
`make validate` 跑 Python 侧重放，`make web-check` 跑浏览器侧。

---

## 追加：端到端闭环（2026-09-19）

第一个动作跟踪策略，`squat`，`mimic_opensource_v2` 契约，**纯 CPU 训练**：

```
2048 环境 x 200 迭代 x horizon 24 = 983 万策略步，i9-14900K，约 9 分钟
throughput 19.0k 策略步/秒（含 PPO 前反向；纯物理是 44k）

iter    reward   ep_len   fall%   joint_pos  body_pos  root_ori
   1     3.605    200.0     0.0        0.93      0.97      0.97
  10     3.248     75.2   100.0        0.92      0.77      0.72
  50     4.404    184.9    30.5        0.93      0.93      0.96
 100     4.952    195.9     9.3        0.94      0.94      0.98
 200     5.143    198.5     3.7        0.95      0.92      0.98
```

第 10 迭代摔倒率冲到 100% 是探索噪声起来了，不是发散——之后单调收敛。

导出的 ONNX 891 KB，`onnxruntime` 与 torch 一致到 **7.88e-07**，
契约元数据（joint 顺序、action_scale、policy_hz）烘在 ONNX metadata 里。

同一个策略在浏览器里（Chrome 153 headless，MuJoCo WASM + onnxruntime-web）：

```
600 策略步 = 12.0 s 仿真，用了 0.45 s 墙钟
realtime factor      26.6x
joint tracking RMSE  0.0999 rad
max tilt             1.2 deg
final base height    0.4193 m   （全程没有低于 0.22）
```

## 追加：两个模型的吞吐与碰撞代价

| 模型 | 8 线程 | 32 线程 |
|---|---:|---:|
| `q1_physics.xml`（保留碰撞 mesh） | 422,968 | 888,045 |
| `q1_web.xml`（碰撞 mesh 换包围盒） | 490,538 | 984,080 |

保真的代价是 **11% 吞吐**。包围盒近似的代价用 `tools/check_mesh_substitution.py` 量：
3 秒翻滚轨迹相对保真模型最大偏离 **0.21 rad**（删掉碰撞 mesh 是 0.13 rad，
但那是"少了体积"而不是"体积对了"——站立完全一致，所以只测站立发现不了）。

---

## 追加：mujoco_warp 后端（2026-09-19）

`q1_rl/backends/warp_backend.py`，和 mjbatch 走同一套五个动词。同一台机器、同一份环境代码
（`MotionTrackingEnv` 一行没改）、同一段固定动作序列，`tools/compare_backends.py`。

> **2026-09-24 更正。** 这一节最初的数字（reward 差 0.16%、端到端 26,631 步/秒、0.61 倍）
> 是在一个**会丢写入**的 warp 后端上测出来的：脏标记只在 `bind()` 时设一次，
> 第一次上传后就清掉了，之后没有伴随重置的写入全部丢失——下发 +0.6 rad 的膝关节指令只动了 0.012 rad，
> 随机化的摩擦从没到过设备。大批量时几乎每步都有环境重置、顺带把镜像重新上传，所以掩盖了它。
> 修复后（`tools/check_backend_writes.py` 作回归测试）重测如下，旧数字作废。
> 下面的环境也已包含同日修掉的奖励与重置问题，所以绝对 reward 值和旧表不可比。

### 行为一致性 —— 这是抽象层成立的判据

| 环境数 | 后端 | reward | 摔倒率 | joint_pos | body_pos | root_ori |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | mjbatch | 3.410 | 0.89% | 0.9232 | 0.8258 | 0.7779 |
| 1024 | warp | 3.410 | 0.90% | 0.9230 | 0.8253 | 0.7776 |
| 4096 | mjbatch | 3.412 | 0.89% | 0.9234 | 0.8265 | 0.7785 |
| 4096 | warp | 3.410 | 0.90% | 0.9231 | 0.8251 | 0.7764 |

reward 差 **0.01%（1024）/ 0.06%（4096）**，摔倒率差 0.01 个百分点。
不追求逐位相同也做不到：mjbatch 在 CPU 上跑 float64，warp 在 GPU 上跑 float32。

### 速度 —— 结论和直觉相反

| | 纯物理吞吐 | 端到端 1024 环境 | 端到端 4096 环境 |
|---|---:|---:|---:|
| mjbatch（32 线程 CPU） | 0.97M substeps/s | **42,716 策略步/s** | **43,894 策略步/s** |
| warp（1× RTX A6000） | **2.13M substeps/s** | 15,622 策略步/s | 20,596 策略步/s |

**物理快 2.2 倍，端到端反而只有 0.37–0.47 倍。**

原因是同步粒度：warp 的状态在显存里，NumPy 看不到，所以 `bind()` 返回的是主机镜像，
每个策略步都要过一次总线；而 mjbatch 的 `bind()` 是**零拷贝的活视图**，
环境的 obs/reward 直接读写仿真内存。对 4096 环境这条链路上：

- 每步传输量本身不大（约 1.3 MB 每向）
- 但每个策略步要 upload + `mjw.forward`/`step` + `wp.synchronize()` + download 一轮
- 环境的 obs/reward 是 CPU 上的 NumPy，GPU 在这段时间闲着

排查过程中修掉的两件事（都记在代码注释里）：

1. `put_data` 默认 `njmax=64`，这个机器人双脚着地就超，mujoco_warp 会**丢约束**
   并只打印一次 `nefc overflow` 继续跑错的物理。现在按模型尺寸算 njmax。
2. `_download()` 原本把 `expand()` 的模型参数也拖回主机——那是常量，每步白传约 10 MB。
   现在只回传 Data 字段；上传方向，模型参数用影子副本比较、变了才传。
3. 环境 `reset()` 里曾有一处死代码（修正量乘了 0.0）带着一次多余的 `forward()`。
   在 mjbatch 上局部 forward 几乎免费，在 warp 上却要跑满 4096 个 world 再同步一次。
4. （2026-09-24）上面说的丢写入问题。
（第 2、3 条当时报的提速数字是在丢写入的后端上量的，已从本文删去。）

### 所以怎么选

**mjbatch 仍然是默认，而且在这个环境上 warp 现在不值得切。**
warp 后端是可用且已验证行为等价的，它要真正发挥作用，得把 obs 和 reward 也搬到设备上——
那正是 mjlab / IsaacLab 的做法（它们的环境代码是写在设备数组命名空间里的）。
那会让环境代码失去"在任何机器上跑"的性质，也就是这一层存在的理由。

换句话说：**这次测量反过来加强了 CPU 优先的结论。**
对 22 自由度、接触点不多的人形，把物理搬上 GPU 换来的 2.2 倍，
被同步粒度吃掉还倒亏。

```bash
python3 tools/compare_backends.py --envs 4096 --steps 60
```

---

## 追加：SAC（FastSAC 路线）实现与调试记录（2026-09-19）

`q1_rl/algos/sac.py`，约 250 行，和 `ppo.py` 共用 `RunningNorm` 和同一条 ONNX 导出路径。
参照 <https://github.com/NPCLEI/KungFuAthleteBot> 的 FastSAC 思路，但**不引入它的栈**
（它建在 Unitree RL Mjlab 上，NVIDIA-only，正是我们绕开的东西）。

为什么值得试：PPO 每批数据用几个 epoch 就丢，成本随环境步数走；SAC 有 replay buffer，
每一步仿真被反复利用。而后端对照已经测出**仿真不是这台机器上的瓶颈**，
所以"拿仿真换梯度"是在往正确方向换。

### 调试过程中找到的三个问题，都不是调参问题

**1. 目标网络的归一化器永远不更新。**
`RunningNorm` 的统计量是 buffer 不是 parameter，而 Polyak 软更新走的是 `parameters()`——
所以目标 critic 的归一化器会永远停在 mean 0 / var 1，在线 critic 却看归一化后的输入。
两个 Q 在不同输入尺度上算，bootstrap 目标是无意义的。
改成两边都用 `net.norm`（归一化器本来就不是学出来的参数）。

**2. 动作空间尺度。** 这条走了两次弯路，值得记下来：

| 方案 | 结果 |
|---|---|
| tanh 张成整个关节行程 | 尺度到 12.6，即每步 ±3.1 rad 偏移。新初始化的策略一探索就把每个关节甩到限位，**每个回合第一步就摔**，buffer 里全是失败 |
| 统一压到 ±0.5 rad | 探索正常了，但 squat 的膝关节参考要偏离标称姿态 **0.9 rad**——策略**物理上够不到**被奖励的目标。表现是 reward 平台 + 100% 摔倒，看起来像优化失败，其实是可达性失败 |
| **按参考动作定，逐关节** | 动作范围 = 动作需要的偏移 × 1.5，下限 0.2 rad。膝关节拿到 1.35 rad，squat 里不动的手臂只有 0.2 rad——不动的关节不再乱甩 |

参照系：PPO 在同一任务上的实际工作幅度是 |a| ≈ 0.5，也就是 **±0.125 rad**。

**3. 梯度预算才是真瓶颈。**
每迭代 8 次更新时，SAC 在 125 万步之后仍然平在 reward 3.18 / 100% 摔倒，看着像坏了。
提到 48 次（并把 learner 放到 GPU）之后立刻开始学：

```
iter   reward   ep_len   fall%   alpha  entropy
  50    3.274     81.2   100.0   0.11    13.34
  80    3.349     91.8   100.0   0.08    12.43
 100    3.391    116.0    98.4   0.06    10.32
 120    3.459    159.6    66.7   0.05    10.04
```

这正是那笔交易本身：**SAC 用梯度算力买样本效率，把梯度那一侧饿着就没有用它的理由了。**
在 CPU learner 上这 48 次更新比它替掉的仿真还贵——所以
**SAC 需要 `--device cuda`，而 PPO 纯 CPU 就很好。**
默认值已经按这个设成 `updates_per_iter=48, batch=4096`。

### PPO vs SAC 重跑（2026-09-24）：3 个种子，确定性评估

```bash
make ab-algos SECONDS=420                    # 3 个种子 x 2 个算法，约 50 分钟
python3 tools/analyse_ab.py runs/ab_algos.json
```

同样 420 秒等墙钟、1024 环境、squat、learner 在 A6000。和下面那版的区别：

- **打分用确定性评估**（`q1_rl/evaluate.py`）：均值动作，固定种子的评估环境，
  每个机器人完整跑一个回合、不重置。每 60 秒训练评估一次，评估时间不计入预算。
- **3 个种子**（0、1、2）。
- 环境是修复后的：参考根部高度正确、摩擦随机化生效、`action_rate` 惩罚生效、重置不再悬空。

| | PPO | SAC |
|---|---|---|
| 最终评估摔倒率 | **0.0%**（3/3 个种子都是 0） | 16.8%（14.8–52.7%） |
| 最终关节跟踪 RMSE | **0.097 rad**（0.073–0.120） | 0.176 rad（0.147–0.186） |
| 最终每步奖励 | **3.610**（3.540–3.637） | 3.390（3.375–3.517） |
| 训练中最好的一次评估摔倒率 | 0.0% | 0.4%（0.0–0.8%） |
| 评估摔倒率 ≤ 5% 用时 | **≤ 60 s**（第一次评估就已达到） | 约 180 s |
| 420 秒内环境步数 | 11.1M | 9.7M |

（表中为三个种子的中位数，括号内为最小–最大值。）

**PPO 在所有最终指标上都更好，且三个种子一致；SAC 能摸到好的点但守不住。**
SAC 三个种子在训练中都一度把评估摔倒率压到 1% 以内，到 420 秒时却全部退化，
其中一个种子退回到 52.7%。这是跨种子一致的不稳定，不是运气。
时间分辨率受评估间隔（60 秒）限制：PPO 的"≤ 60 s"意思是第一次评估时已经达标。

局限照旧：只有 squat 一个动作；这是照 FastSAC 思路的移植，没有它的分布式 critic 等组件；
420 秒预算。**不构成对 FastSAC 的评价。**

### ~~PPO vs SAC 对照~~（已被上面的重跑取代）（420 秒等墙钟，1024 环境，squat，seed 0，learner 在 A6000）

> **2026-09-24：这一节的结论"PPO 快，SAC 高"不成立，保留仅作记录。**
> 它比的是带探索噪声的训练奖励（SAC 的 alpha 衰减到 0.05，PPO 的 log_std 没有同步收缩），
> 只有一个种子，并且当时的环境有四个问题：参考根部高了 81 mm、摩擦随机化无效、
> `action_rate` 恒为 0、重置时悬空。见上面的重跑。

```bash
make ab-algos SECONDS=420          # 跑
python3 tools/analyse_ab.py        # 从存下来的曲线重新打分，不用重训
```

| | PPO | SAC |
|---|---|---|
| 420 秒内环境步数 | 11.40M | 9.99M |
| 探索谷底 | reward 2.981 @ 3s | 3.022 @ 2s |
| 摔倒率 ≤ 20% | **49s** | 133s |
| 摔倒率 ≤ 5% | **199s** | 150s |
| 回合长度 ≥ 190 | **59s** | 133s |
| 最佳工作点 | reward 3.534，摔倒 1.7%，回合 200/200 | **reward 3.662，摔倒 0.6%，回合 200/200** @ 328s |
| 结束时 | 3.534 / 1.7% | 3.589 / 5.8%（已从最佳点退化） |

奖励正项权重之和是 3.8，所以 3.66 已经相当接近上限。

~~**PPO 快，SAC 高。**~~（不成立，见上）PPO 单调爬升，49 秒就把摔倒率压到 20% 以下；
SAC 前 97 秒几乎没有进展（摔倒率一直 100%），然后**突然**起飞，
最终摸到的工作点在三个指标上全面优于 PPO 曾经达到的任何一点。
代价是 SAC 后期不稳：387 秒时退回到 29.5% 摔倒率。

**必须声明的局限**：单一种子、单一动作（squat 很简单，PPO 能打满 200/200）、
420 秒预算、而且这是照 FastSAC 思路的移植而非它的实现（没有分布式 critic）。
**这不构成对 FastSAC 的评价**，只说明在我们这套环境和这个任务上两者的形状。

### 顺带修掉的两个测量 bug（都会给出错误结论）

1. **PPO 报的 reward 含 bootstrap。** rollout 里把超时的 bootstrap 值加进 `reward`
   再存 buffer，而统计取的是 buffer 均值——于是 PPO 报 5.251，而奖励正项上限只有 3.8。
   第一版对照据此判"PPO 大胜"，是错的。现在两边都报原始环境奖励。
2. **没有回合结束时 `fall_rate` 报 0。** `falls / max(episodes, 1)` 在 episodes=0 时给 0，
   于是"多久把摔倒率压到 5%"在第 1 次迭代就满足。现在报 NaN，
   `analyse_ab.py` 也从"探索造成的最差点"之后才开始计时。

两个 bug 的共同点：**它们都让结果更好看，而且都不会报错。**
