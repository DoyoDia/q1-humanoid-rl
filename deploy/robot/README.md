# 上机部署

把训练好的动作策略注册成机器人的自定义动作（`MIMIC_OPENSOURCE_V2`），再用局域网网页播放。
本目录的脚本都**在机器人上运行**：先把文件 `scp` 到机器人的 `/tmp/q1rl/`。

> ⚠️ 自定义动作会直接驱动全身关节。第一次上机请把机器人吊起或有人扶着，遥控器拿在手里（遥控器是唯一的急停）。
> 重启运控会掉力矩，只在机器人躺着或吊着时进行。

## 1. 打包并注册动作

```bash
# 在开发机上
python tools/check_vendor_policies.py                  # 必过：观测布局与机器人一致
python tools/make_deploy_package.py runs/<run> --name dance   # -> runs/<run>/deploy/q1rl_dance.{onnx,npz,csv}

# 在机器人上（要保留的动作每次都列上；每次安装都从原始配置开始）
bash /tmp/q1rl/install.sh /tmp/q1rl Q1RL_DANCE="自训舞蹈"
yamo em stop-app mc && yamo em start-app mc            # 运控只在启动时读取动作列表
bash /tmp/q1rl/trigger.sh Q1RL_DANCE                   # 机器人需在走跑模式且 RUNNING
bash /tmp/q1rl/rollback.sh                             # 撤销（同样需要重启运控）
```

`install.sh` 做两件事：
- 在 `motion_list.yaml` 里注册动作。
- 在 `rl/mimic_opensource_v2_config.yaml` 的 `CustomConfigs` 里打开该动作的重力保护（gravity guard）。运控从重力保护的 csv 读取观测里的参考重力（obs[44:47]），见 `robots/q1/contract/README.md`。

包里有 `q1rl_<name>.detector.onnx`（由 `tools/train_lift_detector.py` 训练）时，还会同时打开离地检测：机器人被抱离地面时，运控会切换到阻尼状态。

### SDK v4 固件

新固件把 `rl/*.yaml` 放进了不可编辑的配置包，`CustomConfigs` 改不了，所以自定义动作**没有重力保护，也没有离地检测**，obs[44:47] 也不会被写入。

`install.sh` 遇到这种固件默认拒绝安装。设置 `Q1RL_NO_GUARD=1` 后，它只修改 `motion_list.yaml`。只有确认不依赖这 3 维观测的策略才应该这样装，比如本仓库的 SuperMimic 119 维策略：在仿真里把这 3 维固定为 0 或直立重力，跟踪误差基本不变。

运控只在启动时加载模型，改完要整机重启。可以用 SDK 的 `MigrateSystemState`：

```bash
Q1RL_NO_GUARD=1 bash /tmp/q1rl/install.sh /tmp/q1rl "Q1RL_NLDX_SM=nldx 舞（SuperMimic）"
source /tmp/q1rl/env.sh && python3 -c '
import rclpy; from aimdk_msgs.srv import MigrateSystemState as M; rclpy.init(); n = rclpy.create_node("rb")
c = n.create_client(M, "/aimdk_5Fmsgs/srv/MigrateSystemState"); c.wait_for_service(10)
r = M.Request(); r.state = "Reboot"; f = c.call_async(r); rclpy.spin_until_future_complete(n, f, timeout_sec=10); print(f.result())'
```

### 推理耗时：上机前先蒸馏

运控按 20 ms 一拍推理，跑在一颗 A53 小核上，预算很紧。实测结果：

| 策略 | ONNX 大小 | 每遍超时步数 | `joint_command`（5 ms）漏拍 |
|---|---|---|---|
| SuperMimic 原版 2048-1024-512-256 | 12 MB | 约 70 / 306（中位 22 ms） | 上百次 |
| 蒸馏版 512-256-128（`tools/distill_policy.py`） | 0.9 MB | **0** | **0** |

所以大网络要先蒸馏到和厂商模型同量级（约 1 MB）再装。

## 2. 局域网网页（`web/`）

在手机或电脑上列出机器人的全部动作并播放，可以给动作配乐。

```bash
# 把 web/* 传到机器人 /tmp/q1rl/web/，配乐文件传到 /tmp/q1rl/media/，然后在机器人上：
bash /tmp/q1rl/web/install_web.sh /tmp/q1rl/web /tmp/q1rl/media
bash ~/q1rl_web/uninstall_web.sh                       # 卸载
```

访问地址是 `http://<机器人IP>:8090/#key=<口令>`，口令在机器人的 `~/q1rl_web/key`，安装时随机生成。

**安装位置与自启**
- 有 `/home/user` 分区时，网页装在 `/home/user/q1rl_web`（OTA 不会清掉这个分区），`~/q1rl_web` 是指向它的链接。
- 有免密 sudo 时，开机自启用 systemd（`q1rl-web.service`）；否则用 crontab 的 `@reboot`。

**功能**
- 动作分成官方 / 自定义、全身 / 上肢两类：
  - 全身动作在走跑模式下用 `SetMcMotion` 播放。
  - 上肢动作（`ANIMATION` 和示教动作）在全身控制模式下用 `SetMcPresetMotion` 播放。
- 页面可以在这两种站立模式之间切换，服务端每次播放前都会检查当前模式。
- 从躺姿开始的动作只列出，不允许播放。
- 刻意没有做停止和阻尼按钮，急停用遥控器。

**配乐**
- `web/media.json` 把动作绑定到音乐，例如 `{"Q1RL_BT": {"wav": "bt.wav", "mp3": "bt.mp3", "offset_s": 0}}`。
- 动作开始时，机器人喇叭播放 wav（24 kHz、16 bit、单声道）。勾选“用手机播放”时改由手机播放 mp3。
- `offset_s` 用来调音画同步：为正时音乐晚于动作开始。
- 动作提前结束时，音乐自动停止。

**与导航栈共存**
- 如果有节点发布 `/cmd_vel_bridge/enabled`（`std_msgs/Bool`，latched）且值为 true，说明导航正在用速度指令控制走路。这时页面不播放动作，也不切换模式。
- 运控模式刚变化过的 3 秒内，页面也不操作。
- 设置环境变量 `Q1RL_AFTER_MOTION_NOTE` 后，每个动作结束时页面会显示这段提示，比如提醒重启里程计。
- 服务端不订阅 50 Hz 的话题，只在页面打开时每秒查询一次 `GetMcAction`。

## 3. 登录与开发者模式

- SDK v4（2026-10）起，新系统的机器人只接受 SSH 密钥登录，私钥向售后索取（`<SN>_soc0/id_ed25519`）。
- 开发者模式在手机 APP 里打开：设置 → 开发者功能 → 开发者模式。保存后重新上下电。
