# Q1 训练框架 —— 常用入口
# 依赖：python3 + mujoco numpy pyyaml onnx（可选 mjbatch）；node 仅用于浏览器检查
PY ?= python3
PORT ?= 8199
# make view serves the LAN (phones, other machines); HOST=127.0.0.1 keeps it on this machine
HOST ?= 0.0.0.0

.PHONY: help contract validate web-check deploy-check bench train demo view ab-algos ab-backends clean
ALGO ?= ppo
RUN ?= squat_$(ALGO)
MOTION ?= squat
ITERS ?= 200
ENVS ?= 2048
SECONDS ?= 360

help:
	@echo "make contract    重新生成 joints.yaml / contract.json / golden_trace.json 和物理模型"
	@echo "make validate    契约自洽性检查（CI 跑这个，退出码即结论）"
	@echo "make deploy-check 上机前必过：厂商自己的策略要能用我们的观测构造器跑稳"
	@echo "make web-check   在浏览器里跑 JS/Python 观测构造器对拍（需要 chrome + node）"
	@echo "make bench       mjbatch 吞吐基准"
	@echo "make train       训一个动作跟踪策略（ALGO=ppo|sac MOTION=squat ITERS=200 ENVS=2048）"
	@echo "make ab-algos    PPO vs SAC，同墙钟预算对照（SECONDS=360）"
	@echo "make ab-backends mjbatch vs mujoco_warp，同环境代码对照"
	@echo "make demo        把最近训好的策略打包进 web/ 并在浏览器里验一遍"
	@echo "make view        实时交互 sim2sim：打包标准动作集（待机站立、WASD 走路、广播体操、舞蹈）并起服务"
	@echo "                 只加一个：make view RUN=bt_v6 LABEL=广播体操v6 [AUDIO=runs/bt/bt.mp3]"
	@echo "make serve       本地起静态服务，看 web/ 下的页面"

contract:
	$(PY) robots/q1/contract/gen_joints.py
	$(PY) robots/q1/build_model.py
	$(PY) robots/q1/build_model.py --self-contained
	$(PY) tools/export_contract_json.py
	$(PY) tools/make_golden_trace.py

validate:
	$(PY) robots/q1/contract/validate.py

# Deployment gate: the vendor's own policies must track through our observation builder.
deploy-check:
	$(PY) tools/check_vendor_policies.py

serve:
	@echo "http://127.0.0.1:$(PORT)/web/golden_check.html"
	$(PY) -m http.server $(PORT) --bind 127.0.0.1

web-check:
	@$(PY) -m http.server $(PORT) --bind 127.0.0.1 >/dev/null 2>&1 & echo $$! > .http.pid; \
	google-chrome --headless=new --no-sandbox --disable-gpu \
	  --remote-debugging-port=9223 --user-data-dir=/tmp/q1-chrome >/dev/null 2>&1 & echo $$! > .chrome.pid; \
	sleep 3; \
	node bench/cdp.js http://127.0.0.1:$(PORT)/web/golden_check.html 9223; \
	kill `cat .http.pid` `cat .chrome.pid` 2>/dev/null; rm -f .http.pid .chrome.pid

train:
	$(PY) train.py --algo $(ALGO) --motion $(MOTION) --envs $(ENVS) --iters $(ITERS)

ab-algos:
	$(PY) tools/ab_algos.py --seconds $(SECONDS) --envs 1024 --motion $(MOTION)

ab-backends:
	$(PY) tools/check_backend_writes.py
	$(PY) tools/compare_backends.py --envs 4096 --steps 60

demo:
	$(PY) tools/export_web_bundle.py runs/$(RUN)
	@$(PY) -m http.server $(PORT) --bind 127.0.0.1 >/dev/null 2>&1 & echo $$! > .http.pid; \
	google-chrome --headless=new --no-sandbox --disable-gpu \
	  --remote-debugging-port=9224 --user-data-dir=/tmp/q1-chrome2 >/dev/null 2>&1 & echo $$! > .chrome.pid; \
	sleep 3; \
	node bench/cdp.js "http://127.0.0.1:$(PORT)/web/sim2sim.html?run=$(RUN)" 9224; \
	kill `cat .http.pid` `cat .chrome.pid` 2>/dev/null; rm -f .http.pid .chrome.pid

# the interactive viewer's standard set; RUN=... LABEL=... adds (or refreshes) one more
VIEW_IDLE ?= stand_v6
VIEW_WALK ?= walk_v8
view:
	test -d web/node_modules/three || (cd web && npm install)
	test -f web/assets/q1_visual.bin || $(PY) tools/export_web_visual.py
ifneq ($(origin RUN),file)
	$(PY) tools/export_web_bundle.py runs/$(RUN) --label "$(or $(LABEL),$(RUN))" $(if $(AUDIO),--audio $(AUDIO))
else
	test ! -d runs/$(VIEW_IDLE) || $(PY) tools/export_web_bundle.py runs/$(VIEW_IDLE) --label 站立 --role idle
	test ! -d runs/$(VIEW_WALK) || $(PY) tools/export_web_bundle.py runs/$(VIEW_WALK) --label 走路 --role walk --order 0
	test ! -d runs/walk_v7 || $(PY) tools/export_web_bundle.py runs/walk_v7 --label 走路v7 --role walk --order 25
	test ! -d runs/walk_v3 || $(PY) tools/export_web_bundle.py runs/walk_v3 --label 走路v3 --role walk --order 30
	test ! -d runs/walk_v4 || $(PY) tools/export_web_bundle.py runs/walk_v4 --label 走路v4 --role walk --order 40
	test ! -d runs/velocity_v6 || $(PY) tools/export_web_bundle.py runs/velocity_v6 --label 走路·velocity --role walk
	$(PY) tools/export_web_bundle.py runs/bt_v3 --label 广播体操 --audio runs/bt/bt.mp3 --order 10
	$(PY) tools/export_web_bundle.py runs/ab_critic_asym --label 广播体操·非对称critic --audio runs/bt/bt.mp3 --order 11
	$(PY) tools/export_web_bundle.py runs/dance1_obs119 --label 舞蹈 --order 20
endif
	@echo "http://$$(hostname -I | cut -d' ' -f1):$(PORT)/web/sim2sim_view.html   (?scene=open|room|stage|terrain, ?walk=walk_v7|walk_v3|walk_v4|velocity_v6)"
	$(PY) -m http.server $(PORT) --bind $(HOST)

bench:
	$(PY) _ref/mjbatch/benchmarks/scaling.py robots/q1/q1_physics.xml --num-sims 4096 --threads 1,8,16,32
	$(PY) tools/check_mesh_substitution.py

clean:
	rm -f .http.pid .chrome.pid
