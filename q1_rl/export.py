"""Export a trained policy as the ONNX the robot's runtime will actually load.

Three things this does that a plain torch.onnx.export would not:

  * folds observation normalisation into the graph, so there is no side-car state that can
    be forgotten at deploy time;
  * declares the input at the width the robot's runtime feeds (obs_nums: 119 for
    MIMIC_OPENSOURCE_V2) and uses only the first obs_dim (116) of it. The runtime's config
    says obs_nums must equal the ONNX input width, so a 116-wide graph is rejected by
    onnxruntime the moment mc_main hands it 119 floats;
  * writes the contract identity into the ONNX metadata, so a file found on the robot six
    months from now can be traced back to the layout it was trained for.

After export, run validate.py --onnx on the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class DeployPolicy(nn.Module):
    """obs (batch, onnx_obs_dim) -> action (batch, act_dim), deterministic, no sampling.

    The input is as wide as the runtime feeds; everything past obs_dim is padding the
    runtime fills with zeros and the policy ignores. Callers that build obs_dim-wide
    vectors (the browser, our own bridge) pad with Contract.pad_for_onnx().

    Takes whatever the trainer produced as long as it offers `.norm` and either a
    `deterministic(obs)` method (SAC: tanh of the mean, scaled) or an `.actor` module
    (PPO: the mean directly). One export path for both keeps the deployment contract from
    acquiring a per-algorithm special case.
    """

    def __init__(self, net, obs_dim: int, onnx_obs_dim: int):
        super().__init__()
        self.net = net
        self.obs_dim, self.onnx_obs_dim = obs_dim, onnx_obs_dim
        self.has_deterministic = hasattr(net, "deterministic")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs[:, : self.obs_dim]
        if self.has_deterministic:
            return self.net.deterministic(obs)
        return self.net.actor(self.net.norm(obs))


class StackedDeployPolicy(nn.Module):
    """The Walk runtime's signature: (observations (b, 71), observation_history (b, 5 x 71))
    -> action (b, 22), the policy's own 13 and then zeros for the joints the runtime holds.

    The network was trained on the history alone, newest frame last. That frame IS the
    `observations` input, so it is taken from there: both inputs stay in the graph (the
    runtime feeds both by name) and a runtime that disagreed with itself would show here."""

    def __init__(self, net, frame_dim: int, stack: int, n_policy: int, n_joints: int):
        super().__init__()
        self.net = net
        self.frame_dim, self.stack, self.n_policy, self.n_joints = frame_dim, stack, n_policy, n_joints

    def forward(self, observations: torch.Tensor, observation_history: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observation_history[:, : (self.stack - 1) * self.frame_dim], observations], dim=1)
        a = self.net.actor(self.net.norm(x))[:, : self.n_policy]
        return torch.cat([a, torch.zeros(a.shape[0], self.n_joints - self.n_policy)], dim=1)


def export_onnx_stacked(net, contract, path, algo: str = "", opset: int = 17) -> Path:
    """Export for a frame-stacked contract (walk): the vendor's input and output names, widths
    and the 9 trailing zeros, checked against torch and against the vendor's ONNX signature."""
    path = Path(path)
    net = net.to("cpu").eval()
    fd, k = contract.obs_dim, contract.frame_stack
    npol = int(contract.spec["policy_action_dim"])
    policy = StackedDeployPolicy(net, fd, k, npol, contract.n_joints).eval()
    hist = torch.randn(4, k * fd)
    cur = hist[:, -fd:].clone()
    with torch.no_grad():
        ref = policy(cur, hist).numpy()
    torch.onnx.export(
        policy, (torch.zeros(1, fd), torch.zeros(1, k * fd)), str(path),
        input_names=["observations", "observation_history"], output_names=["action"],
        dynamic_axes={"observations": {0: "batch"}, "observation_history": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
    )
    import onnx
    m = onnx.load(str(path))
    meta = {
        "q1_contract": contract.name, "q1_obs_dim": str(fd), "q1_frame_stack": str(k),
        "q1_policy_action_dim": str(npol), "q1_action_dim": str(contract.n_joints),
        "q1_action_scale": str(contract.action_scale), "q1_policy_hz": str(contract.policy_hz),
        "q1_joint_order": json.dumps(contract.joint_names), "q1_algo": algo,
    }
    for key, v in meta.items():
        e = m.metadata_props.add()
        e.key, e.value = key, v
    onnx.save(m, str(path))
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    sig = {i.name: i.shape[-1] for i in sess.get_inputs()}
    want = {"observations": fd, "observation_history": k * fd}
    if sig != want:
        raise SystemExit(f"exported ONNX inputs {sig}, the Walk runtime feeds {want}")
    got = sess.run(None, {"observations": cur.numpy(), "observation_history": hist.numpy()})[0]
    err = float(np.abs(got - ref).max())
    if err > 1e-4:
        raise SystemExit(f"exported ONNX disagrees with torch by {err:.2e}")
    if got.shape[1] != contract.n_joints or np.abs(got[:, npol:]).max() > 0:
        raise SystemExit("exported ONNX does not end in zeros for the runtime's joints")
    print(f"  onnxruntime: inputs {sig}, output {got.shape[1]} (last {contract.n_joints - npol} zero), "
          f"agrees with torch to {err:.2e}")
    return path


def export_onnx(net, contract, path, motion_name: str = "", algo: str = "",
                opset: int = 17, obs_dim: int | None = None) -> Path:
    """obs_dim: the network's input width when it is not the contract's; the input is then
    taken as is."""
    if contract.frame_stack > 1:
        return export_onnx_stacked(net, contract, path, algo=algo, opset=opset)
    path = Path(path)
    net = net.to("cpu").eval()
    o, oo = (obs_dim, obs_dim) if obs_dim else (contract.obs_dim, contract.onnx_obs_dim)
    policy = DeployPolicy(net, o, oo).eval()
    dummy = torch.zeros(1, oo)
    probe = torch.randn(4, oo)
    with torch.no_grad():
        ref = policy(probe).numpy()

    torch.onnx.export(
        policy, (dummy,), str(path),
        input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
    )

    import onnx
    m = onnx.load(str(path))
    meta = {
        "q1_contract": contract.name,
        "q1_obs_dim": str(o),
        "q1_onnx_obs_dim": str(oo),
        "q1_action_dim": str(contract.n_joints),
        "q1_action_scale": str(contract.action_scale),
        "q1_policy_hz": str(contract.policy_hz),
        "q1_joint_order": json.dumps(contract.joint_names),
        "q1_motion": motion_name,
        "q1_algo": algo,
    }
    for k, v in meta.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, str(path))

    # Prove the saved file (a) takes exactly the width the runtime feeds, (b) reproduces the
    # torch output, and (c) really ignores the padding -- before anyone trusts it.
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (install onnxruntime to verify the exported graph)")
        return path
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    width = sess.get_inputs()[0].shape[-1]
    if width != oo:
        raise SystemExit(f"exported ONNX takes {width} wide, runtime feeds {oo}")
    got = sess.run(None, {"obs": probe.numpy()})[0]
    err = float(np.abs(got - ref).max())
    # float32 through two different runtimes: a few 1e-5 apart is rounding, not a wrong
    # graph (a walk policy came out at 1.9e-5). A layout or normaliser mistake is >= 1e-2.
    if err > 1e-4:
        raise SystemExit(f"exported ONNX disagrees with torch by {err:.2e}")
    if oo > o:
        noisy = probe.numpy().copy()
        noisy[:, o:] = 123.0
        leak = float(np.abs(sess.run(None, {"obs": noisy})[0] - got).max())
        if leak > 0:
            raise SystemExit(f"exported ONNX reads the padding: output moved by {leak:.2e}")
    print(f"  onnxruntime: input {width} wide, agrees with torch to {err:.2e}, padding ignored")
    return path
