"""Exercise quaternion conversion and recurrent ONNX feedback without a simulator."""

import hashlib

import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper

from agile.controllers.velocity_policy import OnnxVelocityPolicy


@pytest.fixture
def exported_policy(tmp_path):
    shapes = {
        "root_link_quat_w": [1, 4],
        "root_ang_vel_b": [1, 3],
        "velocity_commands": [1, 3],
        "joint_pos": [1, 29],
        "joint_vel": [1, 29],
        "memory": [1, 14],
    }
    outputs = ["action_joint_pos", "action_joint_pos_kp_gains", "action_joint_pos_kd_gains", "memory_out"]
    initializers = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="first"),
        numpy_helper.from_array(np.full((1, 14), 100, np.float32), name="kp"),
        numpy_helper.from_array(np.full((1, 14), 2.5, np.float32), name="kd"),
    ]
    nodes = [
        helper.make_node("Gather", ["root_link_quat_w", "first"], ["w"], axis=1),
        helper.make_node("Gather", ["velocity_commands", "first"], ["vx"], axis=1),
        helper.make_node("Add", ["w", "vx"], ["increment"]),
        helper.make_node("Add", ["memory", "increment"], ["action_joint_pos"]),
        helper.make_node("Identity", ["action_joint_pos"], ["memory_out"]),
        helper.make_node("Identity", ["kp"], ["action_joint_pos_kp_gains"]),
        helper.make_node("Identity", ["kd"], ["action_joint_pos_kd_gains"]),
    ]
    graph = helper.make_graph(
        nodes,
        "known_recurrent_policy",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in shapes.items()],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 14]) for name in outputs],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    checkpoint = tmp_path / "policy.onnx"
    onnx.save(model, checkpoint)
    inputs = [{"name": name, "shape": shape} for name, shape in shapes.items()]
    inputs[0]["element_names"] = [["pelvis"], ["w", "x", "y", "z"]]
    for item in inputs:
        if item["name"] in ("joint_pos", "joint_vel"):
            item["element_names"] = [[f"joint_{i}" for i in range(29)]]
    # Reverse metadata ordering to ensure outputs are requested by name.
    out_spec = [
        {"name": name, "shape": [1, 14], "element_names": [[f"joint_{i}" for i in range(14)]]}
        for name in reversed(outputs)
    ]
    document = {
        "models": {
            "Velocity-G1-History-v0": {
                "inputs": inputs,
                "outputs": out_spec,
                "parameters": {"sha256sum": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
            }
        },
        "semantic": {"scene": {"dt": 0.02}},
    }
    contract = checkpoint.with_suffix(".yaml")
    contract.write_text(yaml.safe_dump(document))
    return checkpoint, contract


def test_deployment_quaternion_history_reset_and_absolute_actions(exported_policy):
    policy = OnnxVelocityPolicy(*exported_policy)
    inputs = (
        np.array([[np.sqrt(0.75), 0, 0, 0.5]]),
        np.zeros((1, 3)),
        np.array([[2, 0, 0]]),
        np.zeros((1, 29)),
        np.zeros((1, 29)),
    )
    target, kp, kd = policy.step(*inputs)
    np.testing.assert_allclose(target, 2.5)
    np.testing.assert_allclose(kp, 100)
    np.testing.assert_allclose(kd, 2.5)
    np.testing.assert_allclose(policy.step(*inputs)[0], 5)
    policy.reset()
    np.testing.assert_allclose(policy.step(*inputs)[0], 2.5)


def test_rejects_wrong_model_and_invalid_state(exported_policy):
    checkpoint, contract = exported_policy
    policy = OnnxVelocityPolicy(checkpoint, contract)
    with pytest.raises(ValueError, match="unit norm"):
        policy.step(np.zeros((1, 4)), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 29)), np.zeros((1, 29)))
    assert not policy.state["memory"].any()
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        OnnxVelocityPolicy(checkpoint, contract)
