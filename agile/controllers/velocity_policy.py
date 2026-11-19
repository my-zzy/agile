"""Named, stateful G1 ONNX velocity-policy interface (raw state to joint radians).

The deployment graph owns observation scaling, histories, gait phase, and action
scaling. Isaac Lab 3 supplies xyzw quaternions; the export takes wxyz.
"""

import hashlib
from pathlib import Path

import numpy as np
import yaml


class OnnxVelocityPolicy:
    PRIMARY = {"root_link_quat_w", "root_ang_vel_b", "velocity_commands", "joint_pos", "joint_vel"}
    TARGETS = ("action_joint_pos", "action_joint_pos_kp_gains", "action_joint_pos_kd_gains")

    def __init__(self, checkpoint, contract):
        import onnxruntime as ort

        self.checkpoint, self.contract = Path(checkpoint), Path(contract)
        document = yaml.safe_load(self.contract.read_text())
        spec = document["models"]["Velocity-G1-History-v0"]
        self.sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        if self.sha256 != spec["parameters"]["sha256sum"]:
            raise ValueError("Velocity model checksum differs from its deployment contract")
        self.control_dt = float(document["semantic"]["scene"]["dt"])
        self.inputs = {item["name"]: item for item in spec["inputs"]}
        self.outputs = {item["name"]: item for item in spec["outputs"]}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.checkpoint), sess_options=options, providers=["CPUExecutionProvider"]
        )
        for actual, expected in ((self.session.get_inputs(), self.inputs), (self.session.get_outputs(), self.outputs)):
            if {item.name for item in actual} != set(expected):
                raise ValueError("ONNX named inputs/outputs differ from the contract")
            for item in actual:
                if item.type != "tensor(float)" or list(item.shape) != expected[item.name]["shape"]:
                    raise ValueError(f"ONNX shape or dtype mismatch: {item.name}")
        if self.inputs["root_link_quat_w"]["element_names"][1] != ["w", "x", "y", "z"]:
            raise ValueError("Expected a wxyz deployment quaternion")
        self.joint_names = self.inputs["joint_pos"]["element_names"][0]
        self.controlled_names = self.outputs["action_joint_pos"]["element_names"][0]
        if len(set(self.joint_names)) != 29 or len(set(self.controlled_names)) != 14:
            raise ValueError("Expected 29 observed joints and 14 controlled joints")
        if self.inputs["joint_vel"]["element_names"][0] != self.joint_names:
            raise ValueError("Joint position and velocity ordering must match")
        for name in self.TARGETS:
            if self.outputs[name]["element_names"][0] != self.controlled_names:
                raise ValueError("Joint target and PD gain ordering must match")
        for name in set(self.inputs) - self.PRIMARY:
            if name + "_out" not in self.outputs:
                raise ValueError(f"Missing recurrent output for {name}")
            if self.inputs[name]["shape"] != self.outputs[name + "_out"]["shape"]:
                raise ValueError(f"Recurrent state shape mismatch: {name}")
        self.reset()

    def reset(self):
        """Call only when the simulated robot resets, not between trajectory phases."""
        self.state = {
            name: np.zeros(item["shape"], np.float32) for name, item in self.inputs.items() if name not in self.PRIMARY
        }

    def step(self, quat_xyzw, angular_velocity_b, command, joint_pos, joint_vel):
        values = (quat_xyzw, angular_velocity_b, command, joint_pos, joint_vel)
        names = ("root_link_quat_w", "root_ang_vel_b", "velocity_commands", "joint_pos", "joint_vel")
        feed = dict(self.state)
        for name, value in zip(names, values, strict=True):
            value = np.asarray(value, np.float32)
            if list(value.shape) != self.inputs[name]["shape"] or not np.isfinite(value).all():
                raise ValueError(f"Invalid policy input: {name}")
            feed[name] = value
        quaternion = feed["root_link_quat_w"]
        if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1.0, atol=1e-3):
            raise ValueError("Root quaternion must have unit norm")
        feed["root_link_quat_w"] = quaternion[:, [3, 0, 1, 2]]
        result = dict(zip(self.outputs, self.session.run(list(self.outputs), feed), strict=True))
        for name, value in result.items():
            if list(value.shape) != self.outputs[name]["shape"] or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid policy output: {name}")
        self.state = {name: result[name + "_out"].copy() for name in self.state}
        return tuple(result[name] for name in self.TARGETS)

    def as_dict(self):
        return {
            "format": "stateful_onnx_velocity",
            "checkpoint": str(self.checkpoint.resolve()),
            "checkpoint_sha256": self.sha256,
            "contract": str(self.contract.resolve()),
            "contract_sha256": hashlib.sha256(self.contract.read_bytes()).hexdigest(),
            "provider": "CPUExecutionProvider",
            "control_dt_s": self.control_dt,
            "input_quaternion_conversion": "Isaac Lab xyzw to deployment wxyz",
            "observed_joint_names": self.joint_names,
            "controlled_joint_names": self.controlled_names,
            "recurrent_states": list(self.state),
            "action_units": "absolute joint radians; no additional scale or default offset",
        }
