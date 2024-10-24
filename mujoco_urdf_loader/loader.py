import idyntree.bindings as idyn
import xml.etree.ElementTree as ET
from typing import List, Union
import resolve_robotics_uri_py as rru
import tempfile
import dataclasses
import logging

from mujoco_urdf_loader.urdf_fcn import (
    add_mujoco_element,
    get_mesh_path,
    remove_gazebo_elements,
)
from mujoco_urdf_loader.mjcf_fcn import (
    add_position_actuator,
    add_torque_actuator,
    separate_left_right_collision_groups,
)
from mujoco_urdf_loader.generator import load_urdf_into_mjcf

from enum import Enum

class ControlMode(Enum):
    POSITION = "position"
    TORQUE = "torque"
    VELOCITY = "velocity"


@dataclasses.dataclass
class URDFtoMuJoCoLoaderCfg:
    controlled_joints: List[str]
    control_modes: Union[None, List[ControlMode]] = None
    stiffness: Union[None, List[float]] = None
    damping: Union[None, List[float]] = None
    

class URDFtoMuJoCoLoader:
    def __init__(self, mjcf: str, cfg: URDFtoMuJoCoLoaderCfg):
        """
        Initialize the URDF to Mujoco converter.

        Args:
            mjcf (str): The MuJoCo string.
            joints (List[str]): The list of joints to command.
        """
        self.mjcf = mjcf
        self.controlled_joints = cfg.controlled_joints
        if cfg.control_modes is None:
            self.control_mode = {joint: ControlMode.TORQUE for joint in cfg.controlled_joints}
        else:
            self.control_mode = {joint: mode for joint, mode in zip(cfg.controlled_joints, cfg.control_modes)}
        self.set_controlled_joints(cfg.controlled_joints)

    @staticmethod
    def load_urdf(urdf_path: str, cfg: URDFtoMuJoCoLoaderCfg):    
        """
        Load the URDF from the file.

        Args:
            urdf_path (Path): The URDF file path.
            cfg (URDFtoMuJoCoLoaderCfg): The configuration containing the controlled joints, control modes, stiffness and damping.

        Returns:
            str: The URDF string.
        """
        urdf_string = URDFtoMuJoCoLoader.simplify_urdf(urdf_path, cfg.controlled_joints, cfg.stiffness, cfg.damping)
        mesh_path = get_mesh_path(urdf_string)
        urdf_string = remove_gazebo_elements(urdf_string)
        urdf_string = add_mujoco_element(urdf_string, mesh_path)
        mjcf = load_urdf_into_mjcf(urdf_string)
        mjcf = separate_left_right_collision_groups(mjcf)
        return URDFtoMuJoCoLoader(mjcf, cfg)

    @staticmethod
    def simplify_urdf(urdf_path: str, joints: List[str], stiffness: List[float] = None, damping: List[float] = None):
        """
        Simplify the URDF using iDynTree.

        Args:
            urdf_path (str): The URDF string.
            joints (List[str]): The list of joints to command.
            stiffness (List[float]): The list of stiffness values.
            damping (List[float]): The list of damping values.

        Returns:
            str: The simplified URDF string.
        """

        # Load the URDF model
        model_loader = idyn.ModelLoader()
        if not model_loader.loadReducedModelFromFile(urdf_path, joints):
            raise ValueError(f"Error loading the URDF model from {urdf_path}")
        model = model_loader.model()

        if stiffness is not None:
            for i in range(model.getNrOfJoints()):
                joint = model.getJoint(i)
                for dof in range(joint.getNrOfDOFs()):
                    joint.setStaticFriction(dof, stiffness[i])

        if damping is not None:
            for i in range(model.getNrOfJoints()):
                joint = model.getJoint(i)
                for dof in range(joint.getNrOfDOFs()):
                    joint.setDamping(dof, damping[i])


        # Save the simplified model
        model_saver = idyn.ModelExporter()
        model_saver.init(model)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
            model_saver.exportModelToFile(temp_path)

        tree = ET.parse(temp_path)
        root = tree.getroot()

        # Find the joint with name "base_link_fixed_joint" and change its type to "floating"
        for joint in root.findall(".//joint"):
            if joint.attrib.get("name") == "base_link_fixed_joint":
                joint.attrib["type"] = "floating"
                logging.info(f"Modified joint base_link_fixed_joint to type floating.")
                break

        return root
    
    def set_control_mode(self, joint: Union[str, List[str]], mode: ControlMode):
        """
        Set the control mode for the joint.

        Args:
            joint (str): The joint name.
            mode (ControlMode): The control mode.
        """
        if isinstance(joint, str):
            self.control_mode[joint] = mode
        elif isinstance(joint, list):
            for j in joint:
                self.control_mode[j] = mode
        else:
            raise ValueError("joint must be a string or a list of strings.")

    def add_actuator(self, joint: str, control_mode: ControlMode, ctrlrange: List[float] = None):
        """
        Add an actuator to the MJCF model.

        Args:
            joint (str): The joint name.
            control_mode (ControlMode): The control mode.
            ctrlrange (List[float]): The control range.
        """
        if control_mode == ControlMode.POSITION:
            add_position_actuator(self.mjcf, joint=joint, ctrlrange=ctrlrange)
        elif control_mode == ControlMode.TORQUE:
            add_torque_actuator(self.mjcf, joint=joint, ctrlrange=ctrlrange)
        elif control_mode == ControlMode.VELOCITY:
            raise NotImplementedError("Velocity control is not implemented yet.")
        else:
            raise ValueError("Control mode not recognized.")

    def set_controlled_joints(self, joints: List[str]):
        """
        Set the controlled joints.

        Args:
            joints (List[str]): The list of joints.
        """
        self.controlled_joints = joints
        joint_elements = {joint.attrib["name"]: joint for joint in self.mjcf.findall(".//joint")}
        
        for controlled_joint in self.controlled_joints:
            joint_element = joint_elements.get(controlled_joint)
            if joint_element is not None:
                ctrlrange = list(map(float, joint_element.attrib["range"].split()))
                self.add_actuator(controlled_joint, self.control_mode[controlled_joint], ctrlrange)
            else:
                raise ValueError(f"Joint {controlled_joint} not found in the MJCF model.")

    def get_mjcf(self):
        """
        Get the Mujoco XML string.

        Returns:
            str: The Mujoco XML string.
        """
        return self.mjcf
    
    def get_mjcf_string(self):
        """
        Get the Mujoco XML string.

        Returns:
            str: The Mujoco XML string.
        """
        return ET.tostring(self.mjcf, encoding="unicode", method="xml")
    
