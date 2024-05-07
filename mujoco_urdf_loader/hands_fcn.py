import xml.etree.ElementTree as ET
from typing import List

import numpy as np

from .mjcf_fcn import add_joint_eq, add_position_actuator


def add_hand_equalities(mjcf: ET.Element) -> ET.Element:
    """Add the hand equalities to the mjcf file.

    Args:
        mjcf (ET.Element): The mjcf file.
    """

    for joint in mjcf.findall(".//joint"):
        # all dist joints are equal to the proximal joints
        if "dist" in joint.attrib["name"]:
            add_joint_eq(
                mjcf, joint.attrib["name"], joint.attrib["name"].replace("dist", "prox")
            )
        # the pinkie prox joint is equal to the ring prox joint
        if "prox" in joint.attrib["name"]:
            if "pinkie" in joint.attrib["name"]:
                add_joint_eq(
                    mjcf,
                    joint1=joint.attrib["name"],
                    joint2=joint.attrib["name"].replace("pinkie", "ring"),
                )
    return mjcf


def add_hand_actuators(mjcf: ET.Element, hand_elements: List[str]) -> ET.Element:
    """Add the hand actuators to the mjcf file.

    Args:
        mjcf (ET.Element): The mjcf file.
        hand_elements (List[str]): The hand elements.
        joint_limits (dict): The joint limits.
    """

    for joint in mjcf.findall(".//body/joint"):
        if any(
            joint_element in joint.attrib["name"] for joint_element in hand_elements
        ):
            ctrlrange = joint.attrib["range"]
            if "prox" in joint.attrib["name"] and "pinkie" not in joint.attrib["name"]:
                add_position_actuator(
                    mjcf,
                    joint=joint.attrib["name"],
                    ctrlrange=[
                        float(ctrlrange.split()[0]),
                        float(ctrlrange.split()[1]),
                    ],
                    kp=10,
                    group=1 if "r_" in joint.attrib["name"] else 2,
                    name=joint.attrib["name"].replace("prox", "motor"),
                )
            if "add" in joint.attrib["name"]:
                add_position_actuator(
                    mjcf,
                    joint=joint.attrib["name"],
                    ctrlrange=[
                        float(ctrlrange.split()[0]),
                        float(ctrlrange.split()[1]),
                    ],
                    kp=10,
                    group=1 if "r_" in joint.attrib["name"] else 2,
                    name=joint.attrib["name"] + "_motor",
                )

    return mjcf


def add_wrist_actuators(mjcf: ET.Element) -> ET.Element:
    """Add the wrist actuators to the mjcf file.

    Args:
        mjcf (ET.Element): The mjcf file.
        joint_limits (dict): The joint limits.
    """

    for joint in mjcf.findall(".//body/joint"):
        if "wrist" in joint.attrib["name"]:
            ctrlrange = joint.attrib["range"]
            add_position_actuator(
                mjcf,
                joint=joint.attrib["name"],
                ctrlrange=[float(ctrlrange.split()[0]), float(ctrlrange.split()[1])],
                kp=100,
                group=1 if "r_" in joint.attrib["name"] else 2,
                name=joint.attrib["name"] + "_motor",
            )

    return mjcf


def set_thumb_angle(mjcf: ET.Element, angle: float) -> ET.Element:
    """Set the angle of the thumb joint in the mjcf file.

    Args:
        mjcf (ET.Element): The mjcf file.
        angle (float): The angle of the thumb joint (deg).
    """
    angle_rad = angle * np.pi / 180

    for body in mjcf.findall(".//body"):
        if "thumb_2" in body.attrib["name"]:
            original_quat = body.attrib["quat"].split(" ")
            if "r_" in body.attrib["name"]:
                original_quat[3] = str(float(original_quat[3]) + angle_rad / 2)
            else:
                original_quat[2] = str(float(original_quat[2]) + angle_rad / 2)
            body.set("quat", " ".join(original_quat))

    return mjcf
    return mjcf
