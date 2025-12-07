---
sidebar_position: 6
---

# Chapter 3: URDF for Humanoids

This chapter covers the Unified Robot Description Format (URDF) and its application to humanoid robot modeling in ROS 2.

## Learning Objectives

After completing this chapter, students will be able to:
- Create comprehensive URDF models for humanoid robots
- Define kinematic chains and joint constraints for humanoids
- Integrate visual, collision, and inertial properties in URDF
- Validate and visualize humanoid robot models in RViz

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML format used in ROS to describe robot models [@bohren2010]. For humanoid robots, URDF defines the physical structure, kinematic properties, and visual representation of the robot [@chitta2010].

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>
</robot>
```

## Humanoid Robot Kinematic Structure

Humanoid robots have a specific kinematic structure that typically includes:

- **Torso**: The central body with multiple degrees of freedom
- **Head**: Usually with pan and tilt joints
- **Arms**: Shoulders, elbows, wrists, and potentially hands
- **Legs**: Hips, knees, ankles, and feet
- **Optional**: Waist, neck, and finger joints

### Example: Humanoid Torso Definition

```xml
<!-- Torso of the humanoid -->
<link name="torso">
  <visual>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <geometry>
      <box size="0.2 0.15 0.6"/>
    </geometry>
    <material name="light_grey">
      <color rgba="0.7 0.7 0.7 1.0"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <geometry>
      <box size="0.2 0.15 0.6"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="5.0"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.25" iyz="0.0" izz="0.1"/>
  </inertial>
</link>
```

## Joint Definitions for Humanoid Robots

Joints connect links and define the kinematic relationships between them. Humanoid robots require various joint types:

### Revolute Joints (Limited Rotation)

```xml
<!-- Hip joint example -->
<joint name="left_hip_pitch" type="revolute">
  <parent link="torso"/>
  <child link="left_thigh"/>
  <origin xyz="0 -0.075 -0.3" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="3.14"/>
  <dynamics damping="1.0" friction="0.1"/>
</joint>
```

### Continuous Joints (Unlimited Rotation)

```xml
<!-- Waist yaw joint -->
<joint name="waist_yaw" type="continuous">
  <parent link="torso"/>
  <child link="upper_torso"/>
  <origin xyz="0 0 0.3" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <dynamics damping="2.0" friction="0.2"/>
</joint>
```

### Fixed Joints (No Movement)

```xml
<!-- Sensor mounting -->
<joint name="imu_mount" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.2" rpy="0 0 0"/>
</joint>
```

## Complete Humanoid URDF Example

Here's a more comprehensive example showing a simplified humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="tutorial_humanoid">
  <!-- Base link - pelvis -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.15 0.5"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.15 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.25" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1.0 1.0 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.004" ixy="0.0" ixz="0.0" iyy="0.004" iyz="0.0" izz="0.004"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.0 0.0 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.0025"/>
    </inertial>
  </link>

  <!-- Joint definitions -->
  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1.0"/>
  </joint>

  <joint name="torso_to_left_arm" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.075 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="1.5"/>
  </joint>
</robot>
```

## Materials and Colors

URDF allows defining materials for visualization:

```xml
<material name="black">
  <color rgba="0.0 0.0 0.0 1.0"/>
</material>
<material name="blue">
  <color rgba="0.0 0.0 0.8 1.0"/>
</material>
<material name="green">
  <color rgba="0.0 0.8 0.0 1.0"/>
</material>
<material name="grey">
  <color rgba="0.5 0.5 0.5 1.0"/>
</material>
<material name="orange">
  <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
</material>
<material name="brown">
  <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
</material>
<material name="red">
  <color rgba="0.8 0.0 0.0 1.0"/>
</material>
<material name="white">
  <color rgba="1.0 1.0 1.0 1.0"/>
</material>
```

## Transmission Elements

For controlling joints, transmission elements define the relationship between actuators and joints:

```xml
<transmission name="left_hip_pitch_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_hip_pitch">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_hip_pitch_motor">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## URDF Tools and Validation

### Checking URDF Validity

```bash
# Check if URDF is well-formed
check_urdf /path/to/robot.urdf

# Display robot information
urdf_to_graphiz /path/to/robot.urdf
```

### Visualizing in RViz

```bash
# Launch RViz with robot state publisher
ros2 run rviz2 rviz2
```

## Best Practices for Humanoid URDF

1. **Use consistent naming conventions**: `left_arm_shoulder_pitch` instead of `l_shldr_p`
2. **Define proper inertial properties**: Critical for simulation accuracy
3. **Include collision models**: For physics simulation
4. **Use appropriate joint limits**: Based on physical robot capabilities
5. **Group related parts**: Organize URDF for maintainability

## Xacro for Complex Models

For complex humanoid robots, Xacro (XML Macros) helps reduce redundancy:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Macro for creating arm links -->
  <xacro:macro name="arm_link" params="name parent xyz rpy material">
    <link name="${name}">
      <visual>
        <origin xyz="${xyz}" rpy="${rpy}"/>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
        <material name="${material}"/>
      </visual>
      <collision>
        <origin xyz="${xyz}" rpy="${rpy}"/>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <origin xyz="${xyz}" rpy="0 0 0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.0025"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Use the macro to create left and right arms -->
  <xacro:arm_link name="left_arm" parent="torso" xyz="0.1 0 0" rpy="0 0 0" material="red"/>
  <xacro:arm_link name="right_arm" parent="torso" xyz="-0.1 0 0" rpy="0 0 0" material="blue"/>

</robot>
```

## Research Tasks

1. Investigate the differences between URDF and SDF (Simulation Description Format) for humanoid robotics
2. Explore the use of mesh files (STL, DAE) in URDF for more realistic humanoid models
3. Analyze the impact of accurate inertial properties on humanoid robot simulation

## Evidence Requirements

Students must demonstrate understanding by:
- Creating a complete URDF model for a simplified humanoid robot
- Validating the URDF using ROS 2 tools
- Visualizing the robot in RViz with proper joint limits

## References

- Chitta, S., Marder-Eppstein, E., & Prats, M. (2010). Automatic inference of humanoid robot kinematics using the unified robot description format. *2010 IEEE International Conference on Robotics and Automation*, 1798-1803.
- Bohren, J., & Cousins, S. (2010). urdf: Unified Robot Description Format. *Robot Operating System*, 309-328.
- Diankov, R., & Kuffner, J. (2008). OpenRAVE: A planning architecture for autonomous robotics. *Carnegie Mellon University Tech. Report CMU-RI-TR-08-34*, 79.
- Siciliano, B., & Khatib, O. (2016). *Springer handbook of robotics*. Springer Publishing Company, Incorporated.

## Practical Exercises

1. Create a URDF file for a simple 6-DOF humanoid arm
2. Validate the URDF using the check_urdf tool
3. Visualize the robot in RViz and test joint movements
4. Create a Xacro macro for a humanoid leg and instantiate both left and right legs