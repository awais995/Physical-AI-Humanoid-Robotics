---
sidebar_position: 5
---

# Chapter 2: rclpy Integration

This chapter covers the Python client library for ROS 2 (rclpy) and how to integrate it with humanoid robotics applications.

## Learning Objectives

After completing this chapter, students will be able to:
- Create ROS 2 nodes using the rclpy library
- Implement custom message types and interfaces
- Handle asynchronous operations and callbacks in Python
- Integrate rclpy with humanoid robot control systems

## Introduction to rclpy

rclpy is the Python client library for ROS 2, providing Python bindings for the ROS 2 ecosystem [@rclpy2023]. It allows developers to create ROS 2 nodes, publish and subscribe to topics, provide and use services, and work with actions using Python [@lalancette2018].

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

class HumanoidNode(Node):
    def __init__(self):
        super().__init__('humanoid_node')
        self.get_logger().info('Humanoid node initialized')

    def __del__(self):
        self.get_logger().info('Humanoid node shutting down')

def main(args=None):
    rclpy.init(args=args)
    humanoid_node = HumanoidNode()

    try:
        rclpy.spin(humanoid_node)
    except KeyboardInterrupt:
        pass
    finally:
        humanoid_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Working with Parameters

Parameters in rclpy allow runtime configuration of nodes, which is essential for humanoid robot systems that may need different configurations for various tasks.

```python
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Declare parameters with default values
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('control_frequency', 50)
        self.declare_parameter('robot_name', 'default_robot')

        # Access parameter values
        self.max_velocity = self.get_parameter('max_velocity').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.robot_name = self.get_parameter('robot_name').value

        self.get_logger().info(
            f'Controller initialized for {self.robot_name} with max velocity {self.max_velocity}'
        )
```

## Custom Message Types

Creating custom message types is crucial for humanoid robotics applications where standard messages may not capture the specific data requirements.

### Example: Humanoid Joint State Message

```python
# In your package's msg directory, create HumanoidJointState.msg:
# string robot_name
# float64[] joint_positions
# float64[] joint_velocities
# float64[] joint_efforts
# string[] joint_names
# float64 timestamp
```

```python
import rclpy
from rclpy.node import Node
from your_robot_msgs.msg import HumanoidJointState  # Custom message type

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(
            HumanoidJointState,
            'humanoid_joint_states',
            10
        )

    def publish_joint_state(self, joint_data):
        msg = HumanoidJointState()
        msg.robot_name = 'hrp4c'
        msg.joint_names = joint_data['names']
        msg.joint_positions = joint_data['positions']
        msg.joint_velocities = joint_data['velocities']
        msg.joint_efforts = joint_data['efforts']
        msg.timestamp = self.get_clock().now().nanoseconds / 1e9

        self.publisher.publish(msg)
```

## Asynchronous Operations and Callbacks

Handling asynchronous operations efficiently is critical for real-time humanoid robot control.

```python
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
from std_msgs.msg import Float64MultiArray

class HumanoidControlNode(Node):
    def __init__(self):
        super().__init__('humanoid_control')

        # Create a callback group for thread safety
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.sensing_group = MutuallyExclusiveCallbackGroup()

        # Publishers and subscribers with specific callback groups
        self.control_pub = self.create_publisher(
            Float64MultiArray,
            'joint_commands',
            10,
            callback_group=self.control_group
        )

        self.sensor_sub = self.create_subscription(
            Float64MultiArray,
            'sensor_feedback',
            self.sensor_callback,
            10,
            callback_group=self.sensing_group
        )

    def sensor_callback(self, msg):
        # Process sensor data
        self.get_logger().info(f'Received sensor data: {len(msg.data)} values')

    def send_control_command(self, commands):
        msg = Float64MultiArray()
        msg.data = commands
        self.control_pub.publish(msg)
```

## Timer-Based Control

For humanoid robots requiring precise timing control, rclpy provides timer functionality:

```python
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from std_msgs.msg import Float64MultiArray
import math

class WalkingController(Node):
    def __init__(self):
        super().__init__('walking_controller')

        # Timer for walking control loop
        self.control_timer = self.create_timer(
            0.02,  # 50 Hz control loop
            self.walking_control_callback
        )

        self.command_publisher = self.create_publisher(
            Float64MultiArray,
            'walking_commands',
            10
        )

        self.step_phase = 0.0
        self.step_frequency = 0.5  # 0.5 Hz walking

    def walking_control_callback(self):
        # Generate walking pattern based on phase
        joint_commands = self.calculate_walking_pattern(self.step_phase)

        msg = Float64MultiArray()
        msg.data = joint_commands
        self.command_publisher.publish(msg)

        # Update phase for next step
        self.step_phase += 2 * math.pi * self.step_frequency * 0.02
        if self.step_phase >= 2 * math.pi:
            self.step_phase = 0.0

    def calculate_walking_pattern(self, phase):
        # Simplified walking pattern calculation
        # In real implementation, this would use inverse kinematics
        commands = [0.0] * 28  # Example for 28 DOF humanoid

        # Add phase-dependent adjustments for walking
        commands[0] = 0.1 * math.sin(phase)  # Hip movement
        commands[1] = 0.05 * math.cos(phase)  # Balance adjustment

        return commands
```

## Lifecycle Nodes

For humanoid robots that need controlled startup and shutdown sequences:

```python
import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.executors import SingleThreadedExecutor

class HumanoidLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('humanoid_lifecycle_node')
        self.get_logger().info('Lifecycle node created, waiting for configuration')

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring humanoid node')
        # Initialize hardware interfaces, load parameters
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Activating humanoid node')
        # Start publishers, subscribers, timers
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating humanoid node')
        # Stop active components safely
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up humanoid node')
        # Release resources
        return TransitionCallbackReturn.SUCCESS
```

## Integration with Humanoid Control Systems

rclpy nodes can be integrated with humanoid control systems through various approaches:

1. **Joint Control**: Direct control of individual joints
2. **Whole-Body Control**: Coordinated control of multiple joints
3. **Task-Space Control**: Control in Cartesian space

### Example: Joint Trajectory Controller

```python
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, GoalResponse, CancelResponse
import time

class HumanoidTrajectoryController(Node):
    def __init__(self):
        super().__init__('humanoid_trajectory_controller')

        # Action server for trajectory execution
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'humanoid_joint_trajectory',
            execute_callback=self.execute_trajectory,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

    def goal_callback(self, goal_request):
        # Validate trajectory goal
        self.get_logger().info('Received trajectory goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        # Handle trajectory cancellation
        self.get_logger().info('Trajectory cancellation requested')
        return CancelResponse.ACCEPT

    def execute_trajectory(self, goal_handle):
        # Execute the joint trajectory
        trajectory = goal_handle.request.trajectory
        self.get_logger().info(f'Executing trajectory with {len(trajectory.points)} points')

        # Send trajectory points to hardware interface
        for point in trajectory.points:
            # Send position, velocity, acceleration to joints
            self.send_joint_command(point)
            time.sleep(0.01)  # Small delay between points

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        return result

    def send_joint_command(self, point):
        # Interface with actual hardware (simplified)
        self.get_logger().info(f'Sending joint command: {point.positions}')
```

## Research Tasks

1. Investigate the performance implications of using Python vs C++ for real-time humanoid control
2. Explore the integration of rclpy with existing Python-based robotics frameworks like PyRobot
3. Analyze the memory and CPU usage of rclpy nodes in resource-constrained humanoid platforms

## Evidence Requirements

Students must demonstrate understanding by:
- Creating a parameterized rclpy node for humanoid joint control
- Implementing a custom message type for humanoid-specific data
- Developing a multi-threaded rclpy node that handles both sensor input and control output

## References

- ROS 2 Python Client Library Working Group. (2023). rclpy: Python Client Library for ROS 2. *ROS Official Documentation*. https://docs.ros.org/en/rolling/p/rclpy/
- Lalancette, C., Dube, D., & Perez, J. (2018). The ROS 2 system overview. *Proceedings of the 7th European Conference on Mobile Robots*, 1-8.
- Kammerl, J., Blodow, N., & Rusu, R. B. (2012). Real-time RGB-D based ground plane detection and inverse perspective mapping. *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 4234-4241.
- Bohren, J., & Cousins, S. (2010). urdf: Unified Robot Description Format. *Robot Operating System*, 309-328.

## Practical Exercises

1. Create an rclpy node that publishes humanoid joint states with parameters for joint limits
2. Implement a subscriber node that processes custom humanoid messages
3. Develop a timer-based controller that generates periodic joint commands for a humanoid robot