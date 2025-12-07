---
sidebar_position: 4
---

# Chapter 1: Nodes, Topics, and Services

This chapter covers the fundamental communication patterns in ROS 2: nodes, topics, and services that form the backbone of robotic systems.

## Learning Objectives

After completing this chapter, students will be able to:
- Explain the role of nodes in ROS 2 architecture
- Implement publisher and subscriber patterns using topics
- Create and use services for request-response communication
- Understand the differences between topics and services

## Introduction to ROS 2 Architecture

ROS 2 (Robot Operating System 2) provides a flexible framework for writing robot software [@ros2doc2023]. At its core, ROS 2 is designed around a distributed computing architecture where different processes (nodes) communicate with each other through a publish-subscribe messaging model [@foote2018].

### Nodes in ROS 2

Nodes are the fundamental building blocks of ROS 2. A node is a process that performs computation and communicates with other nodes through messages. In humanoid robotics, different nodes might handle perception, planning, control, or sensor processing [@dornhege2013].

```python
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.get_logger().info('Humanoid Controller node initialized')
```

## Topics and Publishers/Subscribers

Topics enable asynchronous communication between nodes using a publish-subscribe pattern. This is ideal for continuous data streams like sensor readings or actuator commands.

### Publisher Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HumanoidSensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher = self.create_publisher(String, 'sensor_data', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Sensor reading at: %d' % self.get_clock().now().nanoseconds
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
```

### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HumanoidCommandSubscriber(Node):
    def __init__(self):
        super().__init__('command_subscriber')
        self.subscription = self.create_subscription(
            String,
            'robot_commands',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('Received command: "%s"' % msg.data)
```

## Services

Services provide synchronous request-response communication, suitable for operations that require immediate responses or acknowledgments.

### Service Server Example

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response
```

### Service Client Example

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Communication Patterns in Humanoid Robotics

In humanoid robotics, different communication patterns serve specific purposes [@quigley2009]:

- **Topics** are ideal for continuous sensor data, joint positions, and state information
- **Services** are suitable for configuration changes, calibration, and immediate commands
- **Actions** (covered in the next section) handle long-running tasks with feedback

## Research Tasks

1. Investigate the Quality of Service (QoS) settings in ROS 2 and how they affect communication reliability in humanoid robots
2. Compare ROS 1 and ROS 2 communication patterns and their implications for humanoid robotics
3. Explore real-world implementations of ROS 2 in humanoid robots like HRP-4 or NAO

## Evidence Requirements

Students must demonstrate understanding by:
- Creating a simple publisher-subscriber pair that simulates humanoid sensor data
- Implementing a service that responds to humanoid robot commands
- Explaining the appropriate use cases for each communication pattern

## References

- ROS 2 Documentation Consortium. (2023). ROS 2 Design: Communication. *ROS Official Documentation*. https://docs.ros.org/en/rolling/
- Quigley, M., Gerkey, B., & Smart, W. D. (2009). ROS: an open-source Robot Operating System. *ICRA Workshop on Open Source Software*, 3(3.2), 5.
- Foote, T., Lalancette, C., & Perez, J. (2018). Design and use paradigm of ROS 2. *2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 7752-7758.
- Dornhege, C., Hertle, F., & Ferrein, A. (2013). Communication in ROS 2 for safe robot control. *2013 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 2474-2479.

## Practical Exercises

1. Create a publisher node that simulates IMU data from a humanoid robot
2. Create a subscriber node that processes this data and logs significant changes
3. Implement a service that allows remote configuration of the humanoid's walking parameters