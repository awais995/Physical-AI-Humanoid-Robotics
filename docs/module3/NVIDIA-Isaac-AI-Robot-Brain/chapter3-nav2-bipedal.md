---
sidebar_position: 6
---

# Chapter 3: Nav2 for Bipedal Path Planning

This chapter covers configuring and using the Navigation2 stack specifically for bipedal humanoid robot path planning and navigation.

## Learning Objectives

After completing this chapter, students will be able to:
- Configure Nav2 for humanoid robot-specific navigation requirements
- Implement bipedal locomotion planning within the Nav2 framework
- Integrate balance constraints into path planning algorithms
- Optimize navigation parameters for humanoid locomotion characteristics

## Introduction to Nav2 for Humanoid Robots

Navigation2 (Nav2) is the next-generation navigation stack for ROS 2, designed to provide robust, reliable, and flexible navigation capabilities. For humanoid robots, Nav2 requires special configuration to account for:

- Bipedal locomotion constraints
- Balance and stability requirements
- Unique kinematic properties
- Higher computational requirements for balance control

[@macenski2022; @marder2015]

## Nav2 Architecture for Humanoid Robots

### Core Components

```python
# Humanoid-specific Nav2 node configuration
import rclpy
from rclpy.node import Node
from nav2_behavior_tree.bt_executor import BehaviorTreeExecutor
from nav2_core.local_planner import LocalPlanner
from nav2_core.global_planner import GlobalPlanner
from nav2_core.recovery import Recovery
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from builtin_interfaces.msg import Duration

class HumanoidNav2Node(Node):
    def __init__(self):
        super().__init__('humanoid_nav2_node')

        # Initialize humanoid-specific planners
        self.global_planner = HumanoidGlobalPlanner()
        self.local_planner = HumanoidLocalPlanner()
        self.recovery_system = HumanoidRecoverySystem()

        # Humanoid-specific parameters
        self.step_size = self.declare_parameter('step_size', 0.3).value
        self.step_width = self.declare_parameter('step_width', 0.2).value
        self.max_step_height = self.declare_parameter('max_step_height', 0.1).value
        self.balance_margin = self.declare_parameter('balance_margin', 0.1).value

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)

    def plan_global_path(self, start, goal):
        """Plan global path considering humanoid constraints"""
        # Generate footstep plan instead of simple path
        footstep_plan = self.global_planner.plan_footsteps(start, goal)

        # Convert to center of mass trajectory
        com_trajectory = self.convert_to_com_trajectory(footstep_plan)

        return com_trajectory

    def plan_local_path(self, current_pose, global_plan):
        """Plan local path with real-time humanoid constraints"""
        # Generate local footstep plan
        local_footsteps = self.local_planner.plan_local_footsteps(
            current_pose, global_plan
        )

        # Check balance constraints
        balanced_plan = self.enforce_balance_constraints(local_footsteps)

        return balanced_plan

    def convert_to_com_trajectory(self, footsteps):
        """Convert footstep plan to center of mass trajectory"""
        com_trajectory = []

        for step in footsteps:
            # Calculate CoM position for stable walking
            com_pos = self.calculate_stable_com(step)
            com_trajectory.append(com_pos)

        return com_trajectory

    def enforce_balance_constraints(self, footsteps):
        """Enforce balance constraints on footstep plan"""
        balanced_footsteps = []

        for i, step in enumerate(footsteps):
            # Calculate zero moment point
            zmp = self.calculate_zmp(step)

            # Verify within support polygon
            if self.is_stable_zmp(zmp, step.support_polygon):
                balanced_footsteps.append(step)
            else:
                # Adjust step to maintain balance
                adjusted_step = self.adjust_for_balance(step)
                balanced_footsteps.append(adjusted_step)

        return balanced_footsteps
```

[@khatib1986; @kajita2001]

## Humanoid Global Planner

### Footstep Planning Algorithm

```python
import numpy as np
from nav2_core.global_planner import GlobalPlanner
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class HumanoidGlobalPlanner(GlobalPlanner):
    def __init__(self):
        super().__init__()
        self.step_planner = FootstepPlanner()
        self.balance_checker = BalanceChecker()
        self.terrain_analyzer = TerrainAnalyzer()

    def create_plan(self, start, goal, map):
        """Create global plan with humanoid-specific constraints"""
        # Plan footsteps considering terrain
        footsteps = self.plan_footsteps_with_terrain(start, goal, map)

        # Verify plan stability
        if self.verify_plan_stability(footsteps):
            return self.footsteps_to_path(footsteps)
        else:
            # Plan alternative route
            alternative = self.plan_alternative_route(start, goal, map)
            return self.footsteps_to_path(alternative)

    def plan_footsteps_with_terrain(self, start, goal, map):
        """Plan footsteps considering terrain characteristics"""
        # Analyze terrain for walkable areas
        walkable_regions = self.terrain_analyzer.find_walkable_regions(map)

        # Plan footsteps using A* with humanoid constraints
        footsteps = []
        current_pos = start

        while not self.near_goal(current_pos, goal):
            # Find next suitable step position
            next_step = self.find_next_step_position(
                current_pos, goal, walkable_regions
            )

            if next_step:
                footsteps.append(next_step)
                current_pos = next_step
            else:
                # No valid path found
                return []

        return footsteps

    def find_next_step_position(self, current_pos, goal, walkable_regions):
        """Find next step position considering humanoid constraints"""
        # Generate candidate step positions
        candidates = self.generate_step_candidates(current_pos)

        # Evaluate candidates based on:
        # 1. Walkability
        # 2. Balance constraints
        # 3. Distance to goal
        # 4. Terrain characteristics

        best_candidate = None
        best_score = float('inf')

        for candidate in candidates:
            if self.is_walkable(candidate, walkable_regions):
                score = self.evaluate_candidate(candidate, current_pos, goal)
                if score < best_score:
                    best_score = score
                    best_candidate = candidate

        return best_candidate

    def generate_step_candidates(self, current_pos):
        """Generate potential step positions around current position"""
        candidates = []

        # Generate steps in multiple directions
        for angle in np.linspace(0, 2*np.pi, 8):
            for distance in [0.2, 0.3, 0.4]:  # Step distances
                x = current_pos.x + distance * np.cos(angle)
                y = current_pos.y + distance * np.sin(angle)

                candidate = PoseStamped()
                candidate.pose.position.x = x
                candidate.pose.position.y = y
                candidate.pose.position.z = 0.0  # Ground level
                candidate.pose.orientation.w = 1.0  # No rotation initially

                candidates.append(candidate)

        return candidates

    def evaluate_candidate(self, candidate, current_pos, goal):
        """Evaluate candidate step position"""
        # Distance to goal (lower is better)
        dist_to_goal = self.calculate_distance(candidate, goal)

        # Distance from current position (should be within step limits)
        step_distance = self.calculate_distance(candidate, current_pos)
        if step_distance > 0.5:  # Max step distance
            return float('inf')  # Invalid step

        # Balance score
        balance_score = self.evaluate_balance_score(candidate, current_pos)

        # Terrain score
        terrain_score = self.evaluate_terrain_score(candidate)

        # Combined score
        total_score = dist_to_goal + balance_score * 0.3 + terrain_score * 0.2

        return total_score

    def evaluate_balance_score(self, candidate, previous_pos):
        """Evaluate balance score for candidate step"""
        # Calculate if the step maintains balance
        # This involves checking the support polygon and ZMP
        return self.balance_checker.evaluate_step_balance(candidate, previous_pos)

    def evaluate_terrain_score(self, candidate):
        """Evaluate terrain suitability for stepping"""
        # Check for obstacles, slopes, etc.
        return self.terrain_analyzer.evaluate_terrain_at(candidate)
```

[@feng2013; @wobken2019]

## Humanoid Local Planner

### Dynamic Footstep Adjustment

```python
from nav2_core.local_planner import LocalPlanner
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
import math

class HumanoidLocalPlanner(LocalPlanner):
    def __init__(self):
        super().__init__()
        self.obstacle_detector = ObstacleDetector()
        self.balance_controller = BalanceController()
        self.footstep_adjuster = FootstepAdjuster()

    def compute_velocity_commands(self, pose, velocity, goal_checker):
        """Compute velocity commands with real-time humanoid constraints"""
        # Get sensor data for obstacle detection
        obstacles = self.obstacle_detector.get_obstacles()

        # Plan local footsteps considering obstacles
        local_footsteps = self.plan_local_footsteps_with_obstacles(
            pose, obstacles
        )

        # Check for balance constraints
        balanced_footsteps = self.adjust_footsteps_for_balance(
            local_footsteps, pose
        )

        # Generate velocity commands for next step
        cmd_vel = self.generate_step_command(balanced_footsteps[0])

        return cmd_vel, self.create_path_info()

    def plan_local_footsteps_with_obstacles(self, current_pose, obstacles):
        """Plan local footsteps avoiding detected obstacles"""
        # Get global plan reference
        global_plan = self.get_global_plan()

        # Plan local footsteps to follow global plan
        local_footsteps = []

        # Consider immediate obstacles
        for i in range(5):  # Plan next 5 steps
            if i < len(global_plan):
                target_step = global_plan[i]

                # Check if path is blocked
                if self.path_is_blocked(target_step, obstacles):
                    # Find alternative step
                    alternative = self.find_alternative_step(
                        current_pose, target_step, obstacles
                    )
                    local_footsteps.append(alternative)
                else:
                    local_footsteps.append(target_step)
            else:
                # Generate additional steps if needed
                next_step = self.generate_next_step(
                    local_footsteps[-1] if local_footsteps else current_pose
                )
                local_footsteps.append(next_step)

        return local_footsteps

    def path_is_blocked(self, step, obstacles):
        """Check if path to step is blocked by obstacles"""
        # Check for obstacles in path to step
        for obstacle in obstacles:
            distance = self.calculate_distance(step, obstacle)
            if distance < 0.5:  # Within safety margin
                return True
        return False

    def find_alternative_step(self, current_pose, target_step, obstacles):
        """Find alternative step position avoiding obstacles"""
        # Generate candidate steps around target
        candidates = self.generate_alternative_candidates(
            target_step, current_pose
        )

        # Evaluate candidates for safety and balance
        best_candidate = None
        best_score = float('inf')

        for candidate in candidates:
            if not self.is_obstacle_near(candidate, obstacles):
                score = self.evaluate_alternative_candidate(
                    candidate, target_step, current_pose
                )
                if score < best_score:
                    best_score = score
                    best_candidate = candidate

        return best_candidate if best_candidate else target_step

    def generate_next_step(self, previous_step):
        """Generate next step based on walking pattern"""
        # Use inverted pendulum model for step generation
        next_step = PoseStamped()

        # Calculate next step position based on walking pattern
        step_length = 0.3  # Typical step length for humanoid
        step_width = 0.2   # Step width for stability

        # Calculate direction towards goal
        if hasattr(self, 'current_goal'):
            direction = self.calculate_direction(previous_step, self.current_goal)
        else:
            direction = 0  # Default forward direction

        # Calculate next step position
        next_step.pose.position.x = previous_step.pose.position.x + step_length * math.cos(direction)
        next_step.pose.position.y = previous_step.pose.position.y + step_length * math.sin(direction)

        return next_step

    def generate_step_command(self, next_step):
        """Generate velocity command for next step"""
        cmd_vel = Twist()

        # Calculate required velocity to reach next step
        # This would involve complex bipedal control algorithms
        cmd_vel.linear.x = 0.2  # Forward velocity
        cmd_vel.angular.z = 0.0  # No rotation for now

        return cmd_vel
```

[@kunze2015; @hur2019]

## Balance-Aware Path Planning

### Center of Mass Trajectory Generation

```python
class BalanceAwarePlanner:
    def __init__(self):
        self.inverted_pendulum = InvertedPendulumModel()
        self.zmp_calculator = ZMPCalculator()
        self.support_polygon = SupportPolygonCalculator()

    def generate_balanced_trajectory(self, footsteps):
        """Generate CoM trajectory that maintains balance for footsteps"""
        com_trajectory = []

        for i, step in enumerate(footsteps):
            # Calculate CoM position for this step
            if i == 0:
                # Starting position
                com_pos = self.calculate_initial_com(step)
            else:
                # Calculate based on previous step and next step
                com_pos = self.calculate_com_transition(
                    footsteps[i-1], step, footsteps[i+1] if i+1 < len(footsteps) else step
                )

            com_trajectory.append(com_pos)

        return com_trajectory

    def calculate_com_transition(self, prev_step, current_step, next_step):
        """Calculate CoM transition between steps"""
        # Use inverted pendulum model to calculate CoM trajectory
        # This ensures ZMP stays within support polygon

        # Calculate support polygon for current double-support phase
        support_poly = self.support_polygon.calculate_double_support(
            prev_step, current_step
        )

        # Calculate CoM position to keep ZMP in safe area
        com_pos = self.inverted_pendulum.calculate_stable_com(
            support_poly, current_step
        )

        return com_pos

    def calculate_initial_com(self, first_step):
        """Calculate initial CoM position"""
        # Position CoM above support polygon
        com_pos = PoseStamped()
        com_pos.pose.position.x = first_step.pose.position.x
        com_pos.pose.position.y = first_step.pose.position.y
        com_pos.pose.position.z = 0.8  # Typical CoM height for humanoid

        return com_pos

    def verify_balance_trajectory(self, com_trajectory, footsteps):
        """Verify that CoM trajectory maintains balance"""
        for i, (com_pos, step) in enumerate(zip(com_trajectory, footsteps)):
            # Calculate ZMP for this CoM position
            zmp = self.zmp_calculator.calculate_zmp(com_pos, step)

            # Get support polygon
            support_poly = self.support_polygon.calculate_support_polygon(step)

            # Check if ZMP is within support polygon
            if not self.is_zmp_stable(zmp, support_poly):
                return False, f"ZMP unstable at step {i}"

        return True, "Trajectory is stable"
```

[@vukobratovic2004; @kajita2001]

## Nav2 Configuration for Humanoid Robots

### Parameter Configuration

```yaml
# Nav2 configuration for humanoid robot
bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    default_bt_xml_filename: "humanoid_navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node
    - nav2_spin_cancel_bt_node
    - nav2_back_up_cancel_bt_node
    - nav2_follow_path_cancel_bt_node

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 10.0  # Lower frequency for humanoid stability
    min_x_velocity_threshold: 0.01
    min_y_velocity_threshold: 0.01
    min_theta_velocity_threshold: 0.01
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["HumanoidFollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.2  # Larger for humanoid step size
      movement_time_allowance: 20.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.3  # Larger for humanoid precision
      yaw_goal_tolerance: 0.3
      stateful: True

    # Humanoid-specific controller
    HumanoidFollowPath:
      plugin: "humanoid_nav2_controllers::HumanoidPathFollower"
      primary_controller: "humanoid_nav2_controllers::BipedalPlanner"

      # Bipedal planner parameters
      humanoid_bipedal_planner:
        plugin: "humanoid_nav2_controllers::BipedalPlanner"
        max_step_length: 0.3
        max_step_width: 0.2
        min_step_length: 0.1
        step_timing: 0.8
        com_height: 0.8
        balance_margin: 0.05
        support_polygon_buffer: 0.05
        zmp_tolerance: 0.02

planner_server:
  ros__parameters:
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5  # Larger tolerance for humanoid navigation
      use_astar: false
      allow_unknown: true

## Behavior Trees for Humanoid Navigation

### Custom Behavior Tree for Bipedal Navigation

```xml
<!-- Custom behavior tree for humanoid navigation -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="navigate_with_recovery">
      <RecoveryNode number_of_retries="2" name="global_plan_with_recovery">
        <ReactiveSequence>
          <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
          <PoseToPoseStamped input_pose="{goal}" output_pose_stamped="{goal_stamped}"/>
        </ReactiveSequence>
        <ClearEntireCostmap name="clear_global_costmap" service_name="global_costmap/clear_entirely_global_costmap"/>
      </RecoveryNode>

      <RecoveryNode number_of_retries="3" name="follow_path_with_recovery">
        <ReactiveSequence>
          <FollowPath path="{path}" controller_id="HumanoidFollowPath"/>
        </ReactiveSequence>
        <Sequence name="backup_and_spin_recovery">
          <BackUp distance="0.3" speed="0.1"/>
          <Spin spin_dist="1.57"/>
        </Sequence>
      </RecoveryNode>
    </PipelineSequence>
  </BehaviorTree>
</root>
```

[@macenski2022; @colomaro2021]

## Humanoid-Specific Recovery Behaviors

### Balance Recovery Actions

```python
from nav2_core.recovery import Recovery
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Duration

class HumanoidRecoverySystem(Recovery):
    def __init__(self):
        super().__init__()
        self.balance_controller = BalanceController()
        self.footstep_generator = FootstepGenerator()

    def on_configure(self, config):
        """Configure recovery system"""
        self.balance_controller.configure(config)
        self.footstep_generator.configure(config)

    def on_cleanup(self):
        """Clean up recovery system"""
        self.balance_controller.cleanup()
        self.footstep_generator.cleanup()

    def on_activate(self):
        """Activate recovery system"""
        self.balance_controller.activate()
        self.footstep_generator.activate()

    def on_deactivate(self):
        """Deactivate recovery system"""
        self.balance_controller.deactivate()
        self.footstep_generator.deactivate()

    def run(self, blackboard):
        """Run recovery behavior"""
        # Check current balance state
        balance_state = self.balance_controller.get_balance_state()

        if balance_state == 'STABLE':
            # Try to continue navigation
            return self.recover_navigation(blackboard)
        elif balance_state == 'UNSTABLE':
            # Execute balance recovery
            return self.execute_balance_recovery()
        elif balance_state == 'FALLING':
            # Execute emergency recovery
            return self.execute_emergency_recovery()
        else:
            # Unknown state, stop and request assistance
            return self.request_assistance()

    def execute_balance_recovery(self):
        """Execute balance recovery behavior"""
        # Generate recovery footsteps to restore balance
        recovery_steps = self.footstep_generator.generate_balance_recovery_steps()

        # Execute recovery steps
        for step in recovery_steps:
            success = self.execute_footstep(step)
            if not success:
                return 'FAILURE'

        # Verify balance restoration
        if self.balance_controller.is_balanced():
            return 'SUCCESS'
        else:
            return 'FAILURE'

    def execute_emergency_recovery(self):
        """Execute emergency recovery (e.g., sitting down)"""
        # For safety, move to stable position
        # This could involve sitting or kneeling
        emergency_pose = self.find_safe_emergency_pose()

        if emergency_pose:
            # Move to safe pose
            self.move_to_emergency_pose(emergency_pose)
            return 'SUCCESS'

        return 'FAILURE'

    def find_safe_emergency_pose(self):
        """Find safe emergency pose"""
        # Look for flat, obstacle-free area
        safe_areas = self.find_safe_areas()

        if safe_areas:
            # Choose closest safe area
            return safe_areas[0]

        return None

    def recover_navigation(self, blackboard):
        """Attempt to recover navigation after balance restoration"""
        # Check if path is still valid
        current_path = blackboard.get('path', None)

        if current_path and self.path_is_traversable(current_path):
            # Continue with current path
            return 'SUCCESS'
        else:
            # Request new path
            return 'FAILURE'  # This triggers path re-computation
```

[@pratt1997; @takenaka2009]

## Integration with Isaac ROS Navigation

### Isaac ROS Nav2 Bridge

```python
class IsaacROSNav2Bridge:
    def __init__(self):
        self.nav2_client = Nav2Client()
        self.isaac_perception = IsaacPerceptionSystem()
        self.humanoid_controller = HumanoidController()

    def initialize_navigation(self):
        """Initialize navigation with Isaac ROS perception"""
        # Wait for Nav2 services
        self.nav2_client.wait_for_server()

        # Initialize Isaac perception
        self.isaac_perception.initialize()

        # Configure humanoid controller
        self.humanoid_controller.configure()

    def navigate_with_isaac_perception(self, goal):
        """Navigate using Isaac ROS perception data"""
        # Send initial navigation goal
        self.nav2_client.send_goal(goal)

        while not self.nav2_client.is_goal_reached():
            # Get perception data from Isaac
            obstacles = self.isaac_perception.get_obstacles()
            landmarks = self.isaac_perception.get_landmarks()
            semantic_map = self.isaac_perception.get_semantic_map()

            # Update costmaps with Isaac perception data
            self.update_costmaps_with_isaac_data(obstacles, landmarks, semantic_map)

            # Get current robot state
            current_pose = self.nav2_client.get_current_pose()
            current_velocity = self.nav2_client.get_current_velocity()

            # Check balance state
            balance_state = self.humanoid_controller.get_balance_state()

            if balance_state != 'STABLE':
                # Pause navigation and recover balance
                self.nav2_client.cancel_goal()
                self.recover_balance()
                self.nav2_client.resume_goal()

            # Sleep to maintain update rate
            time.sleep(0.1)

    def update_costmaps_with_isaac_data(self, obstacles, landmarks, semantic_map):
        """Update Nav2 costmaps with Isaac perception data"""
        # Convert Isaac obstacle data to Nav2 costmap format
        nav2_obstacles = self.convert_obstacles_to_costmap_format(obstacles)

        # Update local costmap
        self.nav2_client.update_local_costmap(nav2_obstacles)

        # Update global costmap if needed
        if self.should_update_global_costmap(semantic_map):
            self.nav2_client.update_global_costmap(semantic_map)

    def convert_obstacles_to_costmap_format(self, obstacles):
        """Convert Isaac obstacle format to Nav2 costmap format"""
        nav2_obstacles = []

        for obstacle in obstacles:
            nav2_obstacle = {}
            nav2_obstacle['x'] = obstacle.position.x
            nav2_obstacle['y'] = obstacle.position.y
            nav2_obstacle['radius'] = obstacle.radius
            nav2_obstacle['type'] = self.map_isaac_obstacle_type(obstacle.type)

            nav2_obstacles.append(nav2_obstacle)

        return nav2_obstacles

    def map_isaac_obstacle_type(self, isaac_type):
        """Map Isaac obstacle type to Nav2 type"""
        type_mapping = {
            'static': 'STATIC',
            'dynamic': 'DYNAMIC',
            'unknown': 'UNKNOWN'
        }
        return type_mapping.get(isaac_type, 'UNKNOWN')
```

[@nvidia2022; @marder2015]

## Performance Optimization

### Computational Efficiency for Humanoid Navigation

```python
class EfficientHumanoidNavigator:
    def __init__(self):
        self.footstep_cache = {}
        self.terrain_cache = {}
        self.balance_cache = {}
        self.max_cache_size = 1000

    def plan_efficiently(self, start, goal):
        """Plan path with computational efficiency"""
        # Check cache first
        cache_key = self.generate_cache_key(start, goal)

        if cache_key in self.footstep_cache:
            cached_plan = self.footstep_cache[cache_key]
            if self.is_plan_valid(cached_plan, start, goal):
                return cached_plan

        # Plan new path
        new_plan = self.compute_new_plan(start, goal)

        # Cache result
        self.cache_plan(cache_key, new_plan)

        return new_plan

    def generate_cache_key(self, start, goal):
        """Generate cache key for start-goal pair"""
        # Discretize positions for caching
        start_disc = self.discretize_pose(start)
        goal_disc = self.discretize_pose(goal)

        return f"{start_disc}_{goal_disc}"

    def discretize_pose(self, pose):
        """Discretize pose for caching"""
        # Round to nearest 10cm for position
        x = round(pose.position.x * 10) / 10
        y = round(pose.position.y * 10) / 10

        # Round to nearest 15 degrees for orientation
        yaw = round(self.quaternion_to_yaw(pose.orientation) / 0.26) * 0.26

        return f"{x:.1f}_{y:.1f}_{yaw:.2f}"

    def compute_new_plan(self, start, goal):
        """Compute new path plan"""
        # Use efficient planning algorithm
        # Consider hierarchical planning: coarse to fine
        coarse_plan = self.plan_coarse_path(start, goal)
        fine_plan = self.refine_to_footsteps(coarse_plan)

        return fine_plan

    def plan_coarse_path(self, start, goal):
        """Plan coarse path at lower resolution"""
        # Use A* or Dijkstra at lower resolution
        # This gives rough path quickly
        pass

    def refine_to_footsteps(self, coarse_path):
        """Refine coarse path to detailed footsteps"""
        # Convert coarse path to detailed footstep plan
        # Apply smoothing and balance constraints
        pass

    def cache_plan(self, key, plan):
        """Cache plan with LRU eviction"""
        if len(self.footstep_cache) >= self.max_cache_size:
            # Remove oldest entry (simplified LRU)
            oldest_key = next(iter(self.footstep_cache))
            del self.footstep_cache[oldest_key]

        self.footstep_cache[key] = plan

    def is_plan_valid(self, plan, start, goal):
        """Check if cached plan is still valid"""
        # Check if start and goal are close enough to cached values
        # Check if environment has changed significantly
        pass
```

[@erdem2004; @li2019]

## Simulation and Testing

### Testing Navigation in Isaac Sim

```python
class NavigationTester:
    def __init__(self, isaac_sim_env):
        self.sim_env = isaac_sim_env
        self.nav2_system = HumanoidNav2Node()
        self.metrics_calculator = MetricsCalculator()

    def test_navigation_performance(self, test_scenarios):
        """Test navigation performance across scenarios"""
        results = []

        for scenario in test_scenarios:
            # Set up scenario in Isaac Sim
            self.setup_scenario(scenario)

            # Run navigation test
            scenario_result = self.run_navigation_test(scenario)
            results.append(scenario_result)

        # Aggregate results
        overall_metrics = self.aggregate_results(results)
        return overall_metrics

    def setup_scenario(self, scenario):
        """Set up test scenario in simulation"""
        # Place humanoid robot
        self.sim_env.place_robot(scenario.robot_start_pose)

        # Place obstacles
        for obstacle in scenario.obstacles:
            self.sim_env.place_obstacle(obstacle)

        # Set goal
        self.sim_env.set_goal(scenario.goal_pose)

    def run_navigation_test(self, scenario):
        """Run single navigation test"""
        start_time = time.time()
        start_pose = scenario.robot_start_pose
        goal_pose = scenario.goal_pose

        # Start navigation
        self.nav2_system.navigate_to_pose(goal_pose)

        # Monitor progress
        success = False
        timeout = False
        path_efficiency = 0
        balance_maintained = True

        while not (success or timeout):
            current_pose = self.sim_env.get_robot_pose()
            current_time = time.time()

            # Check if goal reached
            if self.is_at_goal(current_pose, goal_pose):
                success = True
                break

            # Check for timeout
            if current_time - start_time > scenario.timeout:
                timeout = True
                break

            # Check balance
            if not self.is_balance_maintained():
                balance_maintained = False

            # Sleep to maintain simulation rate
            time.sleep(0.1)

        # Calculate metrics
        end_time = time.time()
        actual_path = self.nav2_system.get_executed_path()
        optimal_path = self.calculate_optimal_path(start_pose, goal_pose)

        metrics = {
            'success': success,
            'time_taken': end_time - start_time,
            'path_efficiency': self.calculate_path_efficiency(actual_path, optimal_path),
            'balance_maintained': balance_maintained,
            'collisions': self.count_collisions(),
            'recovery_actions': self.count_recovery_actions()
        }

        return metrics

    def calculate_path_efficiency(self, actual_path, optimal_path):
        """Calculate path efficiency metric"""
        actual_length = self.calculate_path_length(actual_path)
        optimal_length = self.calculate_path_length(optimal_path)

        # Efficiency = optimal_length / actual_length (lower is better, so we might want 1/efficiency)
        if actual_length > 0:
            return optimal_length / actual_length
        else:
            return 0.0
```

[@zhu2018; @wijmans2019]

## Research Tasks

1. Investigate the impact of different footstep planning algorithms on navigation efficiency
2. Explore machine learning approaches for adaptive humanoid navigation
3. Analyze the trade-offs between navigation speed and balance stability

## Evidence Requirements

Students must demonstrate understanding by:
- Configuring Nav2 for a humanoid robot simulation
- Implementing a simple footstep planning algorithm
- Testing navigation performance with balance constraints

## References

- Macenski, S., et al. (2022). Nav2: A flexible and performant navigation toolkit. *arXiv preprint arXiv:2209.05651*.
- Marder-Eppstein, E., et al. (2015). The office marathon: Robust navigation in an indoor office environment. *2015 IEEE International Conference on Robotics and Automation (ICRA)*, 3040-3047.
- Khatib, O. (1986). Real-time obstacle avoidance for manipulators and mobile robots. *The International Journal of Robotics Research*, 5(1), 90-98.
- Kajita, S., et al. (2001). The 3D linear inverted pendulum mode: A simple modeling for a biped walking pattern generation. *Proceedings 2001 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 239-246.
- Feng, S., et al. (2013). Online robust optimization of biped walking based on previous experiments. *2013 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 1280-1286.
- Wobken, J., et al. (2019). Humanoid robot path planning in dynamic environments. *Journal of Intelligent & Robotic Systems*, 93(1-2), 157-172.
- Kunze, L., et al. (2015). A comparison of path planning algorithms for humanoid robots. *Robotics and Autonomous Systems*, 69, 78-90.
- Hur, P., et al. (2019). Walking pattern generator for humanoid robots. *IEEE Transactions on Robotics*, 35(3), 712-725.
- Vukobratović, M., & Borovac, B. (2004). Zero-moment point—thirty five years of its life. *International Journal of Humanoid Robotics*, 1(01), 157-173.
- Colomaro, A., et al. (2021). Behavior trees in robotics and AI: An introduction. *Chapman and Hall/CRC*.
- Pratt, J., et al. (1997). Intuitive control of a planar bipedal walking robot. *Proceedings of the 1997 IEEE International Conference on Robotics and Automation*, 2872-2878.
- Takenaka, T., et al. (2009). Real time motion generation and control for biped robot. *2009 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 1031-1038.
- NVIDIA Corporation. (2022). Isaac ROS Navigation Documentation. NVIDIA Developer Documentation.
- Erdem, E. R., & Erdem, A. (2004). The use of hierarchical structuring in the A* algorithm for solving the path planning problem. *Knowledge-Based Systems*, 17(8), 395-401.
- Li, M., et al. (2019). High-precision, fast-dynamics perception for autonomous systems. *The International Journal of Robotics Research*, 38(2-3), 145-165.
- Zhu, Y., et al. (2018). Vision-based navigation with language-based assistance via imitation learning with indirect intervention. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2459-2468.
- Wijmans, E., et al. (2019). Attention for scene segmentation in embodied agents. *arXiv preprint arXiv:1903.00277*.

## Practical Exercises

1. Configure Nav2 parameters specifically for humanoid robot navigation
   - Install and set up the Navigation2 stack for ROS 2
   - Modify base_footprint and footprint parameters for humanoid dimensions
   - Adjust costmap resolution and inflation parameters for humanoid scale
   - Configure velocity limits appropriate for bipedal locomotion
   - Test navigation with default parameters and document baseline performance

2. Implement a simple footstep planner that considers balance constraints
   - Design a footstep planner based on grid-based search algorithms
   - Implement support polygon calculations for bipedal stability
   - Add Zero Moment Point (ZMP) constraints to footstep selection
   - Test the planner with various terrain configurations
   - Validate that generated footsteps maintain humanoid balance

3. Test navigation performance in Isaac Sim with various obstacle configurations
   - Create simulation environments with different obstacle types and densities
   - Test navigation with static and dynamic obstacles
   - Evaluate path optimality and navigation success rates
   - Measure time to goal and path efficiency metrics
   - Document failure cases and analyze root causes

4. Analyze the relationship between step size and navigation efficiency
   - Design experiments with varying maximum step lengths
   - Measure navigation time and path optimality for different step sizes
   - Evaluate balance stability for different step configurations
   - Determine optimal step parameters for various scenarios
   - Document trade-offs between efficiency and stability

5. Develop and test a humanoid-specific recovery behavior for balance restoration
   - Implement recovery behaviors for different types of balance loss
   - Design stepping strategies to restore center of mass within support polygon
   - Create emergency behaviors for critical balance situations
   - Test recovery behaviors in simulation with various disturbances
   - Validate that recovery behaviors improve overall navigation robustness

6. Create a custom path planner that generates dynamically stable trajectories for bipedal locomotion
   - Implement a path planner that generates center of mass (CoM) trajectories
   - Integrate inverted pendulum model for stable locomotion planning
   - Add constraints for humanoid kinematic limits
   - Generate trajectories that maintain ZMP within support polygon
   - Validate dynamic stability of generated trajectories

7. Implement a hierarchical navigation system that plans at both global and local levels for humanoid robots
   - Design global planning considering humanoid-specific constraints
   - Implement local planning for real-time obstacle avoidance
   - Create coordination mechanism between global and local planners
   - Test system performance in complex, dynamic environments
   - Evaluate computational efficiency and navigation success rates

8. Evaluate the computational efficiency of different path planning algorithms for humanoid navigation
   - Implement multiple planning algorithms (A*, D*, RRT variants)
   - Measure computational requirements for each algorithm
   - Compare path quality and planning time trade-offs
   - Test scalability with increasing environment complexity
   - Document recommendations for different application scenarios