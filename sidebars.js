// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: ROS 2 (Robotic Nervous System)',
      items: [
        'module1/ROS-2-Robotic-Nervous-System/index',
        'module1/ROS-2-Robotic-Nervous-System/chapter1-nodes-topics-services',
        'module1/ROS-2-Robotic-Nervous-System/chapter2-rclpy-integration',
        'module1/ROS-2-Robotic-Nervous-System/chapter3-urdf-humanoids',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin (Gazebo + Unity)',
      items: [
        'module2/Digital-Twin-Gazebo-Unity/index',
        'module2/Digital-Twin-Gazebo-Unity/chapter1-gazebo-unity',
        'module2/Digital-Twin-Gazebo-Unity/chapter2-physics-simulation',
        'module2/Digital-Twin-Gazebo-Unity/chapter3-sensors-simulation',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac (AI-Robot Brain)',
      items: [
        'module3/NVIDIA-Isaac-AI-Robot-Brain/index',
        'module3/NVIDIA-Isaac-AI-Robot-Brain/chapter1-isaac-sim',
        'module3/NVIDIA-Isaac-AI-Robot-Brain/chapter2-vslam-navigation',
        'module3/NVIDIA-Isaac-AI-Robot-Brain/chapter3-nav2-bipedal',
        'module3/NVIDIA-Isaac-AI-Robot-Brain/chapter4-sim-to-real',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA) + Capstone',
      items: [
        'module4/Vision-Language-Action-VLA/index',
        'module4/Vision-Language-Action-VLA/chapter1-whisper-commands',
        'module4/Vision-Language-Action-VLA/chapter2-llm-planning',
        'module4/Vision-Language-Action-VLA/chapter3-voice-to-action',
        'module4/Vision-Language-Action-VLA/chapter4-capstone-plagiarism-check',
      ],
    },
    {
      type: 'category',
      label: 'References',
      items: [
        'references/citations',
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/getting-started',
      ],
    },
  ],
};

module.exports = sidebars;