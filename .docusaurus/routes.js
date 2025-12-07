import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/humanoid-robotics/docs',
    component: ComponentCreator('/humanoid-robotics/docs', 'e76'),
    routes: [
      {
        path: '/humanoid-robotics/docs',
        component: ComponentCreator('/humanoid-robotics/docs', '040'),
        routes: [
          {
            path: '/humanoid-robotics/docs',
            component: ComponentCreator('/humanoid-robotics/docs', '591'),
            routes: [
              {
                path: '/humanoid-robotics/docs/',
                component: ComponentCreator('/humanoid-robotics/docs/', '5a9'),
                exact: true
              },
              {
                path: '/humanoid-robotics/docs/intro',
                component: ComponentCreator('/humanoid-robotics/docs/intro', '38e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module1/ROS-2-Robotic-Nervous-System/',
                component: ComponentCreator('/humanoid-robotics/docs/module1/ROS-2-Robotic-Nervous-System/', 'a32'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module1/ROS-2-Robotic-Nervous-System/chapter1-nodes-topics-services',
                component: ComponentCreator('/humanoid-robotics/docs/module1/ROS-2-Robotic-Nervous-System/chapter1-nodes-topics-services', '599'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module1/ROS-2-Robotic-Nervous-System/chapter2-rclpy-integration',
                component: ComponentCreator('/humanoid-robotics/docs/module1/ROS-2-Robotic-Nervous-System/chapter2-rclpy-integration', 'c16'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module1/ROS-2-Robotic-Nervous-System/chapter3-urdf-humanoids',
                component: ComponentCreator('/humanoid-robotics/docs/module1/ROS-2-Robotic-Nervous-System/chapter3-urdf-humanoids', 'a3e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module1/ROS-2-Robotic-Nervous-System/citations-validation',
                component: ComponentCreator('/humanoid-robotics/docs/module1/ROS-2-Robotic-Nervous-System/citations-validation', '64a'),
                exact: true
              },
              {
                path: '/humanoid-robotics/docs/module1/ROS-2-Robotic-Nervous-System/plagiarism-check',
                component: ComponentCreator('/humanoid-robotics/docs/module1/ROS-2-Robotic-Nervous-System/plagiarism-check', '3dd'),
                exact: true
              },
              {
                path: '/humanoid-robotics/docs/module2/Digital-Twin-Gazebo-Unity/',
                component: ComponentCreator('/humanoid-robotics/docs/module2/Digital-Twin-Gazebo-Unity/', 'a4e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module2/Digital-Twin-Gazebo-Unity/chapter1-gazebo-unity',
                component: ComponentCreator('/humanoid-robotics/docs/module2/Digital-Twin-Gazebo-Unity/chapter1-gazebo-unity', 'a96'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module2/Digital-Twin-Gazebo-Unity/chapter2-physics-simulation',
                component: ComponentCreator('/humanoid-robotics/docs/module2/Digital-Twin-Gazebo-Unity/chapter2-physics-simulation', '729'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module2/Digital-Twin-Gazebo-Unity/chapter3-sensors-simulation',
                component: ComponentCreator('/humanoid-robotics/docs/module2/Digital-Twin-Gazebo-Unity/chapter3-sensors-simulation', 'a45'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module2/Digital-Twin-Gazebo-Unity/citations-validation',
                component: ComponentCreator('/humanoid-robotics/docs/module2/Digital-Twin-Gazebo-Unity/citations-validation', 'a68'),
                exact: true
              },
              {
                path: '/humanoid-robotics/docs/module2/Digital-Twin-Gazebo-Unity/plagiarism-check',
                component: ComponentCreator('/humanoid-robotics/docs/module2/Digital-Twin-Gazebo-Unity/plagiarism-check', 'bbe'),
                exact: true
              },
              {
                path: '/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/',
                component: ComponentCreator('/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/', '33b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter1-isaac-sim',
                component: ComponentCreator('/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter1-isaac-sim', '65a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter2-vslam-navigation',
                component: ComponentCreator('/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter2-vslam-navigation', '642'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter3-nav2-bipedal',
                component: ComponentCreator('/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter3-nav2-bipedal', 'a89'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter4-sim-to-real',
                component: ComponentCreator('/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter4-sim-to-real', 'd7d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/citations-validation',
                component: ComponentCreator('/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/citations-validation', '121'),
                exact: true
              },
              {
                path: '/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/plagiarism-check',
                component: ComponentCreator('/humanoid-robotics/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/plagiarism-check', 'e92'),
                exact: true
              },
              {
                path: '/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/',
                component: ComponentCreator('/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/', 'ef0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/chapter1-whisper-commands',
                component: ComponentCreator('/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/chapter1-whisper-commands', 'b24'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/chapter2-llm-planning',
                component: ComponentCreator('/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/chapter2-llm-planning', 'cf3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/chapter3-voice-to-action',
                component: ComponentCreator('/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/chapter3-voice-to-action', 'c44'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/chapter4-capstone-plagiarism-check',
                component: ComponentCreator('/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/chapter4-capstone-plagiarism-check', '0bf'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/citations-validation',
                component: ComponentCreator('/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/citations-validation', '29e'),
                exact: true
              },
              {
                path: '/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/plagiarism-check',
                component: ComponentCreator('/humanoid-robotics/docs/module4/Vision-Language-Action-VLA/plagiarism-check', '6d7'),
                exact: true
              },
              {
                path: '/humanoid-robotics/docs/references/citations',
                component: ComponentCreator('/humanoid-robotics/docs/references/citations', '425'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/humanoid-robotics/docs/tutorials/getting-started',
                component: ComponentCreator('/humanoid-robotics/docs/tutorials/getting-started', 'af2'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/humanoid-robotics/',
    component: ComponentCreator('/humanoid-robotics/', 'ea3'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
