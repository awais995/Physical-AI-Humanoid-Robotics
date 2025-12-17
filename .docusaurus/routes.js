import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/chatbot',
    component: ComponentCreator('/chatbot', '522'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '761'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', 'c20'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', 'e06'),
            routes: [
              {
                path: '/docs/',
                component: ComponentCreator('/docs/', '0ee'),
                exact: true
              },
              {
                path: '/docs/chatbot-configuration',
                component: ComponentCreator('/docs/chatbot-configuration', '85f'),
                exact: true
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '61d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module1/ROS-2-Robotic-Nervous-System/',
                component: ComponentCreator('/docs/module1/ROS-2-Robotic-Nervous-System/', '92c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module1/ROS-2-Robotic-Nervous-System/chapter1-nodes-topics-services',
                component: ComponentCreator('/docs/module1/ROS-2-Robotic-Nervous-System/chapter1-nodes-topics-services', 'aa2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module1/ROS-2-Robotic-Nervous-System/chapter2-rclpy-integration',
                component: ComponentCreator('/docs/module1/ROS-2-Robotic-Nervous-System/chapter2-rclpy-integration', '1d3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module1/ROS-2-Robotic-Nervous-System/chapter3-urdf-humanoids',
                component: ComponentCreator('/docs/module1/ROS-2-Robotic-Nervous-System/chapter3-urdf-humanoids', 'a0a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module1/ROS-2-Robotic-Nervous-System/citations-validation',
                component: ComponentCreator('/docs/module1/ROS-2-Robotic-Nervous-System/citations-validation', 'eac'),
                exact: true
              },
              {
                path: '/docs/module1/ROS-2-Robotic-Nervous-System/plagiarism-check',
                component: ComponentCreator('/docs/module1/ROS-2-Robotic-Nervous-System/plagiarism-check', 'ecb'),
                exact: true
              },
              {
                path: '/docs/module2/Digital-Twin-Gazebo-Unity/',
                component: ComponentCreator('/docs/module2/Digital-Twin-Gazebo-Unity/', '98e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module2/Digital-Twin-Gazebo-Unity/chapter1-gazebo-unity',
                component: ComponentCreator('/docs/module2/Digital-Twin-Gazebo-Unity/chapter1-gazebo-unity', 'cd7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module2/Digital-Twin-Gazebo-Unity/chapter2-physics-simulation',
                component: ComponentCreator('/docs/module2/Digital-Twin-Gazebo-Unity/chapter2-physics-simulation', '90d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module2/Digital-Twin-Gazebo-Unity/chapter3-sensors-simulation',
                component: ComponentCreator('/docs/module2/Digital-Twin-Gazebo-Unity/chapter3-sensors-simulation', 'c81'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module2/Digital-Twin-Gazebo-Unity/citations-validation',
                component: ComponentCreator('/docs/module2/Digital-Twin-Gazebo-Unity/citations-validation', '7c5'),
                exact: true
              },
              {
                path: '/docs/module2/Digital-Twin-Gazebo-Unity/plagiarism-check',
                component: ComponentCreator('/docs/module2/Digital-Twin-Gazebo-Unity/plagiarism-check', '67d'),
                exact: true
              },
              {
                path: '/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/',
                component: ComponentCreator('/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/', 'f14'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter1-isaac-sim',
                component: ComponentCreator('/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter1-isaac-sim', 'bb1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter2-vslam-navigation',
                component: ComponentCreator('/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter2-vslam-navigation', '240'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter3-nav2-bipedal',
                component: ComponentCreator('/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter3-nav2-bipedal', '0f6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter4-sim-to-real',
                component: ComponentCreator('/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter4-sim-to-real', '0af'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/citations-validation',
                component: ComponentCreator('/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/citations-validation', 'aa9'),
                exact: true
              },
              {
                path: '/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/plagiarism-check',
                component: ComponentCreator('/docs/module3/NVIDIA-Isaac-AI-Robot-Brain/plagiarism-check', '8fc'),
                exact: true
              },
              {
                path: '/docs/module4/Vision-Language-Action-VLA/',
                component: ComponentCreator('/docs/module4/Vision-Language-Action-VLA/', '030'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module4/Vision-Language-Action-VLA/chapter1-whisper-commands',
                component: ComponentCreator('/docs/module4/Vision-Language-Action-VLA/chapter1-whisper-commands', '66a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module4/Vision-Language-Action-VLA/chapter2-llm-planning',
                component: ComponentCreator('/docs/module4/Vision-Language-Action-VLA/chapter2-llm-planning', 'a2a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module4/Vision-Language-Action-VLA/chapter3-voice-to-action',
                component: ComponentCreator('/docs/module4/Vision-Language-Action-VLA/chapter3-voice-to-action', '136'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module4/Vision-Language-Action-VLA/chapter4-capstone-plagiarism-check',
                component: ComponentCreator('/docs/module4/Vision-Language-Action-VLA/chapter4-capstone-plagiarism-check', '78d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module4/Vision-Language-Action-VLA/citations-validation',
                component: ComponentCreator('/docs/module4/Vision-Language-Action-VLA/citations-validation', 'e39'),
                exact: true
              },
              {
                path: '/docs/module4/Vision-Language-Action-VLA/plagiarism-check',
                component: ComponentCreator('/docs/module4/Vision-Language-Action-VLA/plagiarism-check', '368'),
                exact: true
              },
              {
                path: '/docs/references/citations',
                component: ComponentCreator('/docs/references/citations', 'e93'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorials/getting-started',
                component: ComponentCreator('/docs/tutorials/getting-started', '079'),
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
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
