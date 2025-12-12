// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Modules',
      items: [
        {
          type: 'category',
          label: 'Module 1: The Robotic Nervous System (ROS 2)',
          items: [
            'modules/ros-nervous-system/index',
            'modules/ros-nervous-system/architecture',
            'modules/ros-nervous-system/nodes-topics-services',
            'modules/ros-nervous-system/urdf-modeling',
            'modules/ros-nervous-system/hands-on-exercises'
          ],
        },
        {
          type: 'category',
          label: 'Module 2: The Digital Twin (Gazebo & Unity)',
          items: [
            'modules/digital-twin/index',
            'modules/digital-twin/gazebo-simulation',
            'modules/digital-twin/unity-integration',
            'modules/digital-twin/sensor-simulation',
            'modules/digital-twin/physics-modeling',
            'modules/digital-twin/hands-on-exercises'
          ],
        },
        {
          type: 'category',
          label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
          items: [
            'modules/ai-brain/index',
            'modules/ai-brain/vslam-navigation',
            'modules/ai-brain/nav2-path-planning',
            'modules/ai-brain/reinforcement-learning',
            'modules/ai-brain/perception-pipelines',
            'modules/ai-brain/hands-on-exercises'
          ],
        },
        {
          type: 'category',
          label: 'Module 4: Vision-Language-Action (VLA) & Capstone',
          items: [
            'modules/vla-capstone/index',
            'modules/vla-capstone/voice-to-action',
            'modules/vla-capstone/cognitive-planning',
            'modules/vla-capstone/multi-modal-perception',
            'modules/vla-capstone/capstone-project',
            'modules/vla-capstone/hands-on-exercises'
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/getting-started',
        'tutorials/hardware-setup',
        'tutorials/simulation-environments'
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/ros2-cheat-sheet',
        'reference/urdf-specification',
        'reference/troubleshooting'
      ],
    }
  ],
};

export default sidebars;