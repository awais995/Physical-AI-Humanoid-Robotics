import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'ROS 2 - Robotic Nervous System',
    description: (
      <>
        Learn to build the nervous system of robots with ROS 2, including nodes, topics, services, and URDF for humanoid robot modeling.
      </>
    ),
  },
  {
    title: 'Digital Twin Technologies',
    description: (
      <>
        Create accurate virtual representations of physical robots using Gazebo and Unity for simulation and testing.
      </>
    ),
  },
  {
    title: 'NVIDIA Isaac - AI Robot Brain',
    description: (
      <>
        Understand how to use NVIDIA Isaac for AI-powered robotics, including Isaac Sim for synthetic data generation and Nav2 for navigation.
      </>
    ),
  },
  {
    title: 'Vision-Language-Action Systems',
    description: (
      <>
        Implement voice-activated robotics systems using Whisper for voice commands, LLMs for cognitive planning, and voice-to-action pipelines.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--3')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}