import asyncio
import os
from dotenv import load_dotenv
from src.services.content_service import content_service
from src.config.settings import settings
import re

# Load environment variables
load_dotenv('backend/.env')

def split_by_headers(content: str, source_url: str, source_title: str, source_section: str):
    """
    Split content by headers to create more meaningful chunks
    """
    chunks = []

    # Split content by headers (## and ###)
    header_pattern = r'(^|\n)(##{1,2}\s+.*?)(?=\n##{1,2}\s+|\n$)'
    parts = re.split(header_pattern, content, flags=re.MULTILINE)

    current_module = source_section
    current_title = source_title

    # Process parts - odd indices are headers, even indices are content
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1].strip():  # This is a header
            header = parts[i + 1].strip()
            content_part = parts[i + 2] if i + 2 < len(parts) else ""

            # Clean up the header
            header = header.lstrip('#').strip()

            # Create chunk with the header as part of the content
            chunk_content = f"## {header}\n{content_part}".strip()

            if len(chunk_content) > 50:  # Only create chunks that are substantial
                chunk = {
                    "content": chunk_content,
                    "source_url": source_url,
                    "source_title": f"{current_title} - {header}",
                    "source_section": current_module
                }
                chunks.append(chunk)

            i += 3  # Move past header and content
        else:
            i += 1

    # If no headers were found, create one chunk with the entire content
    if not chunks and content.strip():
        chunks.append({
            "content": content.strip(),
            "source_url": source_url,
            "source_title": source_title,
            "source_section": source_section
        })

    return chunks

def split_content_by_sections(content: str, source_url: str, source_title: str, source_section: str):
    """
    Split content by major sections (## headers) and subsections (### headers)
    """
    chunks = []

    # Split by main sections (##)
    sections = re.split(r'\n## ', content)

    main_title = sections[0] if sections else ""  # Content before first section

    for i, section in enumerate(sections):
        if i == 0 and section.strip():  # Content before first section
            # Add to main content if it's introductory material
            continue
        elif section.strip():
            # Add '## ' back to each section except the first part
            section_content = f"## {section}"

            # Further split by subsections (###)
            subsections = re.split(r'\n### ', section_content)

            for j, subsection in enumerate(subsections):
                if j == 0:  # Main section content
                    if subsection.strip():
                        # Extract the main section title from the first line
                        lines = subsection.split('\n', 1)
                        if lines:
                            section_title = lines[0].replace('##', '').strip()
                            section_body = lines[1] if len(lines) > 1 else ""
                            chunk_content = f"## {section_title}\n{section_body}".strip()

                            if len(chunk_content) > 50:
                                chunks.append({
                                    "content": chunk_content,
                                    "source_url": source_url,
                                    "source_title": f"{source_title} - {section_title}",
                                    "source_section": f"{source_section} - {section_title}"
                                })
                else:  # Subsection content
                    if subsection.strip():
                        # Extract the subsection title from the first line
                        lines = subsection.split('\n', 1)
                        if lines:
                            subsection_title = lines[0].replace('###', '').strip()
                            subsection_body = lines[1] if len(lines) > 1 else ""
                            chunk_content = f"### {subsection_title}\n{subsection_body}".strip()

                            if len(chunk_content) > 50:
                                chunks.append({
                                    "content": chunk_content,
                                    "source_url": source_url,
                                    "source_title": f"{source_title} - {subsection_title}",
                                    "source_section": f"{source_section} - {subsection_title}"
                                })

    return chunks

async def ingest_book_content_by_modules():
    """
    Script to ingest book content by modules and chapters for better retrieval
    """
    print("Initializing content service...")

    # Initialize the collection
    await content_service.initialize_collection()
    print(f"Collection '{settings.qdrant_collection_name}' is ready.")

    # Define comprehensive content from the Physical AI & Humanoid Robotics book
    book_modules = [
        {
            "module": "Module 1",
            "title": "Physical AI & Humanoid Robotics - Introduction",
            "section": "Module 1 - Introduction",
            "url": "https://physical-ai-humanoid-robotics-nine-red.vercel.app/intro",
            "content": """# Module 1: Physical AI & Humanoid Robotics - Introduction
Also known as Chapter 1: Physical AI & Humanoid Robotics - Introduction

This book explores the fascinating world of physical AI and humanoid robotics. Physical AI represents a paradigm shift from traditional AI that operates on abstract data to AI that understands and interacts with the physical world through robotic systems.

## What is Physical AI?
Physical AI combines machine learning, robotics, and physics to create systems that understand the physical world. Unlike traditional AI that processes text, images, or audio, physical AI learns from direct interaction with the environment.

Key concepts include:
- Embodied cognition: Intelligence emerges from the interaction between body and environment
- Sensorimotor learning: Learning through sensing and moving
- Physics-aware neural networks: Neural networks that understand physical laws
- Real-world grounding: Connecting abstract concepts to physical reality

## The Evolution of Robotics
The field of robotics has evolved significantly over the past decades. Early robots were primarily industrial machines designed for repetitive tasks in controlled environments. Modern robots, especially humanoid robots, are expected to operate in unstructured environments and interact safely with humans.

This evolution has necessitated the development of more sophisticated AI systems that can:
- Understand and navigate complex physical environments
- Manipulate objects with dexterity comparable to humans
- Learn from physical interaction and experience
- Adapt to novel situations in real-time"""
        },
        {
            "module": "Module 2",
            "title": "ROS 2 - The Robotic Nervous System",
            "section": "Module 2 - ROS 2",
            "url": "https://physical-ai-humanoid-robotics-nine-red.vercel.app/ros2",
            "content": """# Module 2: ROS 2 - The Robotic Nervous System
Also known as Chapter 2: ROS 2 - The Robotic Nervous System

Robot Operating System 2 (ROS 2) serves as the communication backbone for robotic applications. It provides a framework for writing robotic software with message passing, package management, and distributed processing capabilities.

## Architecture of ROS 2
ROS 2 features a modern, flexible architecture based on Data Distribution Service (DDS) for communication. This provides:

- Improved security with DDS-based communication
- Better real-time support with real-time scheduling capabilities
- Cross-platform compatibility across Linux, Windows, and macOS
- Lifecycle management for nodes allowing graceful startup and shutdown
- Distributed system support enabling multi-robot systems

## Core Concepts
- Nodes: Processes that perform computation
- Topics: Named buses over which nodes exchange messages
- Services: Synchronous request/response communication
- Actions: Asynchronous goal-oriented communication
- Parameters: Configuration values that can be changed at runtime

## ROS 2 vs ROS 1
ROS 2 addresses many limitations of ROS 1:
- Production readiness with commercial support
- Real-time systems support
- Multi-robot system capabilities
- Improved security and authentication
- Cross-compilation support for embedded systems"""
        },
        {
            "module": "Module 3",
            "title": "NVIDIA Isaac - The AI Robot Brain",
            "section": "Module 3 - NVIDIA Isaac",
            "url": "https://physical-ai-humanoid-robotics-nine-red.vercel.app/isaac",
            "content": """# Module 3: NVIDIA Isaac - The AI Robot Brain
Also known as Chapter 3: NVIDIA Isaac - The AI Robot Brain

NVIDIA Isaac provides a comprehensive platform for developing AI-powered robots. It includes simulation environments, AI frameworks, and hardware acceleration for robotics applications.

## Isaac Sim
Isaac Sim is a robotics simulation application and framework built on NVIDIA Omniverse. It provides:
- Physically accurate simulation environments
- Ground-truth data generation for training AI models
- Synthetic data generation capabilities
- Integration with popular robotics frameworks like ROS/ROS2
- Reproducible testing scenarios

## Isaac ROS
Isaac ROS provides GPU-accelerated perception and navigation capabilities:
- Hardware-accelerated computer vision algorithms
- SLAM (Simultaneous Localization and Mapping) implementations
- Sensor processing pipelines optimized for NVIDIA GPUs
- Real-time perception capabilities for robotics applications

## Isaac Lab
Isaac Lab is a simulation framework for robot learning:
- Reinforcement learning environments for robotics
- Domain randomization capabilities
- Physics simulation for robot learning
- Integration with popular RL libraries like RLlib and Stable-Baselines3

## Hardware Platform
NVIDIA's robotics platform includes:
- Jetson platform for edge AI in robotics
- GPU-accelerated inference capabilities
- Optimized AI model deployment
- Real-time performance for robotic applications"""
        },
        {
            "module": "Module 4",
            "title": "Digital Twin Technology in Robotics",
            "section": "Module 4 - Digital Twin",
            "url": "https://physical-ai-humanoid-robotics-nine-red.vercel.app/digital-twin",
            "content": """# Module 4: Digital Twin Technology in Robotics
Also known as Chapter 4: Digital Twin Technology in Robotics

Digital twin technology creates virtual replicas of physical robots and environments, enabling advanced simulation, testing, and optimization capabilities.

## What is a Digital Twin?
A digital twin is a virtual representation of a physical entity or system. In robotics, this includes:
- Accurate 3D models of robots
- Physics-based simulation of robot dynamics
- Environmental modeling
- Sensor simulation
- Real-time synchronization with physical systems

## Benefits of Digital Twins in Robotics
- Risk-free testing of control algorithms
- Accelerated development cycles
- Predictive maintenance capabilities
- Optimization of robot performance
- Training of AI models in safe virtual environments

## Gazebo and Unity Integration
Popular simulation platforms for robotics digital twins include:
- Gazebo: Physics-based simulation with realistic dynamics
- Unity: High-fidelity visual rendering and game engine capabilities
- Integration with ROS/ROS2 for seamless simulation-to-reality transfer"""
        },
        {
            "module": "Module 5",
            "title": "Vision-Language-Action (VLA) Models",
            "section": "Module 5 - VLA Models",
            "url": "https://physical-ai-humanoid-robotics-nine-red.vercel.app/vla",
            "content": """# Module 5: Vision-Language-Action (VLA) Models
Also known as Chapter 5: Vision-Language-Action (VLA) Models

Vision-Language-Action (VLA) models represent the next generation of embodied AI systems that can perceive, understand, and act in the physical world based on natural language commands.

## Understanding VLA Models
VLA models combine three key modalities:
- Vision: Understanding visual input from cameras and sensors
- Language: Processing natural language commands and instructions
- Action: Executing motor commands to interact with the physical world

## Key Components
- Multi-modal neural networks that process vision and language jointly
- Action generation modules that translate understanding into robot commands
- Attention mechanisms that focus on relevant parts of the environment
- Memory systems that maintain context across interactions

## Applications
- Human-robot interaction with natural language commands
- Instruction following in complex environments
- Task planning and execution
- Adaptive behavior based on environmental feedback

## Challenges and Solutions
- Grounding language in physical reality
- Handling ambiguous or underspecified commands
- Robustness to environmental variations
- Safety and reliability in physical interactions"""
        },
        {
            "module": "Module 6",
            "title": "Sim-to-Real Transfer",
            "section": "Module 6 - Sim-to-Real",
            "url": "https://physical-ai-humanoid-robotics-nine-red.vercel.app/sim-to-real",
            "content": """# Module 6: Sim-to-Real Transfer
Also known as Chapter 6: Sim-to-Real Transfer

Sim-to-real transfer is the process of developing and training robotic systems in simulation environments and then successfully deploying them on physical robots.

## The Sim-to-Real Challenge
The primary challenge in sim-to-real transfer is the "reality gap" - differences between simulated and real environments that can cause policies learned in simulation to fail when deployed on real robots.

## Approaches to Address the Reality Gap
- Domain randomization: Randomizing simulation parameters to make policies robust
- Domain adaptation: Adapting simulation to better match reality
- System identification: Modeling the differences between simulation and reality
- Progressive training: Gradually introducing real-world elements

## Techniques for Successful Transfer
- Robust control methods that are insensitive to modeling errors
- Adaptive control that can adjust to real-world conditions
- Learning from demonstration to initialize policies
- Online learning and fine-tuning on the real system

## Best Practices
- Validate simulation fidelity with real-world data
- Use multiple simulation environments to improve robustness
- Implement safety measures for real-world testing
- Monitor and adapt policies during deployment"""
        },
        {
            "module": "Module 7",
            "title": "Voice Command Processing in Robotics",
            "section": "Module 7 - Voice Commands",
            "url": "https://physical-ai-humanoid-robotics-nine-red.vercel.app/voice",
            "content": """# Module 7: Voice Command Processing in Robotics
Also known as Chapter 7: Voice Command Processing in Robotics

Voice command processing enables robots to understand and respond to natural language instructions from users, making human-robot interaction more intuitive.

## Speech Recognition in Robotics
- Integration with speech-to-text systems like Whisper
- Noise reduction for robotic environments
- Speaker identification and diarization
- Real-time processing capabilities

## Natural Language Understanding
- Intent recognition from voice commands
- Entity extraction for specific objects or locations
- Context-aware interpretation
- Handling ambiguous or incomplete commands

## Implementation with Whisper
OpenAI's Whisper model provides robust speech recognition capabilities:
- Multi-language support
- Robustness to background noise
- Real-time and offline processing options
- Integration with robotic control systems

## Voice-to-Action Pipeline
1. Audio capture from robot's microphones
2. Speech recognition using Whisper or similar models
3. Natural language processing to extract intent and entities
4. Action planning and execution
5. Feedback to user through speech or other modalities"""
        }
    ]

    print(f"Starting ingestion of {len(book_modules)} modules from your book...")
    print(f"Source: https://physical-ai-humanoid-robotics-nine-red.vercel.app/")

    total_chunks = 0
    for i, module_data in enumerate(book_modules):
        print(f"Processing module {i+1}/{len(book_modules)}: {module_data['title']}")

        # Split the content by sections for better chunking
        chunks = split_content_by_sections(
            module_data["content"],
            module_data["url"],
            module_data["title"],
            module_data["section"]
        )

        print(f"  - Split into {len(chunks)} smaller chunks")

        for j, chunk in enumerate(chunks):
            try:
                chunks_ingested = await content_service.ingest_document(
                    content=chunk["content"],
                    source_url=chunk["source_url"],
                    source_title=chunk["source_title"],
                    source_section=chunk["source_section"]
                )
                total_chunks += chunks_ingested
            except Exception as e:
                print(f"  - Error ingesting chunk: {e}")
                import traceback
                traceback.print_exc()

    # Print final stats
    stats = await content_service.get_content_stats()
    print(f"\nIngestion complete!")
    print(f"Total passages in vector database: {stats.get('total_passages', 0)}")
    print(f"Total chunks ingested: {total_chunks}")

    return stats

async def main():
    """
    Main function to run the ingestion process
    """
    print("Physical AI & Humanoid Robotics Book - Content Ingestion by Modules")
    print("=" * 70)

    # Check if required environment variables are set
    if not settings.cohere_api_key:
        print("ERROR: COHERE_API_KEY is not set in .env file")
        print("Please add your Cohere API key to the .env file")
        return

    if not settings.qdrant_url:
        print("ERROR: QDRANT_URL is not set in .env file")
        print("Please add your Qdrant Cloud URL to the .env file")
        return

    if not settings.qdrant_api_key:
        print("ERROR: QDRANT_API_KEY is not set in .env file")
        print("Please add your Qdrant API key to the .env file")
        return

    print("Environment variables are properly configured.")
    print()

    # Run the ingestion process
    stats = await ingest_book_content_by_modules()

    print("\n" + "=" * 70)
    print("Book content ingestion by modules finished successfully!")
    print("The RAG system is now loaded with properly chunked book content.")
    print("It can answer questions about Physical AI, Humanoid Robotics, ROS 2,")
    print("NVIDIA Isaac, Digital Twins, VLA models, and more with better precision!")

if __name__ == "__main__":
    asyncio.run(main())