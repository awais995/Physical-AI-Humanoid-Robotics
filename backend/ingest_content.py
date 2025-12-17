import asyncio
import os
from dotenv import load_dotenv
from src.services.content_service import content_service
from src.config.settings import settings

# Load environment variables
load_dotenv('backend/.env')

async def ingest_book_content():
    """
    Script to ingest book content into the vector database
    """
    print("Initializing content service...")

    # Initialize the collection
    await content_service.initialize_collection()
    print(f"Collection '{settings.qdrant_collection_name}' is ready.")

    # Example: Ingest sample content (you'll need to replace this with your actual book content)
    sample_documents = [
        {
            "content": """
            # Physical AI & Humanoid Robotics

            This book explores the fascinating world of physical AI and humanoid robotics.
            Physical AI represents a paradigm shift from traditional AI that operates on abstract data
            to AI that understands and interacts with the physical world through robotic systems.

            ## Chapter 1: Introduction to Physical AI
            Physical AI combines machine learning, robotics, and physics to create systems that
            understand the physical world. Unlike traditional AI that processes text, images, or
            audio, physical AI learns from direct interaction with the environment.

            Key concepts include:
            - Embodied cognition
            - Sensorimotor learning
            - Physics-aware neural networks
            - Real-world grounding
            """,
            "source_url": "https://physical-ai-humanoid-robotics-nine-red.vercel.app/",
            "source_title": "Physical AI & Humanoid Robotics - Introduction",
            "source_section": "Chapter 1"
        },  
        {
            "content": """
            ## Chapter 2: ROS 2 - The Robotic Nervous System

            Robot Operating System 2 (ROS 2) serves as the communication backbone for robotic
            applications. It provides a framework for writing robotic software with message passing,
            package management, and distributed processing capabilities.

            ROS 2 features include:
            - Improved security with DDS-based communication
            - Better real-time support
            - Cross-platform compatibility
            - Lifecycle management for nodes
            """,
            "source_url": "https://physical-ai-humanoid-robotics-nine-red.vercel.app/",
            "source_title": "Physical AI & Humanoid Robotics - ROS 2",
            "source_section": "Chapter 2"
        },
        {
            "content": """
            ## Chapter 3: NVIDIA Isaac - The AI Robot Brain

            NVIDIA Isaac provides a comprehensive platform for developing AI-powered robots.
            It includes simulation environments, AI frameworks, and hardware acceleration
            for robotics applications.

            Key components:
            - Isaac Sim for physics-accurate simulation
            - Isaac ROS for GPU-accelerated perception
            - Isaac Lab for reinforcement learning
            - Jetson platform for edge AI
            """,
            "source_url": "https://physical-ai-humanoid-robotics-nine-red.vercel.app/",
            "source_title": "Physical AI & Humanoid Robotics - NVIDIA Isaac",
            "source_section": "Chapter 3"
        }
    ]

    print(f"Starting ingestion of {len(sample_documents)} documents...")

    for i, doc in enumerate(sample_documents):
        print(f"Ingesting document {i+1}/{len(sample_documents)}: {doc['source_title']}")
        try:
            chunks_ingested = await content_service.ingest_document(
                content=doc["content"],
                source_url=doc["source_url"],
                source_title=doc["source_title"],
                source_section=doc["source_section"]
            )
            print(f"  - Successfully ingested {chunks_ingested} chunks")
        except Exception as e:
            print(f"  - Error ingesting document: {e}")

    # Print final stats
    stats = await content_service.get_content_stats()
    print(f"\nIngestion complete!")
    print(f"Total passages in vector database: {stats.get('total_passages', 0)}")

    return stats

async def main():
    """
    Main function to run the ingestion process
    """
    print("Physical AI & Humanoid Robotics Book - Content Ingestion Tool")
    print("=" * 60)

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
    stats = await ingest_book_content()

    print("\n" + "=" * 60)
    print("Ingestion process completed successfully!")
    print("The RAG system is now ready to answer questions about the book content.")

if __name__ == "__main__":
    asyncio.run(main())