import asyncio
import os
from dotenv import load_dotenv
from src.services.content_service import content_service
from src.config.settings import settings
import re

# Load environment variables
load_dotenv('backend/.env')

def read_markdown_file(file_path):
    """
    Read content from a markdown file, removing frontmatter if present
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove YAML frontmatter if present (content between --- lines at start)
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            content = parts[2]  # Take content after the second ---

    return content.strip()

async def ingest_book_content_by_modules_and_chapters():
    """
    Script to ingest book content by modules and chapters for better retrieval
    """
    print("Initializing content service...")

    # Initialize the collection
    await content_service.initialize_collection()
    print(f"Collection '{settings.qdrant_collection_name}' is ready.")

    # Define the book structure based on actual documentation
    book_structure = [
        {
            "module": "Module 1",
            "module_title": "ROS 2 - Robotic Nervous System",
            "chapters": [
                {
                    "chapter": "Chapter 1",
                    "chapter_title": "Nodes, Topics, Services",
                    "file": "../docs/module1/ROS-2-Robotic-Nervous-System/chapter1-nodes-topics-services.md",
                    "section": "Module 1 - Chapter 1"
                },
                {
                    "chapter": "Chapter 2",
                    "chapter_title": "RCLPY Integration",
                    "file": "../docs/module1/ROS-2-Robotic-Nervous-System/chapter2-rclpy-integration.md",
                    "section": "Module 1 - Chapter 2"
                },
                {
                    "chapter": "Chapter 3",
                    "chapter_title": "URDF Humanoids",
                    "file": "../docs/module1/ROS-2-Robotic-Nervous-System/chapter3-urdf-humanoids.md",
                    "section": "Module 1 - Chapter 3"
                }
            ]
        },
        {
            "module": "Module 2",
            "module_title": "Digital Twin - Gazebo Unity",
            "chapters": [
                {
                    "chapter": "Chapter 1",
                    "chapter_title": "Gazebo Unity",
                    "file": "../docs/module2/Digital-Twin-Gazebo-Unity/chapter1-gazebo-unity.md",
                    "section": "Module 2 - Chapter 1"
                },
                {
                    "chapter": "Chapter 2",
                    "chapter_title": "Physics Simulation",
                    "file": "../docs/module2/Digital-Twin-Gazebo-Unity/chapter2-physics-simulation.md",
                    "section": "Module 2 - Chapter 2"
                },
                {
                    "chapter": "Chapter 3",
                    "chapter_title": "Sensors Simulation",
                    "file": "../docs/module2/Digital-Twin-Gazebo-Unity/chapter3-sensors-simulation.md",
                    "section": "Module 2 - Chapter 3"
                }
            ]
        },
        {
            "module": "Module 3",
            "module_title": "NVIDIA Isaac - AI Robot Brain",
            "chapters": [
                {
                    "chapter": "Chapter 1",
                    "chapter_title": "Isaac SIM",
                    "file": "../docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter1-isaac-sim.md",
                    "section": "Module 3 - Chapter 1"
                },
                {
                    "chapter": "Chapter 2",
                    "chapter_title": "VSLAM Navigation",
                    "file": "../docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter2-vslam-navigation.md",
                    "section": "Module 3 - Chapter 2"
                },
                {
                    "chapter": "Chapter 3",
                    "chapter_title": "NAV2 Bipedal",
                    "file": "../docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter3-nav2-bipedal.md",
                    "section": "Module 3 - Chapter 3"
                },
                {
                    "chapter": "Chapter 4",
                    "chapter_title": "Sim to Real",
                    "file": "../docs/module3/NVIDIA-Isaac-AI-Robot-Brain/chapter4-sim-to-real.md",
                    "section": "Module 3 - Chapter 4"
                }
            ]
        },
        {
            "module": "Module 4",
            "module_title": "Vision-Language-Action VLA",
            "chapters": [
                {
                    "chapter": "Chapter 1",
                    "chapter_title": "Whisper Commands",
                    "file": "../docs/module4/Vision-Language-Action-VLA/chapter1-whisper-commands.md",
                    "section": "Module 4 - Chapter 1"
                },
                {
                    "chapter": "Chapter 2",
                    "chapter_title": "LLM Planning",
                    "file": "../docs/module4/Vision-Language-Action-VLA/chapter2-llm-planning.md",
                    "section": "Module 4 - Chapter 2"
                },
                {
                    "chapter": "Chapter 3",
                    "chapter_title": "Voice to Action",
                    "file": "../docs/module4/Vision-Language-Action-VLA/chapter3-voice-to-action.md",
                    "section": "Module 4 - Chapter 3"
                }
            ]
        }
    ]

    print(f"Starting ingestion of {len(book_structure)} modules with chapters from your book...")
    print(f"Source: Documentation files in docs/ directory")

    total_chunks = 0
    for i, module_data in enumerate(book_structure):
        print(f"Processing {module_data['module']}: {module_data['module_title']}")

        for j, chapter_data in enumerate(module_data['chapters']):
            print(f"  - Processing {chapter_data['chapter']}: {chapter_data['chapter_title']}")

            # Read the actual chapter content from the file
            try:
                chapter_content = read_markdown_file(chapter_data['file'])

                if len(chapter_content) > 50:  # Only process if there's content
                    # Create a document for this chapter
                    try:
                        chunks_ingested = await content_service.ingest_document(
                            content=chapter_content,
                            source_url=f"https://physical-ai-humanoid-robotics-nine-red.vercel.app/{module_data['module'].lower().replace(' ', '-')}/{chapter_data['chapter'].lower().replace(' ', '-')}",
                            source_title=f"{module_data['module']}: {module_data['module_title']} - {chapter_data['chapter']}: {chapter_data['chapter_title']}",
                            source_section=chapter_data['section']
                        )
                        total_chunks += chunks_ingested
                        print(f"    - Successfully ingested {chunks_ingested} chunk(s)")
                    except Exception as e:
                        print(f"    - Error ingesting chapter: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"    - Skipping empty chapter")

            except FileNotFoundError:
                print(f"    - File not found: {chapter_data['file']}")
            except Exception as e:
                print(f"    - Error reading chapter file: {e}")

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
    print("Physical AI & Humanoid Robotics Book - Content Ingestion by Actual Modules & Chapters")
    print("=" * 85)

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
    stats = await ingest_book_content_by_modules_and_chapters()

    print("\n" + "=" * 85)
    print("Book content ingestion by modules and chapters finished successfully!")
    print("The RAG system is now loaded with properly structured book content.")
    print("It can answer questions about all 4 modules and their chapters with better precision!")

if __name__ == "__main__":
    asyncio.run(main())