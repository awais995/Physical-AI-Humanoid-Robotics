---
sidebar_position: -1
---

# Base Content Entities

This file defines the core entities referenced in the Physical AI & Humanoid Robotics book based on the data model.

## Core Entity Definitions

### Book
- **name**: Physical AI & Humanoid Robotics
- **description**: A comprehensive guide to modern robotics and AI integration
- **wordCount**: 3,000-5,000 range
- **targetAudience**: Computer science audience with robotics interest
- **readabilityLevel**: Flesch-Kincaid grade 10-12
- **citationStyle**: APA format required
- **status**: Draft → In-progress → Review → Published

### Module
- **id**: Unique identifier for the module (e.g., "module1-ros2")
- **title**: Name of the module (e.g., "ROS 2", "Digital Twin")
- **description**: Brief overview of module content
- **learningObjectives**: Specific learning outcomes
- **researchTasks**: Research requirements for this module
- **evidenceRequirements**: Evidence needed to validate learning
- **chapters**: Collection of chapters in this module

### Chapter
- **id**: Unique identifier for the chapter
- **title**: Name of the chapter
- **description**: Brief overview of chapter content
- **subchapters**: Collection of subchapters
- **learningObjectives**: Specific learning outcomes
- **content**: Main content of the chapter

### Subchapter
- **id**: Unique identifier for the subchapter
- **title**: Name of the subchapter
- **description**: Brief overview of subchapter content
- **content**: Main content of the subchapter
- **examples**: Practical examples in this subchapter

### Example
- **id**: Unique identifier for the example
- **title**: Name of the example
- **description**: Brief overview of the example
- **code**: Code snippet or implementation details
- **explanation**: Explanation of the example

### Citation
- **id**: Unique identifier for the citation
- **author**: Author name(s)
- **title**: Title of the referenced work
- **journal**: Journal or publication name
- **year**: Publication year
- **doi**: Digital Object Identifier (if available)
- **url**: URL to the source
- **apaFormat**: Full citation in APA format
- **type**: Type of source (journal, book, conference, online, other)

These entities form the foundational structure for organizing content in the book according to constitutional standards.