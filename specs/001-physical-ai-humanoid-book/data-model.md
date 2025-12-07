# Data Model: Physical AI & Humanoid Robotics Book

**Feature**: 001-physical-ai-humanoid-book
**Date**: 2025-12-07

## Core Entities

### Book
- **name**: String - Title of the book
- **description**: String - Brief overview of the book content
- **wordCount**: Range (3000-5000) - Required word count range
- **targetAudience**: String - Computer science audience with robotics interest
- **readabilityLevel**: String - Flesch-Kincaid grade 10-12
- **citationStyle**: String - APA format required
- **status**: Enum (draft, in-progress, review, published)

### Module
- **id**: String - Unique identifier for the module
- **title**: String - Name of the module (e.g., "ROS 2", "Digital Twin")
- **description**: String - Brief overview of module content
- **learningObjectives**: Array<String> - Specific learning outcomes
- **researchTasks**: Array<String> - Research requirements for this module
- **evidenceRequirements**: Array<String> - Evidence needed to validate learning
- **chapters**: Array<Chapter> - Collection of chapters in this module

### Chapter
- **id**: String - Unique identifier for the chapter
- **title**: String - Name of the chapter
- **description**: String - Brief overview of chapter content
- **subchapters**: Array<Subchapter> - Collection of subchapters
- **learningObjectives**: Array<String> - Specific learning outcomes
- **content**: String - Main content of the chapter

### Subchapter
- **id**: String - Unique identifier for the subchapter
- **title**: String - Name of the subchapter
- **description**: String - Brief overview of subchapter content
- **content**: String - Main content of the subchapter
- **examples**: Array<Example> - Practical examples in this subchapter

### Example
- **id**: String - Unique identifier for the example
- **title**: String - Name of the example
- **description**: String - Brief overview of the example
- **code**: String - Code snippet or implementation details
- **explanation**: String - Explanation of the example

### Citation
- **id**: String - Unique identifier for the citation
- **author**: String - Author name(s)
- **title**: String - Title of the referenced work
- **journal**: String - Journal or publication name
- **year**: Number - Publication year
- **doi**: String - Digital Object Identifier (if available)
- **url**: String - URL to the source
- **apaFormat**: String - Full citation in APA format
- **type**: Enum (journal, book, conference, online, other) - Type of source

### ValidationRule
- **id**: String - Unique identifier for the validation rule
- **name**: String - Name of the rule
- **description**: String - What the rule validates
- **requirement**: String - Specific requirement that must be met
- **testMethod**: String - How to verify compliance

## Relationships

### Book contains Modules
- Book (1) → Module (0..n)
- A book consists of multiple modules

### Module contains Chapters
- Module (1) → Chapter (0..n)
- A module consists of multiple chapters

### Chapter contains Subchapters
- Chapter (1) → Subchapter (0..n)
- A chapter consists of multiple subchapters

### Subchapter contains Examples
- Subchapter (1) → Example (0..n)
- A subchapter may contain multiple examples

### Content has Citations
- Content (0..n) → Citation (0..n)
- Multiple pieces of content can reference multiple citations

## Validation Rules

### Academic Rigor Requirements
- **Rule**: MinimumCitationsPerModule
  - **Requirement**: Each module must have at least 3 credible sources
  - **TestMethod**: Count citations referenced in module content

- **Rule**: PeerReviewedPercentage
  - **Requirement**: At least 50% of citations must be peer-reviewed
  - **TestMethod**: Categorize citations by source type and calculate percentage

### Content Quality Standards
- **Rule**: ReadabilityLevel
  - **Requirement**: Content must maintain Flesch-Kincaid grade level 10-12
  - **TestMethod**: Apply readability analysis tools to content

- **Rule**: PlagiarismFree
  - **Requirement**: Content must have 0% plagiarism
  - **TestMethod**: Run content through plagiarism detection tools

### Structural Integrity
- **Rule**: WordCountRange
  - **Requirement**: Total book content must be between 3,000-5,000 words
  - **TestMethod**: Count words in all content sections

- **Rule**: CitationFormat
  - **Requirement**: All citations must follow APA format
  - **TestMethod**: Validate citation format against APA standards

### Technical Accuracy
- **Rule**: TechnicalConceptAccuracy
  - **Requirement**: Technical concepts must be accurate and up-to-date
  - **TestMethod**: Expert review of technical content

- **Rule**: CodeExampleFunctionality
  - **Requirement**: All code examples must be functional and tested
  - **TestMethod**: Execute or validate code examples