<!-- Sync Impact Report:
Version change: 1.0.0 → 1.1.0
Modified principles: Added Phase-2 RAG Chatbot constitution as extension
Added sections: Phase-2 Constitution — Integrated RAG Chatbot (Feature-Scoped)
Removed sections: None
Templates requiring updates: ⚠ pending - .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md
Follow-up TODOs: None
-->
# Phase 1 AI/Spec-Driven Book Creation Constitution

## Core Principles

### Accuracy with Verified Primary Sources
All content must be based on verified primary sources; Every factual claim must be cited with credible references; Prefer peer-reviewed academic sources over secondary sources
<!-- Rationale: Ensures academic integrity and verifiability of all content -->

### Clarity for Computer-Science Audience
Content must be written with clarity for a computer-science audience; Maintain Flesch-Kincaid grade level 10-12; Technical concepts must be explained with precision and accessibility
<!-- Rationale: Ensures the target audience can understand and benefit from the content -->

### Reproducibility with Traceable Claims
All content creation processes must be reproducible; Every claim must be traceable to its source; Maintain clear audit trail from source to final output
<!-- Rationale: Enables verification and builds trust in the content -->

### Academic Rigor
Maintain academic rigor with preference for peer-reviewed sources; Minimum 15 credible sources required with at least 50% being peer-reviewed; Follow APA citation standards
<!-- Rationale: Maintains scholarly standards and credibility -->

### Consistent Structure and Tone
Maintain consistent structure and tone across all chapters; Follow standardized formatting and style guidelines; Ensure coherent narrative flow throughout the book
<!-- Rationale: Creates a unified reading experience and professional quality -->

### Zero Plagiarism Tolerance
Maintain 0% plagiarism tolerance; All content must be original with proper attribution; Use plagiarism detection tools during quality review
<!-- Rationale: Ensures originality and academic integrity -->

## Standards and Quality Requirements
Standards: APA citations, Minimum 15 credible sources (50% peer-reviewed), 0% plagiarism tolerance, Flesch-Kincaid grade 10-12 clarity, Final book length: 5,000-7,000 words, Output: Docusaurus + PDF with embedded citations
<!-- Technology stack requirements: Docusaurus, GitHub Pages, Spec-Kit Plus, Claude Code -->

## System Rules and Workflow
System rules: All content generated under specs, tasks, and implementations must follow these standards; Every factual claim must be cited; Maintain consistency and reproducibility in the entire workflow through deployment
<!-- Code review requirements: All content must be verified for compliance with standards; testing gates: Plagiarism detection, citation verification, readability checks -->

## Governance
All content must follow the specified standards; Every claim must be cited; Zero plagiarism tolerance enforced; All processes must maintain reproducibility from spec to deployment
<!-- All PRs/reviews must verify compliance; Complexity must be justified; Use .specify/memory/constitution.md for runtime development guidance -->

**Version**: 1.1.0 | **Ratified**: 2025-12-07 | **Last Amended**: 2025-12-14

# Phase-2 Constitution — Integrated RAG Chatbot (Feature-Scoped)

## Scope

This constitution applies ONLY to Phase-2 feature folder: `/specs/1-rag-chatbot/`
It MUST NOT override or modify Phase-1 constitution, specs, plans, or implementations.

## Dependencies

- Phase-1 (Book Creation & Publishing) is COMPLETE and FROZEN
- Phase-1 constitution remains authoritative for all non-RAG concerns
- Phase-2 work is additive and isolated

## Non-Regression Rule

- Phase-1 files MUST NOT be edited, regenerated, or re-implemented
- Any conflict MUST defer to Phase-1 constitution
- Violations require explicit migration approval

## Core Principles

### Faithfulness to Retrieved Content Only
All chatbot responses must be grounded solely in retrieved content from the book; No information from external sources may be incorporated; Responses must be traceable to specific passages in the book content
<!-- Rationale: Ensures the chatbot serves only as an interface to the existing book content without adding external information -->

### Zero Hallucination Tolerance
The system must never generate responses that are not supported by the book content; Any inability to answer from the provided text must result in a refusal to answer rather than fabrication; Strict adherence to source material is paramount
<!-- Rationale: Maintains trust and accuracy by preventing the generation of false information -->

### Deterministic, Source-Grounded Answers
Every response must be directly derived from the retrieved context; The system must provide clear provenance for each answer by citing specific book sections; Answer generation must be reproducible based on the same input and context
<!-- Rationale: Ensures accountability and verifiability of all responses -->

### Modular, Testable, and Reversible Design
The system architecture must be modular with clear separation of concerns; Each component (retrieval, generation, UI) must be independently testable; The system must include comprehensive monitoring and logging capabilities; Implementation must be reversible without affecting Phase-1 functionality
<!-- Rationale: Enables maintainability, debugging, and future enhancements while ensuring reversibility -->

### Embedding Standards Compliance
The system must use Cohere for embeddings; Vector storage must be implemented with Qdrant Cloud (Free Tier); A relational database (Neon Serverless Postgres) must be used for metadata and session tracking
<!-- Rationale: Ensures consistency with the chosen technology stack and performance requirements -->

### Retrieval Mode Implementation
The system must support three distinct retrieval modes: global book, chapter/section, and user-selected text only; The user-selected text mode is a hard constraint that must be implemented with no fallback allowed; Each mode must clearly indicate its scope to the user
<!-- Rationale: Provides flexible interaction patterns while maintaining strict control over the information source -->

## Technical Standards and Quality Requirements
Standards: Cohere embeddings, Qdrant Cloud vector DB, Neon Serverless Postgres, FastAPI backend, Docusaurus frontend integration, Three retrieval modes (global book, chapter/section, user-selected text only), Statelessness requirement, No authentication or personalization, Free-tier limits must be respected
<!-- Technology stack requirements: Cohere, Qdrant, Neon Postgres, FastAPI, Docusaurus -->

## System Rules and Workflow
System rules: All responses must be grounded in book content only; The chatbot must not modify book content; Retrieval and generation must be separate, testable components; The system must be stateless with no persistent user sessions; Phase-1 content must remain unchanged during all operations
<!-- Code review requirements: Verify response grounding in source material; testing gates: Hallucination detection, retrieval accuracy, source citation validation -->

## Success Criteria
- All answers derived strictly from retrieved book text
- Selected-text-only mode strictly enforced and testable
- Zero hallucinations in validation tests
- Chatbot deployed inside the book UI
- Phase-1 content remains unchanged
- No violations of Non-Regression Rule

## Governance
All implementation must follow the specified standards; Every response must be verifiable against book content; Zero hallucination tolerance enforced; All processes must maintain reproducibility from spec to deployment; This constitution is FEATURE-SCOPED and extends but does NOT replace Phase-1 constitution; Future amendments require documentation and impact analysis
<!-- All PRs/reviews must verify compliance; Complexity must be justified; Use .specify/memory/constitution.md for runtime development guidance -->