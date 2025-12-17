# Implementation Plan: Integrated RAG Chatbot for the Docusaurus Book (Phase 2)

**Branch**: `1-rag-chatbot` | **Date**: 2025-12-14 | **Spec**: [specs/1-rag-chatbot/spec.md](specs/1-rag-chatbot/spec.md)
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a Retrieval-Augmented Generation (RAG) chatbot that integrates with the existing Docusaurus book site. The system will provide two query modes: global book retrieval and selected-text-only retrieval, with responses grounded 100% in book content. The architecture includes content ingestion, vector storage with Qdrant Cloud, metadata management with Neon Postgres, and a FastAPI backend integrated into the Docusaurus frontend.

## Technical Context

**Language/Version**: Python 3.11, JavaScript/TypeScript for frontend integration
**Primary Dependencies**: FastAPI, Cohere SDK, Qdrant client, Neon Postgres driver, Docusaurus
**Storage**: Qdrant Cloud (vector DB), Neon Serverless Postgres (metadata/session tracking)
**Testing**: pytest for backend, Jest for frontend components
**Target Platform**: Linux server (backend), Web browser (frontend)
**Project Type**: Web application (backend API + frontend integration)
**Performance Goals**: Response time under 5 seconds for 95% of requests, handle concurrent users
**Constraints**: Free tier limits for Qdrant and Neon, no breaking changes to existing Docusaurus site
**Scale/Scope**: Single book content, multiple concurrent users, persistent conversation history

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

1. **Faithfulness to Retrieved Content Only**: All responses must be grounded solely in retrieved content from the book - PASSED
2. **Zero Hallucination Tolerance**: System must never generate responses not supported by book content - PASSED
3. **Deterministic, Source-Grounded Answers**: Every response must be directly derived from retrieved context with clear provenance - PASSED
4. **Modular, Testable, and Reversible Design**: Architecture must be modular with clear separation of concerns - PASSED
5. **Embedding Standards Compliance**: Must use Cohere for embeddings, Qdrant Cloud, and Neon Postgres - PASSED
6. **Retrieval Mode Implementation**: Must support global book, chapter/section, and user-selected text only modes - PASSED
7. **Non-Regression Rule**: Phase-1 files must not be edited - PASSED

## Project Structure

### Documentation (this feature)

```text
specs/1-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── conversation.py
│   │   ├── query.py
│   │   └── response.py
│   ├── services/
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   ├── generation_service.py
│   │   └── content_service.py
│   ├── api/
│   │   ├── chat_endpoints.py
│   │   └── ingestion_endpoints.py
│   └── config/
│       ├── settings.py
│       └── database.py
└── tests/
    ├── unit/
    ├── integration/
    └── contract/

frontend/
├── src/
│   ├── components/
│   │   └── RagChatbot/
│   │       ├── ChatInterface.jsx
│   │       ├── QueryModeSelector.jsx
│   │       └── ResponseRenderer.jsx
│   ├── services/
│   │   └── chat-api.js
│   └── hooks/
│       └── useChat.js
└── static/
    └── js/
        └── chat-integration.js
```

**Structure Decision**: Web application structure with separate backend API and frontend integration components. The backend handles all RAG operations (ingestion, embedding, retrieval, generation) while the frontend provides the chat interface integrated into the existing Docusaurus site.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |