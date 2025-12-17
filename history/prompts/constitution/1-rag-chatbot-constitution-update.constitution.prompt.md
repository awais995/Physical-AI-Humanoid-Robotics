---
id: 1
title: RAG Chatbot Constitution Update
stage: constitution
date: 2025-12-14
surface: agent
model: Claude Sonnet 4.5
feature: rag-chatbot
branch: main
user: user
command: /sp.constitution --append "Title: Phase-2 Constitution — Integrated RAG Chatbot (Feature-Scoped)"
labels: ["constitution", "rag", "chatbot", "feature"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests: []
---

## Prompt

```text
--append Title: Phase-2 Constitution — Integrated RAG Chatbot (Feature-Scoped)

Scope:
This constitution applies ONLY to Phase-2 feature folder: `/specs/1-rag-chatbot/`
It MUST NOT override or modify Phase-1 constitution, specs, plans, or implementations.

Dependencies:
- Phase-1 (Book Creation & Publishing) is COMPLETE and FROZEN
- Phase-1 constitution remains authoritative for all non-RAG concerns
- Phase-2 work is additive and isolated

Non-Regression Rule:
- Phase-1 files MUST NOT be edited, regenerated, or re-implemented
- Any conflict MUST defer to Phase-1 constitution
- Violations require explicit migration approval

Purpose:
Build and embed a Retrieval-Augmented Generation (RAG) chatbot inside an already published Docusaurus book.
The chatbot answers questions STRICTLY from book content, including a hard-enforced
"user-selected-text-only" mode.

Core Principles:
- Faithfulness to retrieved content ONLY
- Zero hallucination tolerance
- Deterministic and source-grounded answers
- Modular, testable, and reversible design

Technical Standards:
- Embeddings: Cohere
- Vector Database: Qdrant Cloud (Free Tier)
- Relational Database: Neon Serverless Postgres
- Backend API: FastAPI
- Frontend: Embedded inside existing Docusaurus UI

Retrieval Modes (Hard Constraints):
- Global book retrieval
- Chapter / section retrieval
- User-selected text ONLY (no fallback allowed)

Operational Constraints:
- Stateless request handling
- No authentication or personalization
- No content rewriting or translation
- No Claude sub-agents or reusable intelligence
- Free-tier limits must be respected

Success Criteria:
- All answers derived strictly from retrieved book text
- Selected-text-only mode strictly enforced and testable
- Zero hallucinations in validation tests
- Chatbot deployed inside the book UI
- Phase-1 content remains unchanged

Governance:
This constitution is FEATURE-SCOPED.
It extends but does NOT replace Phase-1 constitution.
Future amendments require documentation and impact analysis.
```

## Response snapshot

Updated the constitution file to include both Phase 1 and Phase 2 constitutions, with Phase 2 as an extension that applies specifically to the RAG chatbot feature. The version was updated from 1.0.0 to 1.1.0 to reflect the addition of the Phase 2 constitution while preserving the Phase 1 constitution.

## Outcome

- ✅ Impact: Updated constitution to support feature-scoped governance for RAG chatbot
- 🧪 Tests: No tests run
- 📁 Files: Modified .specify/memory/constitution.md to include Phase 2 constitution
- 🔁 Next prompts: Implementation of RAG chatbot features according to new constitution
- 🧠 Reflection: Successfully implemented feature-scoped constitution approach that preserves Phase 1 while adding Phase 2 governance

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin implementation of RAG chatbot according to new constitution