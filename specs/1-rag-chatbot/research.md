# Research Document: RAG Chatbot Implementation

## Decision: Chunking Strategy (size, overlap) and tradeoffs

**Rationale**: The chunking strategy determines how book content is divided for vector storage and retrieval, affecting both retrieval accuracy and response quality.

**Selected Approach**:
- Chunk size: 512-1024 tokens (approximately 300-600 words)
- Overlap: 50-100 tokens to maintain context across boundaries
- Strategy: Sentence-boundary aware chunking to maintain semantic coherence

**Alternatives considered**:
1. Fixed character count chunks (e.g., 1000 characters) - Rejected due to potential sentence breaks
2. Paragraph-based chunks - Rejected due to variable paragraph lengths
3. Larger chunks (2048+ tokens) - Rejected due to potential irrelevant information in retrieval

## Decision: Metadata Schema (chapter, section, source anchors)

**Rationale**: Proper metadata enables accurate source tracking and contextual retrieval.

**Selected Approach**:
- Source document ID (book identifier)
- Chapter/section title and number
- Page number or document position
- Content type (text, code, diagram caption)
- Semantic tags for topic classification

**Alternatives considered**:
1. Minimal metadata (just source file) - Rejected due to insufficient context
2. Extensive metadata (sentiment, difficulty, etc.) - Rejected as over-engineering

## Decision: Retrieval Strategy (top-k, filtering, reranking)

**Rationale**: The retrieval strategy determines how relevant content is found and ranked for response generation.

**Selected Approach**:
- Top-k retrieval: k=3-5 most relevant chunks
- Similarity threshold filtering to exclude irrelevant results
- No reranking (using vector similarity scores directly) to maintain simplicity

**Alternatives considered**:
1. Reranking with cross-encoder - Rejected due to complexity and potential cost
2. Dense retrieval with sparse fallback - Rejected as over-engineering for initial implementation
3. Semantic clustering before retrieval - Rejected due to complexity

## Decision: Selected-text-only enforcement approach

**Rationale**: The selected-text-only mode must strictly limit responses to user-provided text without fallback.

**Selected Approach**:
- Separate query processing pipeline that only uses selected text as context
- Clear error responses when questions cannot be answered from provided text
- Explicit UI indicators showing which mode is active

**Alternatives considered**:
1. Hybrid approach with warnings - Rejected as it violates the hard constraint
2. Fallback to global mode with clear warnings - Rejected as it violates the hard constraint

## Decision: Prompt structure to prevent hallucination

**Rationale**: Preventing hallucinations is critical for maintaining trust and accuracy.

**Selected Approach**:
- Context-first prompt structure: "Based on the following text: [context]"
- Explicit instruction: "Only use information from the provided text"
- Source citation requirement: "Cite specific parts of the text in your response"
- Confidence threshold: If no relevant context exists, return "Cannot answer from provided text"

**Alternatives considered**:
1. Less structured prompts - Rejected due to higher hallucination risk
2. Complex prompt engineering - Rejected as over-engineering

## Decision: Caching and rate-limit handling under free-tier constraints

**Rationale**: Free-tier services have usage limits that must be respected while maintaining performance.

**Selected Approach**:
- Request-level caching for identical queries
- Simple rate limiting with exponential backoff
- Query result caching with TTL
- Embedding reuse for unchanged content

**Alternatives considered**:
1. Aggressive caching strategies - Rejected due to complexity
2. No caching - Rejected due to potential rate limit issues