import logging
import re
from typing import List, Dict, Any, Optional
import tiktoken
from ..config.settings import settings

logger = logging.getLogger(__name__)


class ChunkingService:
    """
    Service for handling content chunking with specified token strategy and overlap
    """

    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.max_chunk_size = settings.max_chunk_size
        self.min_chunk_size = settings.min_chunk_size
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Use for token counting

    def _tokenize_text(self, text: str) -> int:
        """
        Count the number of tokens in a text string
        """
        return len(self.enc.encode(text))

    def chunk_text(self, text: str, source_url: str, source_title: str,
                   source_section: Optional[str] = None,
                   source_anchor: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks of specified token size with overlap
        """
        chunks = []

        # Use a sliding window approach to chunk the text
        tokens = self.enc.encode(text)

        start_idx = 0
        while start_idx < len(tokens):
            # Determine the end index for this chunk
            end_idx = start_idx + self.chunk_size

            # If this is the last chunk, include all remaining tokens
            if end_idx > len(tokens):
                end_idx = len(tokens)

            # Decode the chunk back to text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.enc.decode(chunk_tokens)

            # Add context before and after
            context_before = ""
            context_after = ""

            # Add context from overlapping tokens if available
            context_start = max(0, start_idx - self.chunk_overlap)
            context_end = min(len(tokens), end_idx + self.chunk_overlap)

            if context_start < start_idx:
                context_tokens = tokens[context_start:start_idx]
                context_before = self.enc.decode(context_tokens)

            if end_idx < context_end:
                context_tokens = tokens[end_idx:context_end]
                context_after = self.enc.decode(context_tokens)

            # Create a chunk with metadata following the schema
            chunk = {
                "content": chunk_text,
                "source_url": source_url,
                "source_title": source_title,
                "source_section": source_section,
                "source_anchor": source_anchor,
                "context_before": context_before,
                "context_after": context_after,
                "char_start_pos": len(self.enc.decode(tokens[:start_idx])),
                "char_end_pos": len(self.enc.decode(tokens[:end_idx])),
                "token_count": len(chunk_tokens),
                "metadata": {
                    "chunk_index": len(chunks),
                    "total_chunks": 0,  # Will be updated after all chunks are created
                    "created_at": self._get_timestamp(),
                    "version": "1.0"
                }
            }

            chunks.append(chunk)

            # Move to the next chunk position (with overlap)
            start_idx = end_idx - (self.chunk_overlap if end_idx < len(tokens) else 0)

        # Update total_chunks in metadata
        for i, chunk in enumerate(chunks):
            chunk["metadata"]["total_chunks"] = len(chunks)
            chunk["metadata"]["chunk_index"] = i

        return chunks

    def _get_timestamp(self) -> str:
        """
        Get current timestamp in ISO format
        """
        from datetime import datetime
        return datetime.now().isoformat()

    def chunk_by_semantic_boundaries(self, text: str, source_url: str, source_title: str,
                                   source_section: Optional[str] = None,
                                   source_anchor: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Chunk text by semantic boundaries (paragraphs, sentences) while respecting token limits
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = self._tokenize_text(paragraph)

            # If adding this paragraph would exceed the chunk size
            if current_tokens + paragraph_tokens > self.chunk_size and current_chunk:
                # Save the current chunk
                chunk_data = self._create_chunk_data(
                    current_chunk, source_url, source_title, source_section, source_anchor, chunks
                )
                chunks.append(chunk_data)

                # Start a new chunk with overlap consideration
                current_chunk = self._get_overlap_content(current_chunk) + paragraph
                current_tokens = self._tokenize_text(current_chunk)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens

                # If this chunk is getting too large, split it
                if current_tokens > self.chunk_size:
                    subchunks = self.chunk_text(current_chunk, source_url, source_title, source_section, source_anchor)
                    chunks.extend(subchunks[:-1])  # Add all but the last chunk
                    current_chunk = subchunks[-1]["content"]  # Keep the last chunk as current
                    current_tokens = self._tokenize_text(current_chunk)

        # Add the final chunk if it has content
        if current_chunk.strip():
            chunk_data = self._create_chunk_data(
                current_chunk, source_url, source_title, source_section, source_anchor, chunks
            )
            chunks.append(chunk_data)

        # Update total_chunks in metadata
        for i, chunk in enumerate(chunks):
            chunk["metadata"]["total_chunks"] = len(chunks)
            chunk["metadata"]["chunk_index"] = i

        return chunks

    def _get_overlap_content(self, chunk: str) -> str:
        """
        Get the overlap content from the end of a chunk
        """
        tokens = self.enc.encode(chunk)
        overlap_tokens = tokens[-self.chunk_overlap:] if len(tokens) > self.chunk_overlap else tokens
        overlap_content = self.enc.decode(overlap_tokens)
        return overlap_content

    def _create_chunk_data(self, content: str, source_url: str, source_title: str,
                          source_section: Optional[str], source_anchor: Optional[str],
                          existing_chunks: List) -> Dict[str, Any]:
        """
        Create a chunk data dictionary with proper metadata
        """
        token_count = self._tokenize_text(content)

        return {
            "content": content,
            "source_url": source_url,
            "source_title": source_title,
            "source_section": source_section,
            "source_anchor": source_anchor,
            "context_before": "",
            "context_after": "",
            "char_start_pos": 0,  # This would need to be calculated based on original position
            "char_end_pos": len(content),
            "token_count": token_count,
            "metadata": {
                "chunk_index": len(existing_chunks),
                "total_chunks": 0,  # Will be updated later
                "created_at": self._get_timestamp(),
                "version": "1.0",
                "chunking_strategy": "semantic"
            }
        }


# Global chunking service instance
chunking_service = ChunkingService()