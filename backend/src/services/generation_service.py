import asyncio
import logging
from typing import List, Optional, Dict, Any
import cohere
from ..models.retrieved_passage import RetrievedPassage
from ..models.query import Query, QueryMode
from ..config.settings import settings
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Service for generating responses based on retrieved passages and preventing hallucinations
    """

    def __init__(self):
        self.co = cohere.Client(settings.cohere_api_key)
        self.embedding_service = EmbeddingService()
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.hallucination_check_enabled = settings.hallucination_check_enabled

    async def generate_response(self, query: Query, retrieved_passages: List[RetrievedPassage]) -> str:
        """
        Generate a response based on the query and retrieved passages
        """
        try:
            # Build context from retrieved passages
            context_text = self._build_context_from_passages(retrieved_passages)

            # Build the prompt based on query mode
            if query.query_mode == QueryMode.SELECTED_TEXT_ONLY:
                prompt = await self._build_selected_text_prompt(query, context_text)
            else:
                prompt = await self._build_global_prompt(query, context_text)

            # Generate response using Cohere Chat API (replacing deprecated Generate API)
            response = self.co.chat(
                model="c4ai-aya-expanse-8b",  # Using Cohere's available model (replacing retired command-r-plus)
                message=prompt,  # The user's message/prompt
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            # Access the response text - Cohere Chat API returns response.text
            generated_text = response.text.strip()

            # Perform hallucination check if enabled
            if self.hallucination_check_enabled:
                is_valid = await self._check_hallucination(generated_text, retrieved_passages)
                if not is_valid:
                    # If hallucination detected, generate a more conservative response
                    warning_text = (
                        "I couldn't find sufficient information in the provided text to answer your question. "
                        "The response is based on the available context, but please verify with the original source."
                    )
                    generated_text = f"{warning_text}\n\n{generated_text}"

            return generated_text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Log more details about the error
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Try to get more information about the response object if it exists
            try:
                logger.error(f"Response object type: {type(response) if 'response' in locals() else 'response not created'}")
                if 'response' in locals():
                    logger.error(f"Response attributes: {dir(response)}")
            except Exception as log_error:
                logger.error(f"Error logging response details: {log_error}")

            return "I encountered an error while generating a response. Please try again."

    def _build_context_from_passages(self, passages: List[RetrievedPassage]) -> str:
        """
        Build context text from retrieved passages
        """
        context_parts = []
        for i, passage in enumerate(passages):
            context_parts.append(
                f"Source {i+1}:\n"
                f"Title: {passage.source_title}\n"
                f"Section: {passage.source_section or 'N/A'}\n"
                f"Content: {passage.content}\n"
                "---\n"
            )

        return "\n".join(context_parts)

    async def _build_global_prompt(self, query: Query, context_text: str) -> str:
        """
        Build prompt for global book mode
        """
        prompt = (
            f"Please answer the following question based ONLY on the provided context from the book. "
            f"Do not use any external knowledge or make up information.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query.query_text}\n\n"
            f"Instructions:\n"
            f"- Answer based only on the provided context\n"
            f"- If the context doesn't contain the answer, say so explicitly\n"
            f"- Be concise and to the point\n"
            f"- Only explain if necessary, otherwise provide direct answers\n"
            f"- Do not include confidence scores or references in the response\n"
            f"- If asked about topics not in the context, clearly state that the information is not in the provided text\n\n"
            f"Answer:"
        )
        return prompt

    async def _build_selected_text_prompt(self, query: Query, context_text: str) -> str:
        """
        Build prompt for selected text only mode
        """
        if not query.selected_text:
            # Fallback to global mode if no selected text
            return await self._build_global_prompt(query, context_text)

        prompt = (
            f"You are given a specific text selection and a question about it. "
            f"Answer the question based ONLY on this selected text and the provided context. "
            f"Do not use any external knowledge or information outside of what's provided.\n\n"
            f"Selected Text: {query.selected_text}\n\n"
            f"Additional Context:\n{context_text}\n\n"
            f"Question: {query.query_text}\n\n"
            f"Instructions:\n"
            f"- Answer based only on the selected text and provided context\n"
            f"- If the selected text and context don't contain the answer, say so explicitly\n"
            f"- Do not make up information or infer beyond what's directly stated\n"
            f"- Be concise and to the point\n"
            f"- Only explain if necessary, otherwise provide direct answers\n"
            f"- Do not include confidence scores or references in the response\n"
            f"- If the question requires information not in the selected text or context, clearly state that\n\n"
            f"Answer:"
        )
        return prompt

    async def _check_hallucination(self, generated_text: str, retrieved_passages: List[RetrievedPassage]) -> bool:
        """
        Check if the generated text contains hallucinations by comparing with retrieved passages
        """
        try:
            if not retrieved_passages:
                return False  # If no context, we can't verify the response

            # Check if the key claims in the response are supported by the context
            passage_texts = [p.content for p in retrieved_passages]
            combined_context = " ".join(passage_texts)

            # Use embedding similarity to check if the response is related to the context
            response_embedding = await self.embedding_service.get_embedding(generated_text)
            context_embedding = await self.embedding_service.get_embedding(combined_context)

            similarity = await self.embedding_service.get_similarity_score(generated_text, combined_context)

            # If similarity is very low, it might indicate hallucination
            if similarity < 0.2:
                logger.warning(f"Low similarity between response and context: {similarity:.3f}")
                return False

            # Additional check: look for specific phrases that indicate uncertainty
            uncertain_phrases = [
                "i think", "i believe", "maybe", "perhaps", "possibly",
                "i'm not sure", "i don't know", "unclear", "not mentioned"
            ]

            response_lower = generated_text.lower()
            has_uncertain_language = any(phrase in response_lower for phrase in uncertain_phrases)

            # If the response shows uncertainty but has high similarity, it might be valid
            # If it shows confidence but low similarity, it might be hallucinated
            if similarity > 0.5 and not has_uncertain_language:
                return True  # Likely valid
            elif similarity < 0.3 and not has_uncertain_language:
                return False  # Likely hallucinated
            else:
                return True  # Uncertain, assume valid

        except Exception as e:
            logger.error(f"Error in hallucination check: {e}")
            return True  # If we can't check, assume it's valid

    async def get_confidence_score(self, query: Query, response: str, retrieved_passages: List[RetrievedPassage]) -> float:
        """
        Calculate a confidence score for the generated response
        """
        try:
            if not retrieved_passages:
                return 0.0

            # Calculate average similarity of retrieved passages
            avg_similarity = sum(p.similarity_score for p in retrieved_passages) / len(retrieved_passages)

            # Check how much of the response is supported by the context
            response_embedding = await self.embedding_service.get_embedding(response)

            # Calculate similarity between response and context
            context_texts = [p.content for p in retrieved_passages]
            context_text = " ".join(context_texts)
            response_context_similarity = await self.embedding_service.get_similarity_score(response, context_text)

            # Combine factors for confidence score
            # Weight: 60% from passage similarity, 40% from response-context alignment
            confidence = (avg_similarity * 0.6) + (response_context_similarity * 0.4)

            # Ensure confidence is between 0 and 1
            return max(0.0, min(1.0, confidence))

        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0

    async def generate_citations(self, retrieved_passages: List[RetrievedPassage]) -> List[Dict[str, Any]]:
        """
        Generate citations for the retrieved passages
        """
        citations = []
        for passage in retrieved_passages:
            citation = {
                "source_title": passage.source_title,
                "source_section": passage.source_section,
                "source_url": passage.source_url,
                "similarity_score": passage.similarity_score,
                "content_preview": passage.content[:200] + "..." if len(passage.content) > 200 else passage.content,
                "char_start_pos": passage.char_start_pos,
                "char_end_pos": passage.char_end_pos
            }
            citations.append(citation)

        return citations


# Global generation service instance
generation_service = GenerationService()