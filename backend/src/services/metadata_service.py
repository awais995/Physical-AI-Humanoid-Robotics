import logging
from typing import Dict, Any, Optional
from datetime import datetime
from ..config.settings import settings

logger = logging.getLogger(__name__)


class MetadataService:
    """
    Service for handling metadata schema with chapter, section, and source anchors
    """

    @staticmethod
    def create_metadata_schema(source_url: str, source_title: str, source_section: Optional[str] = None,
                             source_anchor: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Create a standardized metadata schema for content
        """
        metadata = {
            "source_url": source_url,
            "source_title": source_title,
            "source_section": source_section,
            "source_anchor": source_anchor,
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "content_type": kwargs.get("content_type", "text"),
            "language": kwargs.get("language", "en"),
            "token_count": kwargs.get("token_count", 0),
            "char_count": kwargs.get("char_count", 0),
            "processing_version": settings.app_version
        }

        # Add any additional metadata provided
        for key, value in kwargs.items():
            if key not in metadata:
                metadata[key] = value

        return metadata

    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> bool:
        """
        Validate that metadata contains required fields
        """
        required_fields = ["source_url", "source_title", "created_at"]
        return all(field in metadata for field in required_fields)

    @staticmethod
    def filter_passages_by_threshold(passages: list, threshold: float = None) -> list:
        """
        Filter passages based on similarity threshold
        """
        if threshold is None:
            threshold = settings.similarity_threshold

        return [p for p in passages if hasattr(p, 'similarity_score') and p.similarity_score >= threshold]

    @staticmethod
    def enrich_metadata_with_context(metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich metadata with additional context information
        """
        enriched_metadata = metadata.copy()
        enriched_metadata.update(context)
        enriched_metadata["last_updated"] = datetime.now().isoformat()
        return enriched_metadata


# Global metadata service instance
metadata_service = MetadataService()