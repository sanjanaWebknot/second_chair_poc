"""
Pydantic models for Second Chair claim checking system.
Defines structured output models for LLM response validation.
"""

from pydantic import BaseModel, Field
from typing import Literal


class ClaimCheckResponse(BaseModel):
    """
    Pydantic model for claim checking response from Ollama.
    This ensures structured, validated output for claim verification.
    """
    verdict: Literal["SUPPORT", "REFUTE", "NOT_FOUND", "UNKNOWN"] = Field(
        description="The verdict on the claim. Must be exactly one of: SUPPORT, REFUTE, NOT_FOUND or UNKNOWN"
    )
    confidence: int = Field(
        ge=0, le=100,
        description="Confidence score from 0-100"
    )
    explanation: str = Field(
        min_length=1,
        description="Detailed explanation of the verdict and reasoning"
    )