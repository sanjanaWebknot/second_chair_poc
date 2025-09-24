"""
Pydantic models for Second Chair claim checking system.
Defines structured output models for claim verification responses.
"""

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
from enum import Enum


class VerdictType(str, Enum):
    """Valid verdict types for claim checking."""
    SUPPORT = "SUPPORT"
    REFUTE = "REFUTE"
    NOT_FOUND = "NOT_FOUND"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class ClaimCheckResponse(BaseModel):
    """
    Pydantic model for claim checking response from Ollama.
    This ensures structured, validated output for claim verification.
    """
    verdict: VerdictType = Field(
        description="The verdict on the claim: SUPPORT, REFUTE, NOT_FOUND, ERROR, or UNKNOWN"
    )
    confidence: int = Field(
        ge=0, le=100,
        description="Confidence score from 0-100"
    )
    explanation: str = Field(
        min_length=1,
        description="Detailed explanation of the verdict and reasoning"
    )
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is within valid range."""
        if not isinstance(v, int):
            raise ValueError('Confidence must be an integer')
        if v < 0 or v > 100:
            raise ValueError('Confidence must be between 0 and 100')
        return v
    
    @validator('explanation')
    def validate_explanation(cls, v):
        """Ensure explanation is not empty."""
        if not v or not v.strip():
            raise ValueError('Explanation cannot be empty')
        return v.strip()


class FactCheckResult(BaseModel):
    """
    Pydantic model for the complete fact-check result sent via WebSocket.
    Includes metadata and processing information.
    """
    type: Literal["fact_check_result"] = "fact_check_result"
    pair_id: str = Field(description="Unique identifier for the Q&A pair")
    sequence_number: int = Field(description="Sequence number of the pair")
    verdict: VerdictType = Field(description="Claim verification verdict")
    confidence: int = Field(ge=0, le=100, description="Confidence score")
    explanation: str = Field(description="Detailed explanation")
    question: str = Field(description="The question being fact-checked")
    answer: str = Field(description="The answer being fact-checked")
    evidence_count: int = Field(ge=0, description="Number of evidence documents found")
    processing_time_ms: int = Field(ge=0, description="Processing time in milliseconds")
    timestamp: str = Field(description="ISO timestamp of the result")


class FactCheckError(BaseModel):
    """
    Pydantic model for fact-check error responses.
    """
    type: Literal["fact_check_error"] = "fact_check_error"
    pair_id: str = Field(description="Unique identifier for the Q&A pair")
    sequence_number: int = Field(description="Sequence number of the pair")
    error: str = Field(description="Error message")
    timestamp: str = Field(description="ISO timestamp of the error")


class WordAck(BaseModel):
    """
    Pydantic model for word acknowledgment messages.
    """
    type: Literal["word_ack"] = "word_ack"
    words_processed: int = Field(ge=0, description="Total words processed")
    buffer_size: int = Field(ge=0, description="Current buffer size")
    states: dict = Field(description="Buffer state information")
    current_q: Optional[str] = Field(default="", description="Current question being formed")
    current_a: Optional[str] = Field(default="", description="Current answer being formed")
    transcript_size: Optional[int] = Field(default=None, description="Current transcript size")


class PairStored(BaseModel):
    """
    Pydantic model for pair storage confirmation.
    """
    type: Literal["pair_stored"] = "pair_stored"
    sequence_number: int = Field(description="Sequence number of stored pair")
    question: str = Field(description="Stored question")
    answer: str = Field(description="Stored answer")
    transcript_size: int = Field(description="Updated transcript size")
