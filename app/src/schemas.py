from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class MatchRequest(BaseModel):
    donor_id: int = Field(..., ge=1, description="On-chain donor ID")

class BaselineCandidate(BaseModel):
    rank: int
    recipient_id: int
    score: float
    factors: Dict[str, Any] = Field(default_factory=dict)

class RiskFlag(BaseModel):
    recipient_id: int
    risk_level: str = Field(..., description="low|medium|high")
    flags: List[str] = Field(default_factory=list)

class LLMDecision(BaseModel):
    donor_id: int
    primary_recipient_id: int
    backup_recipient_id: int

    overrode_baseline: bool = False
    override_reason: Optional[str] = None

    risk_flags: List[RiskFlag] = Field(default_factory=list)
    explanation: str = ""

class OnChainRecord(BaseModel):
    tx_hash: str
    match_id: Optional[int] = None
    gas_used: Optional[int] = None
    contract_address: str

class MatchResponse(BaseModel):
    donor_id: int
    baseline_top: List[BaselineCandidate]
    llm_decision: LLMDecision
    match_cid: str
    onchain: OnChainRecord
