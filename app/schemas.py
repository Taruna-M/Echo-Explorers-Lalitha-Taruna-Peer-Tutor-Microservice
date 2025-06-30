from pydantic import BaseModel
from enum import Enum
from typing import Optional, List

class UrgencyLevelEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    none = "none"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        raise ValueError(f"Invalid urgency level: {value}")

class MatchRequest(BaseModel):
    user_id: str
    topic: str
    urgency_level: UrgencyLevelEnum = UrgencyLevelEnum.none
    class Config:
        extra = "forbid"

class MatchPeer(BaseModel):
    peer_id: str
    name: str
    college: str
    karma_in_topic: int
    match_score: float
    predicted_help_probability: float
    last_helped_on: str
    match_reason: List[str]

class MatchResponse(BaseModel):
    user_id: str
    matched_peers: List[MatchPeer]
    status: str # success or fail 
    