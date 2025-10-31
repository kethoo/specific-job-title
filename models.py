from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any

@dataclass
class CandidateAssessment:
    candidate_name: str
    filename: str
    overall_score: float
    fit_level: str

    education_details: Dict[str, Any]
    experience_details: Dict[str, Any]
    skills_details: Dict[str, Any]
    job_fit_details: Dict[str, Any]
    weighted_score_total: float

    executive_summary: Dict[str, Any]
    recommendation: Dict[str, Any]
    interview_focus_areas: List[str]
    red_flags: List[str]
    potential_concerns: List[str]

    assessed_at: str = datetime.now().isoformat()
