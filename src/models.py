# src/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from typing import TypedDict, Optional, List, Dict, Any
# ----------------------------
# Individual Agent Data Classes
# ----------------------------
@dataclass
class VoiceToTextData:
    transcript: str = ""
    confidence: Optional[float] = None
    timestamps: Optional[List[Dict[str, Any]]] = field(default_factory=list)

@dataclass
class ClinicalUnderstandingData:
    encounters: List[Dict[str, Any]] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    diagnoses: List[str] = field(default_factory=list)
    assessment: Optional[str] = None
    plan: Optional[str] = None
    nlu_confidence: Optional[float] = None

@dataclass
class RCMCodingData:
    icd10_codes: List[str] = field(default_factory=list)
    cpt_codes: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    justification: Optional[str] = None
    coding_confidence: Optional[float] = None
    coder_notes: Optional[str] = None

@dataclass
class FrontOfficeData:
    eligibility_status: Optional[str] = None
    prior_auth_required: Optional[bool] = None
    payer: Optional[str] = None
    plan_name: Optional[str] = None
    auth_reference_id: Optional[str] = None
    turnaround_time: Optional[str] = None
    workflow_tasks: List[str] = field(default_factory=list)
    front_office_confidence: Optional[float] = None

@dataclass
class Guidelines:
    policy_text: Optional[str] = ""
    policy_id: Optional[str] = None
    coverage_rules: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class PipelineState(TypedDict, total=False):
    """
    Represents the state object passed between LangGraph nodes.
    Each task updates this shared state.
    """
    audio_file: Optional[str]
    transcript: Optional[str]
    diagnoses: Optional[List[str]]
    codes: Optional[List[str]]
    eligibility: Optional[str]
    compliance: Optional[str]
    ehr_output: Optional[Dict[str, Any]]

# ----------------------------
# Main Agent State Class
# ----------------------------
@dataclass
class AgentState:
    # Input audio file path (provided once at the start)
    source_audio: Optional[str] = None  # ✅ Only one definition now

    # Voice-to-text transcription data
    voice_to_text: VoiceToTextData = field(default_factory=VoiceToTextData)

    # Clinical understanding data
    clinical_understanding: Optional[ClinicalUnderstandingData] = None

    # RCM coding data
    rcm_coding: Optional[RCMCodingData] = None

    # Front office data
    front_office: Optional[FrontOfficeData] = None

    # Compliance and quality data
    compliance_findings: Optional[dict] = None
    phi_redacted_text: Optional[str] = None
    hipaa_audit_passed: Optional[bool] = None

    # Final structured output
    structured_output: Optional[dict] = None
    ehr_submission_status: Optional[str] = None

    # Logging
    logs: List[str] = field(default_factory=list)