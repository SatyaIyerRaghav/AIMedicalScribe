from __future__ import annotations
from crewai import Agent, Task, Crew, Process
from .models import Guidelines
from typing import Any, Callable, Optional
import os

##############################  Helper: Get LLM Configuration  ###########################
def _get_crewai_llm():
    """Get the LLM configuration for crewAI agents."""
    # Try Bedrock first (primary choice)
    bedrock_model = os.environ.get("BEDROCK_MODEL_ID") or os.environ.get("MODEL_NAME")
    
    print(f"🔍 [DEBUG] Checking for Bedrock model...")
    print(f"🔍 [DEBUG] BEDROCK_MODEL_ID: {os.environ.get('BEDROCK_MODEL_ID')}")
    print(f"🔍 [DEBUG] MODEL_NAME: {os.environ.get('MODEL_NAME')}")
    
    if bedrock_model:
        # CrewAI expects: bedrock/<model-id> format
        # If model already starts with 'bedrock/', use it as-is, otherwise add the prefix
        if bedrock_model.startswith("bedrock/"):
            llm_string = bedrock_model
        else:
            llm_string = f"bedrock/{bedrock_model}"
        print(f"✅ [DEBUG] Using Bedrock LLM (Claude Sonnet): {llm_string}")
        return llm_string
    
    # Try OpenAI as fallback (only if Bedrock not configured)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        print(f"🤖 [DEBUG] Bedrock not configured, using OpenAI as fallback (GPT-4)")
        return "gpt-4"  # or gpt-3.5-turbo
    
    # No API keys configured - skip agent processing
    print(f"⚠️ [WARNING] No Bedrock or OpenAI API keys configured. CrewAI agent processing will be skipped.")
    return None

##############################  Lists of Agents created  ###########################
# 1. voice_to_text_agent Agent

voice_to_text_agent = Agent(
	role="Voice-to-Text Transcriber",
    goal="Convert provider-patient audio into timestamped transcripts with speaker labels.",
    backstory="High-accuracy transcription specialist retrained on clinical audio to handle accents, interruptions, and medical terminology.",
    llm=_get_crewai_llm(),
)

# 2. clinical_understanding_agent Agent
clinical_understanding_agent = Agent(
	 role="Clinical Understanding Agent",
    goal="Extract structured clinical data (diagnoses, procedures, medications) from transcripts.",
    backstory="Modeled after experienced nurse practitioners, trained on SOAP notes to reason like a clinician.",
    llm=_get_crewai_llm(),
)

# 3. rcm_coding_agent Agent
rcm_coding_agent = Agent(
	role="RCM Coding Agent",
    goal="Assign ICD/CPT codes and modifiers based on structured clinical data.",
    backstory="Digital apprentice to professional coders trained on real-world coding decisions, building an evidence-linked audit trail.",
    llm=_get_crewai_llm(),
)

# 4. front_office_automation_agent Agent
front_office_automation_agent = Agent(
    role="Front Office Automation Agent",
    goal="Infer eligibility, payer details, authorization needs, and workflow tasks using ICD/CPT context.",
    backstory="An expert in healthcare revenue cycle front-office operations and payer eligibility workflows.",
    llm=_get_crewai_llm(),
)

# 5. compliance_quality_agent Agent
compliance_quality_agent = Agent(
	role="Compliance Oversight Agent",
    goal="Audit transcripts, coding, and documentation for compliance with clinical and billing standards.",
    backstory="Originally developed as a medical audit AI, now serving as the system's ethical conscience to catch coding errors and documentation gaps.",
    llm=_get_crewai_llm(),
)

# 6. document_agent Agent
document_agent = Agent(
    role="Documentation Agent",
    goal="Generate polished, EHR-ready clinical notes and discharge summaries.",
    backstory="Compliance-focused writer ensuring narrative is complete, accurate, and auditable.",
    llm=_get_crewai_llm(),
)

##############################Lists of Task created for above Agent ###########################

# 1. Voice to Text Agent Task

voice_to_text_task = Task(
    description="Transcribe provider-patient audio into structured, timestamped text with speaker labels.",
    expected_output="Structured transcript string with timestamps and speaker labels",
    agent=voice_to_text_agent
)


# 2. Clinical Understanding Agent Task
clinical_understanding_task = Task(
    description="Analyze the transcript to extract symptoms, diagnoses, procedures, medications, and causal relationships.",
    expected_output="Structured JSON containing clinical data like diagnoses, symptoms, medications, and procedures",
    agent=clinical_understanding_agent
)


# 3. RCM Coding Agent Task
rcm_coding_task = Task(
    description="Generate ICD, CPT, HCPCS codes and modifiers from structured clinical data, validating against payer rules.",
    expected_output="List of codes with supporting evidence and audit trail",
    agent=rcm_coding_agent
)


# 4. Front Office Automation Agent Task
front_office_task = Task(
    description="Automate patient scheduling, insurance verification, and pre-visit administrative tasks.",
    expected_output="Confirmation of scheduled appointments, eligibility status, and updated patient records",
    agent=front_office_automation_agent
)

# 5. Compliance Quality Agent Task
compliance_quality_task = Task(
    description="Audit clinical notes and coding outputs for compliance with HIPAA, CMS, and specialty-specific regulations.",
    expected_output="Report highlighting errors, gaps, or compliance issues",
    agent=compliance_quality_agent
)

# 6. Document Agent Task
document_task = Task(
    description="Synthesize verified clinical data into finalized SOAP notes or discharge summaries for EHR upload.",
    expected_output="Polished clinical documents ready for provider signature",
    agent=document_agent
)

# This Crew is a coordinated team of AI agents with predefined tasks executed in order, forming our agentic AI 
# medical scribe pipeline from audio input to final documentation.

crew = Crew(
	agents=[
		voice_to_text_agent,
		clinical_understanding_agent,
		rcm_coding_agent,
		front_office_automation_agent,
		compliance_quality_agent,
		document_agent,
	],
	tasks=[
		voice_to_text_task,
		clinical_understanding_task,
		rcm_coding_task,
		front_office_task,
		compliance_quality_task,
		document_task,
	],

	# our process is sequencial not dynamic 
	process=Process.sequential,
)


__all__ = [
	    "voice_to_text_agent",
		"clinical_understanding_agent",
		"rcm_coding_agent",
		"front_office_automation_agent",
		"compliance_quality_agent",
		"document_agent",
	    "voice_to_text_task",
		"clinical_understanding_task",
		"rcm_coding_task",
		"front_office_task",
		"compliance_quality_task",
		"document_task",
]


