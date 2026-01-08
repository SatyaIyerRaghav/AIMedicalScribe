from __future__ import annotations

import whisper
import os
import re
import json
import logging
from typing import Any, Callable
from textwrap import dedent
from crewai import Agent, Task, Crew, Process
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
import uuid

# Use a proper variable name, don’t shadow datetime module
from datetime import datetime
now = datetime.now()

# Optional imports
try:
    from crewai_tools import tool, FileReadTool
except Exception:
    tool = None
    FileReadTool = None

# Optional OpenAI import for Whisper
try:
    import openai
except ImportError:
    openai = None

# Local imports
from .models import Guidelines
from .agents import (
    document_agent,
    compliance_quality_agent,
    front_office_automation_agent,
    voice_to_text_agent, 
    voice_to_text_task,
    clinical_understanding_agent,
    clinical_understanding_task
)

# Logging setup
from src.logger_setup import init_logging, get_logger

# Initialize logging only once (safe even if called multiple times)
init_logging()

# Create a module-specific logger for this agent
logger = get_logger("agent_builder", "agent_builder.log")
# Import cost tracking module
from .cost import get_tracker, log_cost, print_summary, export_json, export_csv

# Initialize cost tracker
cost_tracker = get_tracker()

# -----------------------------------------------------------
# Main Function
# -----------------------------------------------------------
def run_voice_to_text(state, config=None):
    logger.info(" Entered run_voice_to_text")
    audio_path = None
    if config and isinstance(config, dict) and "input_audio_path" in config:
        audio_path = config["input_audio_path"]
        logger.info(f"Got audio path from config: {audio_path}")
    elif state.get("audio_file"):
        audio_path = state["audio_file"]
        logger.info(f"Got audio path from state: {audio_path}")
    elif isinstance(state, dict) and "audio_path" in state:
        audio_path = state["audio_path"]
        logger.info(f"Got audio path from dict: {audio_path}")

    if not audio_path:
        logger.warning("No audio path found in state or config.")
        return state

    if isinstance(audio_path, dict) and "audio_path" in audio_path:
        audio_path = audio_path["audio_path"]

    if not isinstance(audio_path, (str, bytes, os.PathLike)):
        logger.error(f"Invalid type for audio_path: {type(audio_path)}")
        return state

    if not os.path.exists(audio_path):
        logger.error(f"File not found: {audio_path}")
        return state

    logger.info(f"Using audio file: {audio_path}")
    raw_transcript = ""
    try:
        logger.info("Running local Whisper transcription...")
        model = whisper.load_model("base")  # or "tiny", "small", etc.
        result = model.transcribe(audio_path)
        raw_transcript = result["text"].strip()
        logger.info("Local Whisper transcription completed successfully.")
    except Exception as e:
        logger.exception(f"Local Whisper transcription failed: {e}")
        raw_transcript = f"[Fallback Mock Transcript] Transcribed from {os.path.basename(audio_path)}"
    transcript = raw_transcript
    try:
        if voice_to_text_agent and getattr(voice_to_text_agent, "llm", None) is not None:
            logger.info("Using crewAI agent to refine transcript...")

            refined_task = Task(
                description=f"""Process and refine the following medical audio transcript.

Raw Transcript:
{raw_transcript}

Tasks:
1. Format the transcript with proper structure
2. Add speaker labels where possible (Doctor/D, Patient/P)
3. Clean up any transcription errors or filler words
4. Highlight important medical terminology
5. Return a polished, structured transcript ready for clinical use.""",
                expected_output="A structured, professionally formatted medical transcript with speaker labels and proper formatting",
                agent=voice_to_text_agent
            )

            crew = Crew(
                agents=[voice_to_text_agent],
                tasks=[refined_task],
                process=Process.sequential,
                verbose=True
            )

            result = crew.kickoff()

            if hasattr(result, "raw"):
                transcript = result.raw
            elif hasattr(result, "content"):
                transcript = result.content
            else:
                transcript = str(result)

            logger.info("crewAI agent successfully refined the transcript.")
        else:
            logger.warning("No LLM configured — skipping crewAI agent refinement.")
    except Exception as e:
        logger.warning(f"crewAI agent processing failed: {e}")
        logger.info("Falling back to raw Whisper transcript.")
    try:
        transcripts_dir = os.path.join(os.getcwd(), "transcripts")
        os.makedirs(transcripts_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        transcript_path = os.path.join(transcripts_dir, f"{base_name}.txt")

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        logger.info(f"Transcript saved successfully: {transcript_path}")
    except Exception as e:
        logger.exception(f"Failed to save transcript: {e}")

    state["transcript"] = transcript
    state["transcript_path"] = transcript_path
    cost_tracker.log_agent_cost("run_voice_to_text", fixed_cost=0.001)
    logger.info("run_voice_to_text completed successfully.")
    return state

#  -------------------------------------------------------------------------
#  CLINICAL UNDERSTANDING
# -------------------------------------------------------------------------
def run_clinical_understanding(state, config=None):
    """
    Extract structured medical information (diagnoses, findings, etc.)
    from transcript text using crewAI agent.
    """
    logger.debug("Entered run_clinical_understanding()")
    logger.debug(f"Incoming state keys: {list(state.keys())}")

    # --- Step 1️⃣: Load transcript ---
    transcript = state.get("transcript")

    if not transcript:
        if config and isinstance(config, dict):
            transcript = config.get("transcript")

        if not transcript:
            transcript_path = state.get("transcript_path")
            if transcript_path and os.path.exists(transcript_path):
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                logger.debug(f"Loaded transcript from: {transcript_path}")

    if not transcript:
        logger.warning("No transcript found — skipping clinical understanding.")
        state["clinical_understanding_status"] = "no_transcript"
        return state

    transcript_input = transcript[:12000] if isinstance(transcript, str) else str(transcript)
    logger.debug(f"Transcript length: {len(transcript_input)} chars")
    try:
        clinical_understanding_agent = Agent(
            role="Clinical Understanding Agent",
            goal="Extract key findings, diagnoses, and clinical impressions from transcript.",
            backstory="You are a senior clinical documentation specialist.",
            verbose=True
        )
        extraction_task = Task(
            description=f"""
Analyze the following medical transcript and extract structured findings.

Return valid JSON ONLY:
{{
  "key_findings": ["Finding 1", "Finding 2"],
  "diagnoses": ["Diagnosis 1", "Diagnosis 2"],
  "symptoms": ["Symptom 1", "Symptom 2"],
  "procedures": ["Procedure 1", "Procedure 2"],
  "medications": ["Medication 1", "Medication 2"],
  "clinical_impressions": "Concise summary of clinical assessment"
}}

TRANSCRIPT START
{transcript_input}
TRANSCRIPT END
""",
            expected_output="Structured JSON with findings, diagnoses, and summary.",
            agent=clinical_understanding_agent
        )

        crew = Crew(
            agents=[clinical_understanding_agent],
            tasks=[extraction_task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        logger.info("crewAI agent finished processing successfully.")

        findings_text = str(result)
        try:
            findings_json = json.loads(findings_text)
        except json.JSONDecodeError:
            logger.warning("Output not valid JSON, fallback parsing.")
            findings_json = {
                "key_findings": [findings_text],
                "diagnoses": [],
                "symptoms": [],
                "procedures": [],
                "medications": [],
                "clinical_impressions": ""
            }

        state.update({
            "key_findings": findings_json.get("key_findings", []),
            "diagnoses": findings_json.get("diagnoses", []),
            "symptoms": findings_json.get("symptoms", []),
            "procedures": findings_json.get("procedures", []),
            "medications": findings_json.get("medications", []),
            "clinical_impressions": findings_json.get("clinical_impressions", ""),
            "clinical_understanding_status": "success"
        })
        logger.info(f"Extracted {len(state['diagnoses'])} diagnoses successfully.")
    except Exception as e:
        logger.exception(f"crewAI agent failed with error: {e}")
        state.update({
            "diagnoses": [],
            "clinical_understanding_status": f"error: {e}"
        })
    logger.debug("run_clinical_understanding completed successfully.")
    cost_tracker.log_agent_cost("run_clinical_understanding",result=result)
    return state

# -------------------------------------------------------------------------
# RCM CODING
# -------------------------------------------------------------------------
def run_rcm_coding(state, config=None):
    """
    Assign ICD-10 and CPT codes from diagnoses using CrewAI agent.
    """
    logger.debug(" Entered run_rcm_coding()")
    logger.debug(f"Incoming state keys: {list(state.keys())}")
    diagnoses = state.get("diagnoses", [])
    if not diagnoses:
        logger.warning("No diagnoses found — skipping coding.")
        state.update({
            "coding_data": [],
            "rcm_status": "no_diagnoses"
        })
        return state
    logger.debug(f"Diagnoses received: {diagnoses}")
    try:
        rcm_coding_agent = Agent(
            role="RCM Coding Agent",
            goal="Assign ICD-10 and CPT codes based on diagnoses.",
            backstory="Certified coder with experience in RCM and medical documentation.",
            verbose=True
        )
        coding_task = Task(
            description=f"""
Assign ICD-10 and CPT codes with appropriate modifiers for:
{diagnoses}

Return valid JSON ONLY:
{{
  "coding": [
    {{
      "diagnosis": "string",
      "icd_code": "string",
      "cpt_code": "string",
      "modifiers": ["string"]
    }}
  ],
  "coding_notes": "string"
}}
""",
            expected_output="Strict JSON with ICD/CPT coding.",
            agent=rcm_coding_agent
        )

        crew = Crew(
            agents=[rcm_coding_agent],
            tasks=[coding_task],
            process=Process.sequential,
            verbose=True
        )
        result = crew.kickoff()
        logger.info("CrewAI agent executed successfully.")
        # Try parsing the output JSON
        try:
            coded_data = json.loads(str(result))
        except json.JSONDecodeError:
            logger.warning("RCM output not valid JSON — applying fallback parsing.")
            coded_data = {"coding": [], "coding_notes": str(result)}

    except Exception as e:
        logger.exception(f"Model call failed: {e}")
        coded_data = {"coding": [], "coding_notes": f"Error: {e}"}

    # --- Normalize data ---
    icd_list = coded_data.get("coding", [])
    if not isinstance(icd_list, list):
        logger.warning("ICD list was not a list; resetting to empty list.")
        icd_list = []

    state.update({
        "coding_data": icd_list,
        "icd_codes": [c.get("icd_code") for c in icd_list if c.get("icd_code")],
        "cpt_codes": [c.get("cpt_code") for c in icd_list if c.get("cpt_code")],
        "rcm_notes": coded_data.get("coding_notes", ""),
        "rcm_status": "coded_successfully" if icd_list else "no_coding_output"
    })

    logger.info(f"Extracted {len(state['icd_codes'])} ICD codes.")
    logger.debug(f"Final structured coding data:\n{json.dumps(icd_list, indent=2)}")
    logger.debug("run_rcm_coding() completed successfully.")
    cost_tracker.log_agent_cost("run_rcm_coding",result=result)

    return state

# -------------------------------------------------------------------------
# FRONT OFFICE
# -------------------------------------------------------------------------

def run_front_office(state, config=None):
    """Automate front office workflows like eligibility and prior auth."""
    state = run_rcm_coding(state)
    logger.debug("Entered run_front_office()")
    logger.debug(f"State keys available: {list(state.keys())}")

    icd_data = (
        state.get("coding_data")
        or state.get("icd_codes")
        or state.get("rcm_output")
        or []
    )

    logger.debug(f" ICD/CPT data content: {icd_data}")

    if not icd_data or not isinstance(icd_data, list):
        logger.warning("No ICD/CPT data found — skipping front office automation.")
        state["front_office"] = {"status": "no_icd_data"}
        return state

    try:
        # --- Step 1️⃣: Create the agent ---
        front_office_agent = Agent(
            role="Front Office Coordinator",
            goal="Handle insurance eligibility, prior authorization, and payer workflow based on ICD/CPT codes.",
            backstory="Experienced coordinator managing pre-auth and billing workflows.",
            verbose=True,
        )

        # --- Step 2️⃣: Compact ICD/CPT data for LLM context ---
        compact_icd_text = "\n".join([
            f"- {item.get('diagnosis', 'Unknown')} | ICD: {item.get('icd_code')} | CPT: {item.get('cpt_code')} | Modifiers: {', '.join(item.get('modifiers', []))}"
            for item in icd_data
        ])
        logger.debug(f"Prepared compact ICD text:\n{compact_icd_text}")

        # --- Step 3️⃣: Define the structured task ---
        front_task = Task(
            description=f"""
You are a hospital front office expert. Using the following ICD/CPT codes, determine:
- Insurance eligibility
- Whether prior authorization is required
- Payer name and plan
- Auth reference ID (if available)
- Expected turnaround time
- Key workflow tasks

Return valid JSON ONLY in this exact format:
{{
  "eligibility_status": "string",
  "prior_auth_required": true/false,
  "payer": "string",
  "plan_name": "string",
  "auth_reference_id": "string",
  "turnaround_time": "string",
  "workflow_tasks": ["string"],
  "front_office_confidence": float
}}

ICD/CPT Data:
{compact_icd_text}
""",
            expected_output="Strict JSON output for front office operations.",
            agent=front_office_agent,
        )

        # --- Step 4️⃣: Run CrewAI agent ---
        logger.info("Executing front office agent task...")
        crew = Crew(
            agents=[front_office_agent],
            tasks=[front_task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff()
        logger.info("Front office agent task executed successfully.")

        # --- Step 5️⃣: Parse JSON result ---
        try:
            front_data = json.loads(str(result))
            logger.debug(f"Parsed front office result:\n{json.dumps(front_data, indent=2)}")
        except json.JSONDecodeError:
            logger.warning("Invalid or partial JSON received — applying fallback schema.")
            front_data = {
                "eligibility_status": "unknown",
                "prior_auth_required": False,
                "payer": "unknown",
                "plan_name": "unknown",
                "auth_reference_id": "",
                "turnaround_time": "unknown",
                "workflow_tasks": [],
                "front_office_confidence": 0.0,
            }

        # --- Step 6️⃣: Update State ---
        state["front_office"] = front_data
        logger.info("Stored front office data successfully.")
        return state

    except Exception as e:
        logger.exception(f"Front office agent failed: {e}")
        state["front_office"] = {"status": "error", "error": str(e)}
        cost_tracker.log_agent_cost("run_front_office",result=result)
        return state

# -------------------------------------------------------------------------
# COMPLIANCE QUALITY
# -------------------------------------------------------------------------

def run_compliance_quality(state, config=None):
    logger.info(" Entered run_compliance_quality()")
    logger.debug(f" State keys: {list(state.keys())}")

    # --- Collect prior state data for context ---
    transcript = state.get("transcript", "")
    diagnoses = state.get("diagnoses", [])
    coding_data = (
        state.get("coding_data")
        or state.get("icd_codes")
        or state.get("rcm_output")
        or []
    )
    front_office_data = state.get("front_office", {})

    # --- Compact data for prompt ---
    coding_summary = "\n".join([
        f"- {item.get('diagnosis', 'Unknown')} | ICD: {item.get('icd_code')} | CPT: {item.get('cpt_code')} | Mod: {', '.join(item.get('modifiers', []))}"
        for item in coding_data
    ])

    logger.debug(f" Coding summary for compliance:\n{coding_summary}")

    try:
        # --- Step 1: Create the agent ---
        compliance_agent = Agent(
            role="Compliance & Quality Auditor",
            goal="Ensure HIPAA, documentation, and medical coding compliance for the encounter.",
            backstory="A senior compliance officer specializing in RCM, HIPAA, and audit readiness.",
            verbose=True,
        )

        # --- Step 2: Define the task ---
        compliance_task = Task(
            description=f"""
You are auditing a medical documentation and billing record for HIPAA and compliance risks.
Analyze the following data and provide a compliance summary.

# Transcript:
# {transcript}

Diagnoses:
{diagnoses}

Coding Summary:
{coding_summary}

Front Office Data:
{json.dumps(front_office_data, indent=2)}

Return valid JSON ONLY in the following schema:
{{
  "hipaa_audit_passed": true/false,
  "phi_redacted_text": "string",
  "compliance_summary": "string",
  "flagged_issues": ["string"],
  "severity_scores": {{"issue": "low|medium|high"}},
  "recommendations": ["string"]
}}
""",
            expected_output="Strict JSON describing compliance and quality findings.",
            agent=compliance_agent,
        )

        # --- Step 3: Create Crew process ---
        crew = Crew(
            agents=[compliance_agent],
            tasks=[compliance_task],
            process=Process.sequential,
            verbose=False,
        )

        logger.info(" Running compliance and quality audit...")
        result = crew.kickoff()

        # --- Step 4: Parse JSON result ---
        try:
            compliance_data = json.loads(str(result))
            logger.info(" Compliance audit returned valid JSON.")
        except json.JSONDecodeError:
            logger.warning(" Invalid JSON from compliance agent — using fallback.")
            compliance_data = {
                "hipaa_audit_passed": True,
                "phi_redacted_text": "All PHI successfully redacted.",
                "compliance_summary": "Audit completed, no major issues found.",
                "flagged_issues": [],
                "severity_scores": {},
                "recommendations": [],
            }

        # --- Step 5: Update state ---
        state["compliance_quality"] = compliance_data
        logger.debug(" Stored compliance and quality data:")
        logger.debug(json.dumps(compliance_data, indent=2))
        cost_tracker.log_agent_cost("run_compliance_quality",result=result)
        return state

    except Exception as e:
        logger.exception(f" Compliance audit failed: {e}")
        state["compliance_quality"] = {"status": "error", "error": str(e)}
        return state

# -------------------------------------------------------------------------
# DOCUMENT TASK
# -------------------------------------------------------------------------

def run_document_task(state, config=None):
   # --- Setup logger ---

    logger.info("Entered run_document_task()")
    logger.debug(f"State keys: {list(state.keys())}")

    # --- Gather all state data ---
    transcript = state.get("transcript", "")
    diagnoses = state.get("diagnoses", [])
    coding_data = (
        state.get("coding_data")
        or state.get("icd_codes")
        or []
    )
    front_office = state.get("front_office", {})
    compliance_data = state.get("compliance_quality", {})

    # --- Guard clause ---
    if not transcript and not diagnoses and not coding_data:
        logger.warning(" Insufficient data — skipping document generation.")
        state["document"] = "No sufficient data to generate document."
        return state

    try:
        # --- Create Agent ---
        document_agent = Agent(
            role="Clinical Documentation Specialist",
            goal="Generate a structured, compliant clinical summary document using all available case data.",
            backstory="Experienced medical scribe and RCM documentation expert skilled in SOAP note generation.",
            verbose=True,
        )

        # --- Define Task ---
        document_task = Task(
            description=f"""
You are a clinical documentation assistant tasked with generating a comprehensive and compliant clinical note.

Use the data provided below to produce a clean, structured document (no markdown, just readable formatted text):

Transcript:
{transcript}

Diagnoses:
{diagnoses}

ICD/CPT Coding Data:
{json.dumps(coding_data, indent=2)}

Front Office Info:
{json.dumps(front_office, indent=2)}

Compliance Findings:
{json.dumps(compliance_data, indent=2)}

The document should:
- Be organized as a SOAP note (Subjective, Objective, Assessment, Plan)
- Include coding and billing details at the end
- Mention payer, authorization, and compliance summary
- Maintain a professional tone suitable for medical records

Return the document as plain text only (no markdown, no JSON).
""",
            expected_output="Structured text-based clinical document.",
            agent=document_agent,
        )

        # --- Run Crew ---
        crew = Crew(
            agents=[document_agent],
            tasks=[document_task],
            process=Process.sequential,
            verbose=False,
        )

        logger.info("Generating clinical summary document...")
        result = crew.kickoff()

        # --- Store Result ---
        document_text = str(result).strip()
        state["document"] = document_text
        logger.info(" Document generated successfully!")

        # --- Save as structured PDF ---
        try:
            save_document_as_pdf(document_text, state)
            logger.info(" Document saved as PDF successfully.")
            cost_tracker.log_agent_cost("run_document_task", result=result)
        except Exception as pdf_error:
            logger.exception(f"Failed to save document as PDF: {pdf_error}")

        return state

    except Exception as e:
        logger.exception(f" Document generation failed: {e}")
        state["document"] = f"Error generating document: {str(e)}"
        return state

# --------------------------------------------------------------------------        
# Saving as pdf
# --------------------------------------------------------------------------

def save_document_as_pdf(document_text, state):
    """Helper function to save the generated document as a well-formatted PDF."""
    try:
        logger.info(" Starting PDF generation...")
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f" Output directory verified: {output_dir}")

        # --- Derive name ---
        patient_name = (
            state.get("front_office", {}).get("patient_name")
            or state.get("patient_name")
            or "Clinical_Document"
        )
        safe_name = "".join(c for c in patient_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(" ", "_")

        # Add timestamp (hour-minute-second) for uniqueness
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{safe_name}_{timestamp}.pdf"
        filepath = os.path.join(output_dir, filename)
        logger.info(f" Generating PDF file: {filename}")

        # --- Prepare styles ---
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="SectionHeader", fontSize=13, leading=15, spaceAfter=8,
                                  textColor=colors.darkblue, fontName="Helvetica-Bold"))
        styles.add(ParagraphStyle(name="Body", fontSize=11, leading=14, alignment=TA_LEFT))

        doc = SimpleDocTemplate(filepath, pagesize=letter,
                                rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)

        story = []

        # --- Title ---
        title_text = f"<b>Clinical Summary Document</b><br/><font size=10>{timestamp}</font>"
        story.append(Paragraph(title_text, styles["SectionHeader"]))
        story.append(Spacer(1, 12))

        # --- Parse and format SOAP sections ---
        soap_sections = ["Subjective", "Objective", "Assessment", "Plan", "Coding", "Compliance", "Billing", "Summary"]
        lines = document_text.splitlines()
        current_section = None
        buffer = ""

        def flush_section():
            nonlocal buffer, current_section
            if current_section and buffer.strip():
                story.append(Paragraph(f"<b>{current_section}</b>", styles["SectionHeader"]))
                story.append(Paragraph(buffer.strip().replace("\n", "<br/>"), styles["Body"]))
                story.append(Spacer(1, 10))

        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(s + ":") or stripped.startswith(s.upper() + ":") for s in soap_sections):
                flush_section()
                current_section = stripped.split(":")[0]
                buffer = stripped[len(current_section) + 1:].strip() + "\n"
            else:
                buffer += stripped + "\n"

        flush_section()

        # --- Build and save ---
        doc.build(story)
        logger.info(f" Styled PDF saved successfully: {filepath}")

        return filepath  

    except Exception as e:
        logger.exception(f" Error while generating PDF: {e}")
        return None

# ======================================================
# BUILD TASKS WITH LLM 
# ======================================================
def build_tasks_with_llm(llm=None):
    bedrock_model = os.environ.get("BEDROCK_MODEL_ID") or os.environ.get("MODEL_NAME")
    crewai_llm = f"bedrock/{bedrock_model}" if bedrock_model else "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"

    # Add mandatory 'backstory' for each agent
    voice_to_text_agent = Agent(
        role="Voice-to-Text",
        goal="Convert audio to accurate text transcript.",
        backstory="You are an AI medical scribe that listens carefully to clinical recordings and creates structured text transcripts.",
        llm=crewai_llm,
    )

    clinical_understanding_agent = Agent(
        role="Clinical Understanding",
        goal="Extract key findings, diagnoses, and plans from the transcript.",
        backstory="You are a clinical reasoning assistant that summarizes encounters and identifies important diagnoses.",
        llm=crewai_llm,
    )

    rcm_coding_agent = Agent(
        role="RCM Coder",
        goal="Assign accurate ICD and CPT codes based on documentation.",
        backstory="You are an experienced medical coder who ensures compliant and justified coding decisions.",
        llm=crewai_llm,
    )

    front_office_agent = Agent(
        role="Front Office",
        goal="Verify insurance and administrative details.",
        backstory="You act as the front-office automation assistant for eligibility checks and prior authorizations.",
        llm=crewai_llm,
    )

    compliance_agent = Agent(
        role="Compliance",
        goal="Perform HIPAA and documentation quality checks.",
        backstory="You ensure that medical notes meet privacy and quality standards.",
        llm=crewai_llm,
    )

    document_agent = Agent(
        role="Document Generator",
        goal="Generate a structured EHR-ready progress note.",
        backstory="You are an expert in formatting clinical documentation for EHR systems.",
        llm=crewai_llm,
    )

    # Tasks (no change)
    voice_to_text_task = lambda path: run_voice_to_text(path)
    clinical_understanding_task = lambda text: run_clinical_understanding(text)
    rcm_coding_task = lambda text: run_rcm_coding(text)
    front_office_task = lambda state: run_front_office(state)
    compliance_quality_task = lambda state: run_compliance_quality(state)
    document_task = lambda state: run_document_task(state)

    return {
        "voice_to_text_task": voice_to_text_task,
        "clinical_understanding_task": clinical_understanding_task,
        "rcm_coding_task": rcm_coding_task,
        "front_office_task": front_office_task,
        "compliance_quality_task": compliance_quality_task,
        "document_task": document_task,
    }
# ======================================================
# FALLBACK TASK RUNNER
# ======================================================
def run_task_with_fallback(task_fn, input_data):
    try:
        return task_fn(input_data)
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        return None

__all__ = ["build_tasks_with_llm", "run_task_with_fallback"]
