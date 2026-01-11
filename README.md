GUIDE BOOK

GenAI Based RCM Front Office Operations: AI Medical Scribe 
This guide explains how the AI Medical Scribe system ingests audio recordings, structures the data, runs an agentic workflow, and produces a complete Patient–Doctor interaction summary and documentation. It serves as a single source of truth for setup, configuration, execution, flow internals, troubleshooting, and operations.

1) System Overview

Goal:
Automate clinical documentation and coding from doctor–patient conversations using an Agentic AI pipeline. The system converts raw audio into structured clinical notes and billing-ready data, reducing manual effort and improving accuracy.

Core Frameworks:
•	LangGraph: Orchestrates the full pipeline as a state machine operating on ScribeState.
•	CrewAI: Encapsulates LLM-powered agents responsible for transcription, note generation, coding, validation, and documentation.

Key Outputs:
•	One structured JSON per consultation under /Outputs/, containing:
o	Patient and encounter data
o	SOAP notes (Subjective, Objective, Assessment, Plan)
o	ICD-10/CPT codes
o	Audit and compliance summary
o	Logs under /Logs/, tracking workflow execution and agent performance

2) Repository Components
File	Description
src/run.py	CLI entrypoint for end-to-end processing of a single audio file. Produces raw transcript <audiofile>.txt.
src/cli_ocr.py	Runs ingestion agent to produce structured JSON per new schema.
src/graph.py	LangGraph nodes, transitions, and build_app () to compile the graph.
src/agents.py	Lists all CrewAI agents, their tasks, and definitions.
src/agent_builder.py	Builds CrewAI tasks and agents; manages transcription, clinical understanding, coding, and final PDF/JSON output.
src/models.py	Pydantic models: VoiceToTextData, ClinicalUnderstandingData, RCMCodingData, FrontOfficeData, Guidelines, PipelineState, AgentState.
src/config.py	YAML + env loader and environment application.
src/logger_setup.py	File logging, stdout/stderr tee-ing.
src/config_loader.py	Config settings loader, used by Streamlit app.
src/utils.py	Utilities for audio handling, Whisper transcription, and helper functions.
src/Voice_to_text.py	Logic for transcribing audio to text format.

3) Data Model (Pydantic)

1. PatientData
•	patient_id: string
•	demographics: dict (name, age, gender, DOB, contact)
•	insurance_status: dict with primary/secondary plans
•	network_status: optional string (in/out-of-network)
•	diagnoses: list of strings (ICD-10 or clinical terms)
•	procedures: list of strings (CPT/procedure codes)
•	tests: dict of test/lab results
•	urgency: optional string (urgent or routine)

2. TranscriptionData
•	transcript: full conversation text from audio
•	confidence: float (speech-to-text accuracy)
•	timestamps: list of {word, start, end}

3. ClinicalNotes
•	subjective: patient-reported symptoms
•	objective: clinician observations or vitals
•	assessment: diagnostic reasoning summary
•	plan: treatment/follow-up recommendations
•	summary: condensed visit overview
•	entities: extracted key terms (medications, conditions, etc.)

4. CodingData
•	icd10: list of ICD-10 codes
•	cpt: list of CPT codes
•	modifiers: list of billing modifiers
•	justification: textual explanation for coding
•	coder_notes: optional internal review notes

5. ComplianceData
•	phi_redacted_text: PHI-safe transcript
•	hipaa_audit_passed: bool
•	violations: list of flagged compliance issues
•	audit_notes: optional text

6. ScribeState
•	source_audio: path/URL to audio
•	patient_data: PatientData
•	transcription: TranscriptionData
•	clinical_notes: ClinicalNotes
•	coding: CodingData
•	compliance: ComplianceData
•	structured_output: dict (formatted EHR-ready output)
•	logs: list of pipeline-level logs

4) Configuration
•	The app uses config.yaml and environment variables. At startup, config.yaml is loaded and mapped into env vars.
Sections in config.yaml:
•	aws: region + optional credentials (prefer environment variables for security)
•	llm: provider (openai|groq|bedrock|huggingface), model, optional api_key
•	generation: temperature
•	ocr: Windows paths to poppler_path and tesseract_cmd
Recognized environment variables:
•	LLM: OPENAI_API_KEY, GROQ_API_KEY, HUGGINGFACEHUB_API_TOKEN, MODEL_NAME, MODEL_TEMPERATURE
•	Bedrock: AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BEDROCK_MODEL_ID
Security note: Never commit credentials; use .env or untracked config files.

5) Prerequisites and Installation
•	Python 3.10+
•	Install dependencies:
pip install -r requirements.txt

6) Running the End-to-End Pipeline
Run a single audio file:
python -m src.run audio_recordings/CAR0001.mp3

Outputs:
•	PDF under Outputs/ named Clinical_Document_<datetime>.pdf containing:
o	SUBJECTIVE: Past Medical, Social, Family History
o	OBJECTIVE: Required documentation elements
o	ASSESSMENT: Diagnoses and impressions
o	PLAN: Diagnostic studies, consultations, authorizations
o	COMPLIANCE NOTES
o	BILLING AND CODING SUMMARY: CPT/ICD-10 codes

Logs:
•	Logs/flow_<timestamp>.log (INFO+)
•	Logs/error_<timestamp>.log (ERROR+)
•	Logs/Modules/agent_builder.log – detailed per-agent processing

7) Execution Flow
Node	Input	Actions	Output
*. voice_to_text_task	PipelineState.source_audio	Transcribes audio → raw text + timestamps	state.transcription
*. clinical_understanding_task	state.transcription	Extracts symptoms, conditions, assessment, plan	state.clinical_notes
*. rcm_coding_task	state.clinical_notes	Assigns ICD-10/CPT codes, modifiers, justification	state.coding
*. front_office_task	state.coding	Checks eligibility, payer, workflow tasks	state.patient_data
*. compliance_quality_task	state.patient_data	PHI redaction, HIPAA compliance, audit	state.compliance
*. document_task	All state	Generates JSON/PDF documentation	state.structured_output

Pipeline sequence:
START → voice_to_text → clinical_understanding → rcm_coding → front_office → compliance_quality → document → END

8) Extensibility
•	Adding a node: Define node_<name>(state: ScribeState, tasks) → register in graph.add_node() → add edges.
•	Modifying CrewAI tasks: Update build_tasks_with_llm() → specify output_pydantic for structured validation.

9) Operational Runbook
•	Logs: Flow/error logs per run under /Logs/; stdout/stderr tee’d.
•	Idempotency: Timestamped filenames prevent overwriting.
•	Performance: Chunk audio; smaller ASR models for early tasks.
•	Concurrency: CLI serial; parallel processing possible with threads/process pools.
•	Failure Handling: Partial state fallback, JSON normalization for downstream nodes.

10) Troubleshooting
•	Audio transcription fails → check audio path, format, FFmpeg, ASR model.
•	No LLM provider → install dependencies, set API keys.
•	CrewAI non-JSON output → trim {...}; check logs.
•	Missing coding output → inspect logs, re-run nodes.
•	Large audio files → chunk or stream; smaller models.
•	PDF/report errors → check output dir and Unicode safety.

11) Security & Compliance
•	Credentials: use env-managed secrets, IAM roles.
•	Data/Logging: avoid PHI, encrypt, implement retention policies.
•	Compliance: ensure HIPAA/local regulation compliance; audit compliance_findings.

12) Known Limitations
•	Accuracy depends on ASR/LLM quality; noisy audio may degrade notes.
•	LLM-assisted coding may miss rare cases.
•	Narrative output probabilistic; production requires rule validation.
•	Network/API availability affects speed/reliability.

13) Quickstart Checklist
1.	Install dependencies: pip install -r requirements.txt
2.	Set LLM provider credentials and model
3.	Run: python -m src.run <audio-file path>
4.	Review outputs under /Outputs/ and /Logs/

14) Streamlit Application
Purpose: Web interface for uploading audio, running pipeline, and viewing outputs.
Setup:
pip install streamlit
Launch:
streamlit run app.py
Workflow:
1.	Upload .wav, .mp3, or .m4a audio.
2.	Press Run AI Scribe Pipeline.
3.	View outputs: transcript, SOAP notes, ICD-10/CPT coding, compliance summary.
4.	Download results as JSON/PDF for EHR integration.
File Management:
•	Uploaded audio saved with timestamp: temp_audio/<filename>_YYYY-MM-DD_HH-MM-SS.wav
•	Outputs saved under /Outputs/ with timestamped filenames
Logging: All steps logged in /Logs/; real-time progress displayed in UI
Extensibility: Batch uploads, parallel processing, cloud storage integration, dynamic ASR/LLM models







