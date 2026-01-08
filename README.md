# Prior Authorization Orchestrator (CrewAI + LangGraph + ReAct Framework)

An advanced end-to-end workflow that ingests payer/clinical PDFs, extracts structured fields, and orchestrates intelligent agents for eligibility determination, policy analysis, medical necessity evaluation, authorization decisions, and comprehensive documentation. Built with **CrewAI** for agent execution, **LangGraph** for pipeline orchestration, and implements the **ReAct (Reasoning and Acting) framework** for enhanced reasoning capabilities.

## 🚀 Key Features

### **ReAct Framework Implementation**
- **Step-by-step reasoning**: Agents follow THOUGHT → ACTION → OBSERVATION cycles
- **Tool integration**: Agents can search text, validate medical codes, estimate costs, and read files
- **Dynamic planning**: Adaptive reasoning based on available information and tools
- **Fallback strategies**: 4-tier fallback system (ReAct with tools → ReAct without tools → Simple prompt → Rule-based)

### **Intelligent Data Processing**
- **Missing information inference**: Automatically fills missing fields with context-grounded, plausible values
- **ICD-10/CPT code prediction**: Infers medical codes when missing, with format validation
- **OCR ingestion**: Poppler + Tesseract via `pdf2image`/`pytesseract`; table extraction via `pdfplumber`
- **Flexible LLM providers**: OpenAI, AWS Bedrock, Groq, or Hugging Face (selectable at runtime)

### **Robust Architecture**
- **Agentic pipeline**: CrewAI agents run in a LangGraph state machine
- **Structured models**: Pydantic data models for `PatientData`, `PolicyData`, `Guidelines`, and `PAState`
- **Comprehensive logging**: Flow and error logs with detailed prompt/response tracking
- **Error resilience**: Multiple fallback mechanisms at every level

## 📁 Repository Layout

### Core Components
- `src/run.py`: CLI entrypoint for the full end-to-end pipeline
- `src/graph.py`: LangGraph pipeline nodes and `build_app()` that compiles the graph
- `src/agent_builder.py`: Builds CrewAI agents with ReAct framework and tools integration
- `src/models.py`: Pydantic models: `PatientData`, `PolicyData`, `Guidelines`, `PAState`, etc.

### Supporting Modules
- `src/cli_ocr.py`: Standalone OCR helper that saves extracted text under `Ext_text/`
- `src/extraction.py`: LLM provider selection (`_get_llm`) and PDF container-to-structure helper
- `src/pdf_ingestion.py`: OCR/text and table extraction utilities
- `src/payer_api.py`: Mock payer API for insurance status (demo fallback)
- `src/config.py`: Loads `config.yaml` and applies environment variables
- `src/logging_setup.py`: Initializes file-based logging under `Logs/`
- `src/micro_analysis.py`: Prompt/response logging and analysis utilities

### Configuration & Data
- `config.yaml`: Example configuration for AWS/LLM/OCR (template only)
- `requirements.txt`: Python dependencies including `crewai-tools`
- `ED charts/`: Example PDFs (input)
- `Ext_text/`: OCR outputs (generated)
- `Outputs/`: Final JSON results
- `Logs/`: Flow and error logs

## 🛠️ Requirements

- **Python 3.10+**
- **OS packages for OCR on Windows:**
  - **Poppler**: e.g., `C:\\poppler-24.08.0\\Library\\bin`
  - **Tesseract OCR**: e.g., `C:\\Users\\<you>\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe`

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

The app reads `.env` (via `python-dotenv`) and `config.yaml` (via `src.config.load_config`). `config.yaml` values are applied to environment variables at runtime by `apply_env_from_config`.

### LLM Provider Configuration
```yaml
llm:
  provider: openai|groq|bedrock|huggingface
  model: provider-specific model name/id
  api_key: for openai, groq, or huggingface
  inference_profile_id: optional (Bedrock)
```

### AWS Configuration
```yaml
aws:
  region: us-east-1
  access_key_id: <YOUR_KEY>
  secret_access_key: <YOUR_SECRET>
```

### Generation Settings
```yaml
generation:
  temperature: 0.1
```

### OCR Configuration
```yaml
ocr:
  poppler_path: C:\\poppler-24.08.0\\Library\\bin
  tesseract_cmd: C:\\Users\\<you>\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe
```

### Environment Variables
- `OPENAI_API_KEY`, `GROQ_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`
- `AWS_REGION` / `AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
- `MODEL_NAME` (fallback), `MODEL_TEMPERATURE`, `BEDROCK_MODEL_ID`, `BEDROCK_INFERENCE_PROFILE_ID`/`ARN`

> **Important**: Do not commit real credentials to version control. Prefer `.env` or local-only `config.yaml` and add sensitive files to your VCS ignore list.

## 🔄 How It Works

### 1. Pipeline Overview
1. `src/run.py` loads environment/config and builds the graph via `build_app()`
2. `src/graph.py` defines 6 nodes in sequence:
   - `preprocess_pdf`: OCR + table extraction → ReAct agent ingest → `PatientData`/`PolicyData`
   - `eligibility_node`: Payer API fallback → ReAct agent determines eligibility
   - `guidelines_node`: ReAct agent extracts policy guidelines
   - `necessity_node`: ReAct agent evaluates medical necessity
   - `decision_node`: ReAct agent makes final authorization decision
   - `docs_node`: ReAct agent generates comprehensive documentation

### 2. ReAct Framework Implementation
Each agent follows the ReAct pattern:
```
THOUGHT: <reasoning about what to do next>
ACTION: <specific action/tool to take, or 'none'>
OBSERVATION: <what was learned>
```

**Available Tools:**
- `search_text`: Search within provided text for queries
- `validate_medical_codes`: Validate ICD-10/CPT codes by format
- `estimate_cost_sharing`: Extract cost-sharing information from policy text
- `FileReadTool`: Read files from the filesystem

### 3. Missing Information Inference
The system automatically infers missing information:
- **Demographics**: Name, DOB, gender from text patterns
- **Insurance**: Status, plan details from coverage blocks
- **Medical Codes**: ICD-10/CPT codes from clinical context
- **Cost Sharing**: Deductibles, copays, coinsurance from policy text
- **Policy Details**: Requirements, exclusions, network rules

### 4. Fallback Strategy
4-tier fallback system ensures robust operation:
1. **ReAct with tools**: Full reasoning with tool access
2. **ReAct without tools**: Reasoning without external tools
3. **Simple prompt**: Basic LLM completion
4. **Rule-based**: Conservative heuristics when LLM fails

## 🚀 Running the Pipeline

### Full End-to-End Processing
```bash
python -m src.run "path\to\your1.pdf path\to\your2.pdf"
```

**Outputs:**
- `Outputs/<pdf_basename>.json`: Comprehensive results with eligibility, necessity, decision, guidelines, patient/policy data, and documentation
- `Outputs/<pdf_basename>.full.json`: Detailed verbose output with all inferred fields
- `Logs/flow_<timestamp>.log`: Detailed flow logs with ReAct traces
- `Logs/error_<timestamp>.log`: Error logs for troubleshooting

### OCR-Only Utility
Extract text from a PDF and preview structured data:
```bash
python -m src.cli_ocr path\to\file.pdf [lang]
```
Defaults: `lang=eng`, `dpi=300`

## 📊 Output Structure

### Main JSON Output (`<pdf_basename>.json`)
```json
{
  "patient_id": "string",
  "source_pdf": "filename.pdf",
  "eligibility": "Eligible|Not eligible: reasons",
  "decision": "Approved|Denied: reasons|Approved with note: cost-sharing",
  "necessity_summary": "Medically necessary|Not necessary: reasons",
  "policy_source": "PDF|mock",
  "documentation_summary": "Comprehensive authorization documentation...",
  "guidelines": {
    "policy_id": "string",
    "procedure_code": "string",
    "pa_required": true|false,
    "network_rules": "string",
    "cost_sharing": {},
    "source": "PDF|mock"
  },
  "patient_data": {
    "patient_id": "string",
    "demographics": {"name": "string", "dob": "string", "gender": "string"},
    "insurance_status": {"primary": {...}, "secondary": {...}},
    "network_status": "string",
    "diagnoses": ["string"],
    "procedures": ["string"],
    "urgency": "urgent|routine"
  },
  "policy_data_preview": {
    "policy_id": "string",
    "payer": "string",
    "raw_text_preview": "truncated text..."
  },
  "generated_on": "2024-01-01T12:00:00",
  "logs_count": 15,
  "inferred_fields": ["patient_data.demographics.name", "policy_data.payer"]
}
```

### Full JSON Output (`<pdf_basename>.full.json`)
Contains complete verbose data including:
- Full raw text and tables
- All inferred fields with flags
- Complete guidelines and requirements
- Detailed documentation
- All processing logs

## 🔧 ReAct Framework Details

### Agent Capabilities
Each agent is equipped with specific tools and follows ReAct reasoning:

**Eligibility Reviewer:**
- Tools: `search_text`, `validate_medical_codes`, `FileReadTool`
- ReAct: Determines coverage with dynamic policy lookup

**Policy Extractor:**
- Tools: `search_text`, `validate_medical_codes`, `estimate_cost_sharing`, `FileReadTool`
- ReAct: Extracts requirements, exclusions, cost-sharing with tool assistance

**Clinical Reviewer:**
- Tools: `search_text`, `validate_medical_codes`
- ReAct: Evaluates medical necessity with evidence-based reasoning

**Authorization Adjudicator:**
- ReAct: Synthesizes final decision with cost-sharing awareness

**Documentation Specialist:**
- ReAct: Generates comprehensive, appeal-ready documentation

### ReAct Prompt Template
```
You are a {role} following the ReAct (Reasoning and Acting) framework.

For each step, use this exact format (one block per cycle):
THOUGHT: <your reasoning about what to do next>
ACTION: <the specific action/tool to take, or 'none'>
OBSERVATION: <what you learned>

Repeat until you can conclude.

CRITICAL RULE: When inputs are missing or incomplete, infer and FILL the missing information using conservative, context-grounded values derived from the provided content (synthetic but closely related is acceptable).
- Always adhere to the expected schema, datatypes, and enumerations.
- Prefer plausible values grounded in available evidence; avoid contradictions.
- Only leave null if no reasonable inference can be made.

Specific to clinical coding: If ICD-10-CM diagnoses or CPT procedure codes are not explicitly provided, infer the most plausible codes from the clinical context.
- Favor common, general codes over overly specific ones when uncertain.
- Validate format (ICD-10 like 'E11.9', CPT as 5 digits).
- If uncertainty is extreme, you may leave null, but attempt inference first.

Available tools: {tools}
{task_header}
{body_instructions}

Final answer: {final_answer_rule}
```

## 📝 Logging & Debugging

### Log Files
- `Logs/flow_<timestamp>.log`: INFO+ level logs with ReAct traces
- `Logs/error_<timestamp>.log`: ERROR+ level logs for troubleshooting

### ReAct Trace Analysis
Enable verbose logging to see complete ReAct cycles:
```python
# In agent_builder.py, _strategy_react function
crew = Crew(agents=[react_task.agent], tasks=[react_task], process=Process.sequential, verbose=True)
```

### Micro Analysis
The system logs all prompt/response pairs in `micro_analysis/` for detailed analysis:
- `micro_analysis/<chart_name>/`: Per-PDF analysis
- `micro_analysis/UNKNOWN/`: Fallback analysis

## 🛠️ Dependencies

### Core Dependencies
- `crewai`: Agent framework with ReAct support
- `crewai-tools`: Tool integration for agents
- `langgraph`: Pipeline orchestration
- `langchain`: LLM integration
- `langchain-aws`: AWS Bedrock support
- `pydantic`: Data validation and serialization

### OCR Dependencies
- `pdfplumber`: PDF text and table extraction
- `pdf2image`: PDF to image conversion
- `pytesseract`: OCR text extraction
- `Pillow`: Image processing

### Utility Dependencies
- `python-dotenv`: Environment variable management
- `PyYAML`: Configuration file parsing

## 🔍 Troubleshooting

### Common Issues

**Tesseract not found:**
```bash
# Install Tesseract and configure
set TESSERACT_CMD=C:\Users\<you>\AppData\Local\Programs\Tesseract-OCR\tesseract.exe
```

**Poppler not found:**
```bash
# Install Poppler and configure
set POPPLER_PATH=C:\poppler-24.08.0\Library\bin
```

**LLM provider errors:**
- Ensure matching API key and model ID are set
- For Bedrock, verify region and inference profile settings
- Check provider-specific rate limits and quotas

**ReAct framework not working:**
- Verify `crewai-tools` is installed: `pip install crewai-tools`
- Check that agents have tools assigned in `agent_builder.py`
- Enable verbose logging to see ReAct traces

**Missing information not being inferred:**
- Check that the ReAct prompt includes the inference rules
- Verify agents are using the `_reactify_base_prompt` function
- Review logs for inference attempts and results

### Performance Optimization

**Large PDFs:**
- OCR runs in batches internally
- Reduce DPI if memory issues occur
- Consider pagination for very large documents

**LLM Rate Limits:**
- Implement retry logic with exponential backoff
- Use multiple provider fallbacks
- Consider request batching for multiple PDFs

## 🏗️ Development Notes

### Agent Architecture
- Agents and tasks are constructed in `src/agent_builder.py`
- ReAct framework is implemented via `_reactify_base_prompt`
- Tools are conditionally loaded based on `crewai-tools` availability
- Fallback strategies ensure robust operation

### LLM Provider Selection
`_get_llm()` tries providers in order: OpenAI → Bedrock → Groq → Hugging Face
Pin specific providers via `config.yaml`:
```yaml
llm:
  provider: openai
  model: gpt-4
```

### Mock Payer API
`src/payer_api.query_payer_api` provides demo insurance status for `P001`, `P002`, `P003` with lightweight heuristics. Replace with real payer APIs for production.

### Extending the System

**Adding New Tools:**
```python
@_maybe_tool
def your_custom_tool(param: str) -> dict:
    """Your tool description."""
    # Implementation
    return {"result": "value"}

# Add to agent tools list
tools=[t for t in [search_text, validate_medical_codes, your_custom_tool] if t]
```

**Adding New Agents:**
```python
new_agent = Agent(
    role="Your Role",
    goal="Your goal",
    backstory="Your backstory",
    llm=crewai_llm,
    tools=[t for t in [your_tools] if t],
)
```

## 📋 Example Configuration

### Complete `config.yaml` Template
```yaml
aws:
  region: us-east-1
  access_key_id: <YOUR_KEY>
  secret_access_key: <YOUR_SECRET>

llm:
  provider: openai
  model: gpt-4
  api_key: <YOUR_OPENAI_KEY>

generation:
  temperature: 0.1

ocr:
  poppler_path: C:\\poppler-24.08.0\\Library\\bin
  tesseract_cmd: C:\\Users\\<you>\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe
```

### Environment Variables Example
```bash
# .env file
OPENAI_API_KEY=sk-your-key-here
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
MODEL_TEMPERATURE=0.1
```

## 📚 Additional Documentation

- **Full End-to-End Guide**: See `docs/END_TO_END_GUIDE.md` for detailed setup, flow internals, operations, and troubleshooting
- **API Documentation**: Check individual module docstrings for detailed function documentation
- **Example Outputs**: Review `Outputs/` directory for sample JSON results

## ⚠️ Notes & Limitations

- **Policy parsing accuracy** depends on LLM provider and input quality
- **OCR quality** depends on scan quality, DPI, language packs, and Tesseract configuration
- **Mock payer API** is for demos only and should be replaced for production
- **ReAct framework** requires `crewai-tools` package for full functionality
- **Missing information inference** is conservative and context-grounded to avoid hallucinations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes with proper ReAct framework integration
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Ready to automate prior authorization with AI?** Start with `python -m src.run your_document.pdf` and watch the ReAct agents reason through your authorization workflow!