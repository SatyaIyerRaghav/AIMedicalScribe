# src/graph.py

from langgraph.graph import StateGraph, END
from src.models import PipelineState

from src.agent_builder import (
    run_voice_to_text,
    run_clinical_understanding,
    run_rcm_coding,
    run_front_office,
    run_compliance_quality,
    run_document_task,
)

# ✅ Create a new LangGraph
graph = StateGraph(PipelineState)

# ✅ Register all your nodes (each step of the pipeline)
graph.add_node("voice_to_text_task", run_voice_to_text)
graph.add_node("clinical_understanding_task", run_clinical_understanding)
graph.add_node("rcm_coding_task", run_rcm_coding)
graph.add_node("front_office_task", run_front_office)
graph.add_node("compliance_quality_task", run_compliance_quality)
graph.add_node("document_task", run_document_task)

# ✅ Define the flow
graph.set_entry_point("voice_to_text_task")
graph.add_edge("voice_to_text_task", "clinical_understanding_task")
graph.add_edge("clinical_understanding_task", "rcm_coding_task")
graph.add_edge("rcm_coding_task", "front_office_task")
graph.add_edge("front_office_task", "compliance_quality_task")
graph.add_edge("compliance_quality_task", "document_task")
graph.add_edge("document_task", END)

# ✅ Compile the graph into an executable app
app = graph.compile()

print("✅ LangGraph pipeline built successfully.")
