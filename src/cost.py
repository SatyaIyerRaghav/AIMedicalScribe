"""
cost.py - Comprehensive Token and Cost Tracking Module for Claude API

Usage:
    from cost import CostTracker
    
    tracker = CostTracker()
    
    # After agent execution
    result = crew.kickoff()
    tracker.log_agent_cost(
        agent_name="run_clinical_understanding",
        result=result
    )
    
    # Print summary at end
    tracker.print_summary()
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path

# Setup logger
logger = logging.getLogger("cost_tracker")


class CostTracker:
    """
    Comprehensive cost and token tracking for Claude API calls via CrewAI.
    """
    
    # Claude API Pricing (per 1K tokens) - Update as needed
    CLAUDE_PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-opus-20240229-v1:0": {"input": 0.015, "output": 0.075},
    }
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the cost tracker.
        
        Args:
            model_name: Override model name (otherwise auto-detected from env)
        """
        self.model_name = model_name or self._get_model_from_env()
        
        # Agent-specific tracking
        self.agent_costs: Dict[str, Dict] = {}
        
        # Session-wide tracking
        self.session_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "model_used": self.model_name,
            "start_time": time.time(),
            "end_time": None,
            "total_agents_run": 0
        }
        
        logger.info(f"🚀 CostTracker initialized with model: {self.model_name}")
    
    def _get_model_from_env(self) -> str:
        """Get model name from environment variables."""
        model = (
            os.environ.get("BEDROCK_MODEL_ID") or 
            os.environ.get("MODEL_NAME") or 
            "claude-3-5-sonnet-20240620"
        )
        
        # Clean bedrock prefix if present
        if "/" in model:
            model = model.split("/")[-1]
        
        return model
    
    def _find_pricing(self, model_name: str) -> Dict[str, float]:
        """
        Find pricing for a given model name.
        
        Args:
            model_name: Name of the Claude model
            
        Returns:
            Dictionary with 'input' and 'output' pricing per 1K tokens
        """
        # Clean model name
        clean_model = model_name.replace("bedrock/", "")
        
        # Direct match
        if clean_model in self.CLAUDE_PRICING:
            return self.CLAUDE_PRICING[clean_model]
        
        # Partial match
        for key, pricing in self.CLAUDE_PRICING.items():
            if key in clean_model or clean_model in key:
                return pricing
        
        # Default to Sonnet pricing
        logger.warning(f" No pricing found for {model_name}, using default Sonnet pricing")
        return self.CLAUDE_PRICING["claude-3-5-sonnet-20240620"]
    
    def extract_tokens(self, result: Any) -> Dict[str, int]:
        """
        Extract token usage from CrewAI/LiteLLM result.
        
        Supports multiple response formats:
        - result.token_usage
        - result.llm_response.usage
        - result.usage
        - dict format
        - result.metadata
        
        Args:
            result: CrewAI kickoff result or LLM response
            
        Returns:
            Dictionary with 'input_tokens', 'output_tokens', 'total_tokens'
        """
        tokens = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        
        try:
            # Method 1: token_usage attribute (CrewAI standard)
            if hasattr(result, "token_usage"):
                usage = result.token_usage
                tokens["input_tokens"] = (
                    getattr(usage, "prompt_tokens", 0) or 
                    getattr(usage, "input_tokens", 0)
                )
                tokens["output_tokens"] = (
                    getattr(usage, "completion_tokens", 0) or 
                    getattr(usage, "output_tokens", 0)
                )
                tokens["total_tokens"] = getattr(usage, "total_tokens", 0)
                
                if tokens["total_tokens"] > 0:
                    logger.debug(f"Extracted tokens from token_usage: {tokens}")
                    return tokens
            
            # Method 2: llm_response wrapper
            if hasattr(result, "llm_response") and hasattr(result.llm_response, "usage"):
                usage = result.llm_response.usage
                tokens["input_tokens"] = (
                    getattr(usage, "prompt_tokens", 0) or 
                    getattr(usage, "input_tokens", 0)
                )
                tokens["output_tokens"] = (
                    getattr(usage, "completion_tokens", 0) or 
                    getattr(usage, "output_tokens", 0)
                )
                tokens["total_tokens"] = getattr(usage, "total_tokens", 0)
                
                if tokens["total_tokens"] > 0:
                    logger.debug(f"Extracted tokens from llm_response.usage: {tokens}")
                    return tokens
            
            # Method 3: Direct usage attribute
            if hasattr(result, "usage"):
                usage = result.usage
                tokens["input_tokens"] = (
                    getattr(usage, "prompt_tokens", 0) or 
                    getattr(usage, "input_tokens", 0)
                )
                tokens["output_tokens"] = (
                    getattr(usage, "completion_tokens", 0) or 
                    getattr(usage, "output_tokens", 0)
                )
                tokens["total_tokens"] = getattr(usage, "total_tokens", 0)
                
                if tokens["total_tokens"] > 0:
                    logger.debug(f"Extracted tokens from usage: {tokens}")
                    return tokens
            
            # Method 4: Dictionary format
            if isinstance(result, dict):
                usage = result.get("usage", {})
                tokens["input_tokens"] = (
                    usage.get("prompt_tokens", 0) or 
                    usage.get("input_tokens", 0)
                )
                tokens["output_tokens"] = (
                    usage.get("completion_tokens", 0) or 
                    usage.get("output_tokens", 0)
                )
                tokens["total_tokens"] = usage.get("total_tokens", 0)
                
                if tokens["total_tokens"] > 0:
                    logger.debug(f"Extracted tokens from dict: {tokens}")
                    return tokens
            
            # Method 5: Metadata
            if hasattr(result, "metadata") and isinstance(result.metadata, dict):
                usage = result.metadata.get("usage", {})
                tokens["input_tokens"] = (
                    usage.get("prompt_tokens", 0) or 
                    usage.get("input_tokens", 0)
                )
                tokens["output_tokens"] = (
                    usage.get("completion_tokens", 0) or 
                    usage.get("output_tokens", 0)
                )
                tokens["total_tokens"] = usage.get("total_tokens", 0)
                
                if tokens["total_tokens"] > 0:
                    logger.debug(f"Extracted tokens from metadata: {tokens}")
                    return tokens
            
            logger.warning(f"Could not extract token usage. Result type: {type(result)}")
            
        except Exception as e:
            logger.error(f"Token extraction failed: {e}", exc_info=True)
        
        return tokens
    
    def calculate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int,
        model_name: Optional[str] = None
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate cost based on token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Optional model name override
            
        Returns:
            Dictionary with cost breakdown
        """
        model = model_name or self.model_name
        pricing = self._find_pricing(model)
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "model": model
        }
    
    def log_agent_cost(
        self,
        agent_name: str,
        result: Optional[Any] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        fixed_cost: float = 0.0,
        model_name: Optional[str] = None
    ):
        """
        Log cost for an agent execution.
        
        Args:
            agent_name: Name of the agent (e.g., "run_clinical_understanding")
            result: CrewAI result object (for auto token extraction)
            input_tokens: Manual input token count (if result not provided)
            output_tokens: Manual output token count (if result not provided)
            fixed_cost: Additional fixed cost to add
            model_name: Optional model name override
        """
        # Extract tokens from result if provided
        if result is not None:
            extracted = self.extract_tokens(result)
            input_tokens = extracted["input_tokens"]
            output_tokens = extracted["output_tokens"]
        
        # Calculate cost
        cost_breakdown = self.calculate_cost(input_tokens, output_tokens, model_name)
        total_cost = cost_breakdown["total_cost"] + fixed_cost
        
        # Initialize agent tracking if needed
        if agent_name not in self.agent_costs:
            self.agent_costs[agent_name] = {
                "tokens": {"input": 0, "output": 0},
                "cost": 0.0,
                "runs": 0
            }
        
        # Update agent costs
        self.agent_costs[agent_name]["tokens"]["input"] += input_tokens
        self.agent_costs[agent_name]["tokens"]["output"] += output_tokens
        self.agent_costs[agent_name]["cost"] += total_cost
        self.agent_costs[agent_name]["runs"] += 1
        
        # Update session stats
        self.session_stats["total_input_tokens"] += input_tokens
        self.session_stats["total_output_tokens"] += output_tokens
        self.session_stats["total_cost"] += total_cost
        self.session_stats["total_agents_run"] += 1
        
        # Log details
        logger.info(f"{'='*70}")
        logger.info(f"{agent_name} Cost Breakdown:")
        logger.info(f"   Input tokens:  {input_tokens:,} (${cost_breakdown['input_cost']:.6f})")
        logger.info(f"   Output tokens: {output_tokens:,} (${cost_breakdown['output_cost']:.6f})")
        if fixed_cost > 0:
            logger.info(f"   Fixed cost: ${fixed_cost:.6f}")
        logger.info(f"    Total cost: ${total_cost:.6f}")
        logger.info(f"    Model: {cost_breakdown['model']}")
        logger.info(f"{'='*70}")
    
    def print_summary(self) -> Dict[str, Any]:
        """
        Print comprehensive cost summary.
        
        Returns:
            Dictionary with complete cost breakdown
        """
        self.session_stats["end_time"] = time.time()
        duration = self.session_stats["end_time"] - self.session_stats["start_time"]
        
        logger.info("\n" + "="*80)
        logger.info(" DETAILED AGENT RUN COST SUMMARY")
        logger.info("="*80)
        
        # Agent breakdown
        for agent, data in self.agent_costs.items():
            tokens = data["tokens"]
            cost = data["cost"]
            total_tokens = tokens["input"] + tokens["output"]
            runs = data["runs"]
            
            if total_tokens > 0 or cost > 0:
                logger.info(f"\n {agent}:")
                logger.info(f"    Runs:    {runs}")
                logger.info(f"    Input:   {tokens['input']:,} tokens")
                logger.info(f"    Output:  {tokens['output']:,} tokens")
                logger.info(f"    Total:   {total_tokens:,} tokens")
                logger.info(f"    Cost:    ${cost:.6f}")
                if runs > 1:
                    logger.info(f"   Avg/run: ${cost/runs:.6f}")
        
        # Session totals
        total_tokens = (
            self.session_stats['total_input_tokens'] + 
            self.session_stats['total_output_tokens']
        )
        
        logger.info("\n" + "-"*80)
        logger.info(f"SESSION TOTALS:")
        logger.info(f"   Total input tokens:  {self.session_stats['total_input_tokens']:,}")
        logger.info(f"   Total output tokens: {self.session_stats['total_output_tokens']:,}")
        logger.info(f"   Total tokens:        {total_tokens:,}")
        logger.info(f"   TOTAL COST:          ${self.session_stats['total_cost']:.6f}")
        logger.info(f"   Model used:          {self.session_stats['model_used']}")
        logger.info(f"   Agents run:          {self.session_stats['total_agents_run']}")
        logger.info(f"   Duration:            {duration:.2f} seconds")
        
        if total_tokens > 0:
            cost_per_1k = (self.session_stats['total_cost'] / total_tokens) * 1000
            logger.info(f"   Cost per 1K tokens:  ${cost_per_1k:.6f}")
        
        logger.info("="*80 + "\n")
        
        return {
            "agent_breakdown": self.agent_costs,
            "session_totals": self.session_stats,
            "duration_seconds": duration
        }
    
    def export_to_json(self, filepath: str = "cost_report.json"):
        """
        Export cost data to JSON file.
        
        Args:
            filepath: Path to save JSON report
        """
        self.session_stats["end_time"] = time.time()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "session": self.session_stats,
            "agents": self.agent_costs
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f" Cost report exported to: {filepath}")
        return filepath
    
    def export_to_csv(self, filepath: str = "cost_report.csv"):
        """
        Export cost data to CSV file.
        
        Args:
            filepath: Path to save CSV report
        """
        import csv
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Agent", "Runs", "Input Tokens", "Output Tokens", 
                "Total Tokens", "Cost ($)"
            ])
            
            for agent, data in self.agent_costs.items():
                tokens = data["tokens"]
                total = tokens["input"] + tokens["output"]
                writer.writerow([
                    agent,
                    data["runs"],
                    tokens["input"],
                    tokens["output"],
                    total,
                    f"{data['cost']:.6f}"
                ])
            
            # Add totals row
            writer.writerow([])
            writer.writerow([
                "TOTAL",
                self.session_stats["total_agents_run"],
                self.session_stats["total_input_tokens"],
                self.session_stats["total_output_tokens"],
                self.session_stats["total_input_tokens"] + self.session_stats["total_output_tokens"],
                f"{self.session_stats['total_cost']:.6f}"
            ])
        
        logger.info(f" Cost report exported to: {filepath}")
        return filepath
    
    def reset(self):
        """Reset all tracking data."""
        self.agent_costs = {}
        self.session_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "model_used": self.model_name,
            "start_time": time.time(),
            "end_time": None,
            "total_agents_run": 0
        }
        logger.info(" Cost tracker reset")


# Singleton instance for easy import
_global_tracker: Optional[CostTracker] = None


def get_tracker(model_name: Optional[str] = None) -> CostTracker:
    """
    Get or create the global cost tracker instance.
    
    Args:
        model_name: Optional model name for initialization
        
    Returns:
        CostTracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker(model_name=model_name)
    return _global_tracker


# Convenience functions for direct use
def log_cost(agent_name: str, result: Any = None, **kwargs):
    """Convenience wrapper for logging cost."""
    tracker = get_tracker()
    tracker.log_agent_cost(agent_name, result, **kwargs)


def print_summary() -> Dict[str, Any]:
    """Convenience wrapper for printing summary."""
    tracker = get_tracker()
    return tracker.print_summary()


def export_json(filepath: str = "cost_report.json"):
    """Convenience wrapper for JSON export."""
    tracker = get_tracker()
    return tracker.export_to_json(filepath)


def export_csv(filepath: str = "cost_report.csv"):
    """Convenience wrapper for CSV export."""
    tracker = get_tracker()
    return tracker.export_to_csv(filepath)