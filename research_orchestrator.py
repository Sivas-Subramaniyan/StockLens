"""
LangGraph Orchestration for Complete Research Workflow
"""

import os
import sys
from typing import TypedDict

# Fix Windows encoding issues
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback if langgraph not available - use simple sequential execution
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not available. Using sequential execution.")

from company_selector import CompanySelector
from research_agent import ResearchAgent
from summarization_agent import SummarizationAgent


class ResearchState(TypedDict):
    """State for the research workflow"""
    company_name: str
    financial_data: dict
    research_output_dir: str
    research_files: dict
    research_data: dict  # Full research data
    analyst_report: str
    validation_result: dict
    report_file_path: str
    error: str


class ResearchOrchestrator:
    """LangGraph orchestration for complete research workflow"""
    
    def __init__(
        self,
        tavily_api_key: str,
        openai_api_key: str,
        research_output_dir: str = "research_output",
        reports_dir: str = "reports"
    ):
        """
        Initialize orchestrator.
        
        Args:
            tavily_api_key: Tavily API key
            openai_api_key: OpenAI API key
            research_output_dir: Directory for research outputs
            reports_dir: Directory for final reports
        """
        self.tavily_api_key = tavily_api_key
        self.openai_api_key = openai_api_key
        self.research_output_dir = research_output_dir
        self.reports_dir = reports_dir
        
        # Initialize agents
        self.company_selector = CompanySelector()
        self.research_agent = ResearchAgent(
            tavily_api_key=tavily_api_key,
            output_dir=research_output_dir
        )
        self.summarization_agent = SummarizationAgent(
            openai_api_key=openai_api_key,
            model="gpt-4o-mini"  # Using gpt-4o-mini
        )
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        
        if not LANGGRAPH_AVAILABLE:
            return None  # Will use sequential execution
        
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("select_company", self._select_company_node)
        workflow.add_node("run_research", self._run_research_node)
        workflow.add_node("summarize_research", self._summarize_research_node)
        workflow.add_node("validate_decision", self._validate_decision_node)
        workflow.add_node("save_report", self._save_report_node)
        
        # Define edges
        workflow.set_entry_point("select_company")
        workflow.add_edge("select_company", "run_research")
        workflow.add_edge("run_research", "summarize_research")
        workflow.add_edge("summarize_research", "validate_decision")
        workflow.add_edge("validate_decision", "save_report")
        workflow.add_edge("save_report", END)
        
        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _select_company_node(self, state: ResearchState) -> ResearchState:
        """Node: Select company from ranked_companies.csv"""
        print("\n" + "="*60)
        print("STEP 1: Company Selection")
        print("="*60)
        
        try:
            # Interactive selection
            company_data = self.company_selector.interactive_select()
            
            if not company_data:
                state["error"] = "Company selection cancelled"
                return state
            
            state["company_name"] = company_data["company_name"]
            state["financial_data"] = company_data["financial_data"]
            state["research_output_dir"] = self.research_output_dir
            
            print(f"\n[OK] Selected: {state['company_name']}")
            print(f"  Rank: {state['financial_data'].get('rank', 'N/A')}")
            print(f"  Investment Score: {state['financial_data'].get('investment_score', 'N/A')}")
            
            return state
            
        except Exception as e:
            state["error"] = f"Error in company selection: {str(e)}"
            return state
    
    def _run_research_node(self, state: ResearchState) -> ResearchState:
        """Node: Run research agent"""
        print("\n" + "="*60)
        print("STEP 2: Running Research Agent")
        print("="*60)
        
        try:
            saved_files = self.research_agent.run_research(
                company_name=state["company_name"],
                financial_data=state["financial_data"]
            )
            
            state["research_files"] = saved_files
            
            print(f"\n[OK] Research completed for {state['company_name']}")
            print(f"  Categories processed: {len([f for f in saved_files.values() if f])}")
            
            return state
            
        except Exception as e:
            state["error"] = f"Error in research: {str(e)}"
            return state
    
    def _summarize_research_node(self, state: ResearchState) -> ResearchState:
        """Node: Generate analyst report from research outputs"""
        print("\n" + "="*60)
        print("STEP 3: Generating Analyst Report")
        print("="*60)
        
        try:
            # Load research data
            research_data = self.summarization_agent.load_research_outputs(
                research_output_dir=state["research_output_dir"],
                company_name=state["company_name"]
            )
            
            if not research_data:
                state["error"] = "No research data found to summarize"
                return state
            
            # Store research data in state
            state["research_data"] = research_data
            
            # Generate analyst report using full research data
            print("\nGenerating analyst report from research data...")
            report = self.summarization_agent.create_analyst_report(
                company_name=state["company_name"],
                financial_data=state["financial_data"],
                research_data=research_data
            )
            
            state["analyst_report"] = report
            
            print(f"\n[OK] Analyst report generated for {state['company_name']}")
            
            return state
            
        except Exception as e:
            state["error"] = f"Error in summarization: {str(e)}"
            return state
    
    def _validate_decision_node(self, state: ResearchState) -> ResearchState:
        """Node: Validate buy/avoid decision"""
        print("\n" + "="*60)
        print("STEP 4: Validating Buy/Avoid Decision")
        print("="*60)
        
        try:
            # Use research data for validation
            research_data = state.get("research_data", {})
            
            if not research_data:
                # Fallback: load if not already done
                research_data = self.summarization_agent.load_research_outputs(
                    research_output_dir=state["research_output_dir"],
                    company_name=state["company_name"]
                )
                state["research_data"] = research_data
            
            # Validate decision using full research data
            validation = self.summarization_agent.validate_buy_avoid(
                company_name=state["company_name"],
                financial_data=state["financial_data"],
                research_data=research_data,
                report=state["analyst_report"]
            )
            
            state["validation_result"] = validation
            
            print(f"\n[OK] Validation completed")
            print(f"  Recommendation: {validation.get('recommendation', 'N/A')}")
            print(f"  Confidence: {validation.get('confidence', 'N/A')}")
            print(f"  Expected Return: {validation.get('expected_return_3y', 'N/A')}")
            
            return state
            
        except Exception as e:
            state["error"] = f"Error in validation: {str(e)}"
            return state
    
    def _save_report_node(self, state: ResearchState) -> ResearchState:
        """Node: Save final report"""
        print("\n" + "="*60)
        print("STEP 5: Saving Final Report")
        print("="*60)
        
        try:
            filepath, tldr = self.summarization_agent.save_report(
                company_name=state["company_name"],
                report=state["analyst_report"],
                validation=state["validation_result"],
                output_dir=self.reports_dir
            )

            state["report_file_path"] = filepath
            state["tldr"] = tldr

            print(f"\n[OK] Final report saved: {filepath}")
            
            return state
            
        except Exception as e:
            state["error"] = f"Error saving report: {str(e)}"
            return state
    
    def run(self, config: dict = None):
        """
        Run the complete research workflow.
        
        Args:
            config: Optional configuration for LangGraph
            
        Returns:
            Final state
        """
        initial_state: ResearchState = {
            "company_name": "",
            "financial_data": {},
            "research_output_dir": self.research_output_dir,
            "research_files": {},
            "research_data": {},
            "analyst_report": "",
            "validation_result": {},
            "report_file_path": "",
            "error": ""
        }
        
        print("\n" + "#"*60)
        print("RESEARCH WORKFLOW ORCHESTRATOR")
        print("#"*60)
        
        # Run the graph or sequential execution
        if LANGGRAPH_AVAILABLE and self.graph:
            if config is None:
                config = {"configurable": {"thread_id": "1"}}
            
            # Run using LangGraph
            final_state = None
            for state in self.graph.stream(initial_state, config):
                final_state = state
                # Check for errors
                if "error" in state and state["error"]:
                    print(f"\n[ERROR] {state['error']}")
                    break
        else:
            # Sequential execution (fallback)
            print("Using sequential execution (LangGraph not available)\n")
            state = initial_state
            
            # Execute nodes sequentially
            state = self._select_company_node(state)
            if state.get("error"):
                return state
            
            state = self._run_research_node(state)
            if state.get("error"):
                return state
            
            state = self._summarize_research_node(state)
            if state.get("error"):
                return state
            
            state = self._validate_decision_node(state)
            if state.get("error"):
                return state
            
            state = self._save_report_node(state)
            final_state = state
        
        if final_state:
            print("\n" + "#"*60)
            print("WORKFLOW COMPLETED")
            print("#"*60)
            
            if final_state.get("error"):
                print(f"Error: {final_state['error']}")
            else:
                print(f"Company: {final_state.get('company_name', 'N/A')}")
                print(f"Recommendation: {final_state.get('validation_result', {}).get('recommendation', 'N/A')}")
                print(f"Report: {final_state.get('report_file_path', 'N/A')}")
        
        return final_state


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Research Workflow Orchestrator")
    parser.add_argument("--tavily-key", required=True, help="Tavily API key")
    parser.add_argument("--openai-key", required=True, help="OpenAI API key")
    parser.add_argument("--research-dir", default="research_output", help="Research output directory")
    parser.add_argument("--reports-dir", default="reports", help="Reports output directory")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = ResearchOrchestrator(
        tavily_api_key=args.tavily_key,
        openai_api_key=args.openai_key,
        research_output_dir=args.research_dir,
        reports_dir=args.reports_dir
    )
    
    # Run workflow
    orchestrator.run()


if __name__ == "__main__":
    main()

