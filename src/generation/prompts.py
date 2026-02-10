"""Prompt management for generation."""

from pathlib import Path
from typing import Optional

from src.utils.io_utils import load_yaml, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptManager:
    """Manager for loading and formatting prompts."""
    
    def __init__(self, prompts_dir: Optional[str | Path] = None):
        """
        Initialize prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt files
        """
        if prompts_dir is None:
            prompts_dir = get_project_root() / "src" / "prompts"
        
        self.prompts_dir = Path(prompts_dir)
        self._prompts_cache = {}
    
    def load_answerer_prompts(self) -> dict[str, dict]:
        """Load answerer prompts from YAML file."""
        if "answerer" in self._prompts_cache:
            return self._prompts_cache["answerer"]
        
        prompts_file = self.prompts_dir / "answerer_prompts.yaml"
        
        if not prompts_file.exists():
            logger.warning(f"Prompts file not found: {prompts_file}")
            return {}
        
        prompts = load_yaml(prompts_file)
        self._prompts_cache["answerer"] = prompts
        return prompts
    
    def get_prompt_template(self, prompt_id: str) -> str:
        """Get a specific prompt template by ID."""
        prompts = self.load_answerer_prompts()
        
        if prompt_id not in prompts:
            raise ValueError(f"Unknown prompt ID: {prompt_id}")
        
        return prompts[prompt_id]["template"]
    
    def get_prompt_info(self, prompt_id: str) -> dict:
        """Get full prompt info including description."""
        prompts = self.load_answerer_prompts()
        
        if prompt_id not in prompts:
            raise ValueError(f"Unknown prompt ID: {prompt_id}")
        
        return prompts[prompt_id]
    
    def list_prompts(self) -> list[str]:
        """List available prompt IDs."""
        prompts = self.load_answerer_prompts()
        return list(prompts.keys())
    
    def format_prompt(
        self,
        prompt_id: str,
        question: str,
        context: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Format a prompt template with the given values.
        
        Args:
            prompt_id: Prompt template ID
            question: The medical question
            context: Optional context (for RAG)
            **kwargs: Additional template variables
            
        Returns:
            Formatted prompt string
        """
        template = self.get_prompt_template(prompt_id)
        
        # Build format dict
        format_dict = {"question": question, **kwargs}
        
        if context is not None:
            format_dict["context"] = context
        
        try:
            return template.format(**format_dict)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            raise


def load_judge_prompt() -> str:
    """Load the LLM judge prompt template."""
    prompts_dir = get_project_root() / "src" / "prompts"
    judge_file = prompts_dir / "judge_prompt.txt"
    
    if not judge_file.exists():
        raise FileNotFoundError(f"Judge prompt not found: {judge_file}")
    
    return judge_file.read_text(encoding="utf-8")


def load_rag_prompt() -> str:
    """Load the RAG prompt template."""
    prompts_dir = get_project_root() / "src" / "prompts"
    rag_file = prompts_dir / "rag_prompt.txt"
    
    if not rag_file.exists():
        raise FileNotFoundError(f"RAG prompt not found: {rag_file}")
    
    return rag_file.read_text(encoding="utf-8")


# Global prompt manager instance
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """Get or create the global prompt manager."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
