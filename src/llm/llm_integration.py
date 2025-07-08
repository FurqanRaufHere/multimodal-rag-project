"""
LLM Integration Module for Mistral Model
Supports advanced prompting techniques like Chain-of-Thought, zero-shot, few-shot
"""

import os
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class MistralLLM:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1", device: str = None):
        """
        Initialize Mistral LLM model and tokenizer
        
        Args:
            model_name: Hugging Face model name for Mistral
            device: Device to run model on ("cuda" or "cpu"). Auto-detect if None.
        """
        self.api_key = os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables.")
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        self.model.to(self.device)
        # Note: If using API key for remote API, pipeline initialization would differ.
        # Here we keep local model loading; API key usage depends on actual deployment.
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device=="cuda" else -1)
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate response from Mistral model given a prompt
        
        Args:
            prompt: Input prompt string
            max_length: Maximum length of generated tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
        
        Returns:
            Generated text response
        """
        outputs = self.generator(prompt, max_length=max_length, temperature=temperature, top_p=top_p, do_sample=True)
        return outputs[0]['generated_text']
    
    def generate_cot_response(self, prompt: str, cot_prefix: str = "Let's think step by step.", **kwargs) -> str:
        """
        Generate Chain-of-Thought (CoT) style response
        
        Args:
            prompt: Input prompt string
            cot_prefix: Prefix to encourage CoT reasoning
        
        Returns:
            Generated CoT response text
        """
        cot_prompt = cot_prefix + "\n" + prompt
        return self.generate_response(cot_prompt, **kwargs)

    def generate_zero_shot(self, prompt: str, **kwargs) -> str:
        """
        Generate zero-shot response
        
        Args:
            prompt: Input prompt string
        
        Returns:
            Generated zero-shot response text
        """
        return self.generate_response(prompt, **kwargs)

    def generate_few_shot(self, examples: list, prompt: str, **kwargs) -> str:
        """
        Generate few-shot response by prepending examples to prompt
        
        Args:
            examples: List of example strings (input-output pairs)
            prompt: Input prompt string
        
        Returns:
            Generated few-shot response text
        """
        few_shot_prompt = "\n".join(examples) + "\n" + prompt
        return self.generate_response(few_shot_prompt, **kwargs)
