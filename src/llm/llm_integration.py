import requests
import os

class MistralLLM:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_response(self, query: str, context: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        system_prompt = (
        "You are a highly intelligent, structured, and helpful assistant. Your job is to answer user queries using ONLY the information found in the provided context from the knowledge base.\n\n"

        "Always organize your responses in a well-structured and user-friendly format. Do NOT use outside or general knowledge under any circumstance.\n\n"
        "- If the context only partially matches the query, try to summarize what's closest and mention that it's not a direct answer."
        
        "---\n"
        "📘 Summary:\n"
        "- Start with a short paragraph that explains the topic based only on the context.\n\n"

        "🔑 Key Points:\n"
        "- Use 3 to5 bullet points.\n"
        "- Begin each with a **bolded heading** followed by a concise explanation.\n\n"

        "💡 Suggestions and Observations:\n"
        "- Point out any gaps, missing info, or areas where the context lacks depth.\n"
        "- Suggest consulting external resources if needed.\n\n"

        "📂 Source:\n"
        "- Include the document name or section, if available. Otherwise, write 'Source not specified.'\n\n"

        "---\n"
        "📏 Style Guidelines:\n"
        "- Be brief where possible, detailed where needed.\n"
        "- Avoid repeating the context word-for-word — summarize and synthesize only what’s relevant.\n"
        "- Use a respectful, intelligent tone with no fluff.\n"
        "- Be user-focused, clear, and helpful.\n\n"

        "❗ Important Instructions:\n"
        "- DO NOT guess or use any external/general knowledge.\n"
        "- If the context does NOT contain enough relevant information to answer the query:\n"
        "- Do NOT respond with 'Sorry, this topic is outside the scope of the provided knowledge base.' unless absolutely no relevant information is found.\n"
        "- Instead, try to provide the best possible answer based on the context, even if partial.\n\n"
        )


        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistral-medium",  # Adjust based on what you're using
            "messages": [
                {"role": "system", "content": system_prompt + f"\n\nContext:\n{context}"},
                {"role": "user", "content": query}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }

        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"API request failed with status {response.status_code}: {response.text}")

        return response.json()['choices'][0]['message']['content']    
    
    # Generate responses with different prompting strategies
    def generate_cot_response(self, prompt: str, cot_prefix: str = "Let's think step by step.", **kwargs) -> str:
        cot_prompt = cot_prefix + "\n" + prompt
        return self.generate_response(cot_prompt, **kwargs)

    def generate_zero_shot(self, prompt: str, **kwargs) -> str:
        return self.generate_response(prompt, **kwargs)

    def generate_few_shot(self, examples: list, prompt: str, **kwargs) -> str:
        few_shot_prompt = "\n".join(examples) + "\n" + prompt
        return self.generate_response(few_shot_prompt, **kwargs)
