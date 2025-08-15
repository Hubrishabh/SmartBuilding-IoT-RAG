import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from typing import List, Optional
from backend.config import OPENAI_API_KEY

def llm_summarize(question: str, contexts: List[str], max_tokens: int = 256) -> Optional[str]:
    """Summarizes maintenance actions based on retrieved contexts using OpenAI API."""
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        system = (
            "You are a helpful maintenance assistant for building equipment. "
            "Given retrieved document excerpts and a sensor alert, provide concise actionable steps "
            "and cite the most relevant source names."
        )
        prompt_parts = [f"Context {i+1}: {c}" for i, c in enumerate(contexts[:6])]
        prompt = "\n\n".join(prompt_parts)
        user_prompt = (
            f"Question/Alert:\n{question}\n\nRetrieved contexts:\n{prompt}\n\n"
            "Provide 6 concise actionable checks or steps, and mention the source file names for each step."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM error: {e})"
if __name__ == "__main__":
    example_question = "HVAC system temperature anomaly detected."
    example_contexts = [
        "Check the air filters for dust accumulation.",
        "Verify that the thermostat sensors are calibrated."
    ]
    print(llm_summarize(example_question, example_contexts))