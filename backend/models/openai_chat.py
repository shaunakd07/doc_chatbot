from __future__ import annotations

import base64
import io
from typing import Optional

from openai import OpenAI
from PIL import Image


class OpenAIChatModel:
    def __init__(
        self,
        model_id: str,
        api_key: str,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai.")
        self.model_id = model_id
        self.client = OpenAI(api_key=api_key)

    def generate_text(self, prompt: str, max_new_tokens: int = 256, images: Optional[list[str]] = None) -> str:
        max_tokens = max(64, int(max_new_tokens))
        
        content = [{"type": "text", "text": prompt}]
        if images:
            for b64 in images:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def answer_image_question(self, image: Image.Image, question: str, max_new_tokens: int = 1500) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        response = self.client.chat.completions.create(
            # Force gpt-4o-mini here specifically for the ingestion diagram summary to save costs
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=max(80, int(max_new_tokens)),
        )
        return (response.choices[0].message.content or "").strip()
