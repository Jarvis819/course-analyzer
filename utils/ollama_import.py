# utils/ollama_import.py

import ollama

def chat_with_phi_system(prompt: str) -> str:
    resp = ollama.chat(
        model="phi",
        messages=[{"role":"user","content":prompt}]
    )
    return resp["message"]["content"]

