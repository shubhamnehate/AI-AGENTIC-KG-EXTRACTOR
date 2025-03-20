import os
import openai
from groq import Groq

def truncate_conversation(conversation, max_tokens=120000):
    """Truncate conversation to fit within context length."""
    total_tokens = sum(len(item["content"].split()) for item in conversation if item["content"])
    while total_tokens > max_tokens and len(conversation) > 1:
        conversation.pop(0)  # Remove the oldest message
        total_tokens = sum(len(item["content"].split()) for item in conversation if item["content"])

def call_openai_api(content, api_key, model="gpt-4o"):
    """Simplified OpenAI API call without persistent conversation.
    # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
    # do not change this unless explicitly requested by the user
    """
    client = openai.OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

def call_groq_api(prompt, api_key, model="llama3-8b-8192"):
    """Make an API call to Groq."""
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model
        )
    return chat_completion.choices[0].message.content

def unified_api_call(api_choice, prompt, api_key, model):
    """Unified API call function."""
    if api_choice == "OpenAI API":
        return call_openai_api(prompt, api_key, model=model)
    elif api_choice == "Groq API":
        return call_groq_api(prompt, api_key, model=model)
    else:
        raise ValueError("Invalid API Choice")
