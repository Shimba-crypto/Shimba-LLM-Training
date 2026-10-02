#!/usr/bin/env python3
"""
chat.py — Interactive chat with a trained Shimba LLM model.

Usage:
    python chat.py --model model.pth
    python chat.py --model model_quantized.pth --temperature 1.0 --max_tokens 150
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.model import GPT
from llm.tokenizer import CharTokenizer
from llm.generate import generate
from llm.compat import enable_safe_output

enable_safe_output()

class ChatSession:
    def __init__(self, model, tokenizer, max_history_tokens=1024, temperature=0.8, top_k=40, top_p=0.95, max_new_tokens=200, temperature_ramp=0.0):
        self.model = model
        self.tokenizer = tokenizer
        # The model asserts seq_len <= block_size, so a larger history budget
        # would crash generation rather than simply be ignored.
        block_size = model.cfg.block_size
        if max_history_tokens > block_size:
            print(f"[chat] --max_history {max_history_tokens} exceeds the model's "
                  f"context window ({block_size}); clamping.")
            max_history_tokens = block_size
        self.max_history_tokens = max_history_tokens
        self.temperature = temperature
        self.temperature_ramp = temperature_ramp
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.history = []

    def _build_prompt(self) -> str:
        lines = []
        for role, text in self.history:
            if role == "user":
                lines.append(f"User: {text}")
            else:
                lines.append(f"Assistant: {text}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _trim_history(self, prompt: str) -> str:
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) <= self.max_history_tokens:
            return prompt
        # Keep the most recent window, then drop the partial leading turn so
        # the model doesn't start mid-word.
        trimmed_tokens = tokens[-self.max_history_tokens:]
        trimmed = self.tokenizer.decode(trimmed_tokens)
        idx = trimmed.find("\n", trimmed.rfind("."))
        if idx != -1:
            trimmed = trimmed[idx+1:].lstrip()
        return trimmed

    def get_response(self, user_input: str) -> str:
        self.history.append(("user", user_input.strip()))
        prompt = self._build_prompt()
        prompt = self._trim_history(prompt)

        # Gradually warmer as the conversation goes on, if enabled.
        temp = self.temperature
        if self.temperature_ramp:
            temp = min(1.5, self.temperature + self.temperature_ramp * (len(self.history) // 2))

        response = generate(
            self.model, self.tokenizer, prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=temp,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        if response.startswith(prompt):
            response = response[len(prompt):].lstrip()
        self.history.append(("assistant", response))
        return response

    def reset(self):
        self.history = []

def main():
    parser = argparse.ArgumentParser(description="Interactive chat with Shimba LLM")
    parser.add_argument("--model", required=True, help="Path to model .pth file")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max new tokens per response")
    parser.add_argument("--max_history", type=int, default=1024, help="Max tokens to keep in conversation history")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N | mps")
    parser.add_argument("--temperature_ramp", type=float, default=0.0,
                        help="Per-turn temperature increase; 0 disables (recommended for "
                             "a char-level model, which drifts fast at high temperature)")
    args = parser.parse_args()

    # Check the model exists before the tokenizer, so a bad path names the
    # actual problem.
    if not os.path.exists(args.model):
        print(f"[error] Model file not found: {args.model}")
        sys.exit(1)

    tok_path = CharTokenizer.default_path(args.model)
    if not os.path.exists(tok_path):
        print(f"[error] Tokenizer not found at {tok_path}")
        print("        The tokenizer JSON must sit beside the model file.")
        sys.exit(1)
    tokenizer = CharTokenizer.load(tok_path)

    print(f"[chat] Loading model from {args.model} ...")
    # GPT.load handles every historical checkpoint layout, including weights
    # saved under "model" and keys prefixed by torch.compile.
    model = GPT.load(args.model, device=args.device)

    # Create chat session
    chat = ChatSession(
        model, tokenizer,
        max_history_tokens=args.max_history,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens,
    temperature_ramp=args.temperature_ramp,
    )

    print("\n" + "="*60)
    print("Shimba LLM – Interactive Chat")
    print(f"Model: {args.model}")
    print("Commands: /reset – clear conversation, /exit – quit")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input(">>> ").strip()
            if not user_input:
                continue
            if user_input == "/exit":
                print("Goodbye!")
                break
            if user_input == "/reset":
                chat.reset()
                print("[Conversation reset]")
                continue

            response = chat.get_response(user_input)
            print(f"\n{response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[error] {e}")
            continue

if __name__ == "__main__":
    main()