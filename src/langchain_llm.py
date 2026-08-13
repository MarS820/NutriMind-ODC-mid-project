"""
LangChain LLM wrapper around the local Qwen model.

Why this file exists:
    src/llm.py's `generate()` blocks until the ENTIRE response is generated,
    then main_app.py fakes "streaming" by revealing words after the fact.
    That's why the UI sits on "Thinking..." for a long time before anything
    shows up -- there's no real streaming happening, just a replay.

    This wrapper uses transformers' TextIteratorStreamer to run generation
    in a background thread and yield tokens AS THEY ARE PRODUCED, wired up
    through LangChain's Runnable `.stream()` interface. Nothing in
    src/llm.py or src/rag_pipeline.py is modified or removed.
"""

import threading
from typing import Any, Iterator, List, Optional

import torch
from transformers import TextIteratorStreamer

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk


class QwenLangChainLLM(LLM):
    """A LangChain-compatible LLM that wraps an already-loaded
    (tokenizer, model) pair and supports real incremental streaming.
    """

    tokenizer: Any
    model: Any
    max_new_tokens: int = 150
    do_sample: bool = False
    temperature: float = 0.7

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "qwen-local-streaming"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Non-streaming path (kept for compatibility with chains that
        call .invoke() / .predict() directly)."""
        chunks = list(self._stream(prompt, stop=stop, run_manager=run_manager, **kwargs))
        return "".join(chunk.text for chunk in chunks)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Real token-by-token streaming using TextIteratorStreamer.

        Generation runs on a background thread; tokens are yielded to the
        caller as soon as the model produces them, instead of waiting for
        the full ~150 tokens to finish first.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        # Drop None values (e.g. temperature when do_sample=False)
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        thread = threading.Thread(target=self._generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if not new_text:
                continue
            chunk = GenerationChunk(text=new_text)
            if run_manager:
                run_manager.on_llm_new_token(new_text, chunk=chunk)
            yield chunk

        thread.join()

    def _generate(self, **generation_kwargs):
        with torch.no_grad():
            self.model.generate(**generation_kwargs)
