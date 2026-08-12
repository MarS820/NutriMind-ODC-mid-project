from src.llm import generate

DIET_KEYWORDS = (
    "eat", "meal", "diet", "food", "calorie", "calories", "macro",
    "protein", "carb", "carbs", "fat intake", "nutrition", "plan"
)


class RAGPipeline:

    def __init__(self, vector_store, tokenizer, model, inbody_data=None,
                 activity_level="moderate"):
        self.vector_store = vector_store
        self.tokenizer = tokenizer
        self.model = model
        self.inbody_data = inbody_data
        self.activity_level = activity_level
        self.chat_history = []

    def answer(self, question, top_k=3):
        results = self.vector_store.search(question, top_k=top_k)
        context = "\n\n".join(
            result["text"] if isinstance(result, dict) else str(result)
            for result in results
        )
        max_tokens = 150
        
        history = ""
        if self.chat_history:
            history = "\n\nPrevious conversation:\n"
            for message in self.chat_history:
                history += f"User: {message['user']}\n"
                history += f"Assistant: {message['assistant']}\n"

    def answer_stream(self, question, top_k=3):
        results = self.vector_store.search(question, top_k=top_k)

        context = "\n\n".join(
            result["text"] if isinstance(result, dict) else str(result)
            for result in results
        )

        max_tokens = 150

        history = ""
        if self.chat_history:
            history = "\n\nPrevious conversation:\n"
            for message in self.chat_history:
                history += f"User: {message['user']}\n"
                history += f"Assistant: {message['assistant']}\n"

        prompt = f"""
You are NutriMind AI, an assistant that answers questions using the user's InBody report.

Rules:
- Use ONLY the information provided in the context. Do not invent values.
- Answer ONLY what is asked. Do not add extra metrics, health interpretations,
  risk assessments, or lifestyle advice unless the question explicitly asks for them.
- Report numbers exactly as they appear in the context (do not label them as
  "healthy", "normal", "moderate", "low risk", etc. unless that exact label
  appears in the context).
- If the question asks for advice, a meal plan, or recommendations, base it
  directly on the CALCULATED NUTRITION TARGETS section if present (calories,
  protein, fat, carbs) rather than generic advice. Explicitly mention the
  numbers you're using.
- This is general informational content, not professional medical or
  dietary advice. If the question asks for a meal plan, calorie target, or
  specific dietary prescription, end your answer with a brief note
  recommending they confirm with a registered dietitian before making
  significant diet changes.
- If the answer is not available in the context, say:
  "I couldn't find this information in your InBody report."

Context:
{context}

{history}

Question:
{question}

Answer:
"""

        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True
        )

        self.model.eval()

        thread = Thread(
            target=self.model.generate,
            kwargs=generation_kwargs
        )

        thread.start()

        full_answer = ""

        for new_text in streamer:
            full_answer += new_text
            yield full_answer

        thread.join()

        if not full_answer:
            full_answer = "I couldn't generate a response. Please try again."
            yield full_answer

        self.chat_history.append({
            "user": question,
            "assistant": full_answer
    })
    def clear_memory(self):
        self.chat_history = []