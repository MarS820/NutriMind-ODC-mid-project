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
        # ... (keep your existing answer method unchanged)
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

        answer = generate(
            self.tokenizer,
            self.model,
            prompt,
            max_new_tokens=max_tokens
        )

        self.chat_history.append({
            "user": question,
            "assistant": answer
        })
        return answer

    def answer_stream(self, question, top_k=3, chunk_size=5):
        """
        Stream the answer token by token. 
        chunk_size controls how many tokens to yield at once (for smoother UI).
        """
        # Get context and build prompt (same as answer method)
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

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Generate tokens and stream them
        import torch
        
        # Set model to eval mode if needed
        self.model.eval()
        
        with torch.no_grad():
            # Generate with streaming
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for faster generation
            )
        
        # Decode the full generated text
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Remove the prompt part to get just the answer
        answer = full_text[len(prompt):].strip()
        
        # Store the full answer in chat history
        self.chat_history.append({
            "user": question,
            "assistant": answer
        })
        
        # Stream the answer in chunks
        words = answer.split()
        
        # If no words, yield the whole thing
        if not words:
            yield answer
            return
        
        # Yield progressively larger chunks for streaming effect
        for i in range(1, len(words) + 1):
            # Yield words in chunks for smoother streaming
            chunk = " ".join(words[:i])
            yield chunk

    def clear_memory(self):
        self.chat_history = []