from src.llm import generate
from src.nutrition_calculator import calculate_targets, targets_to_context_string


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
        # Full parsed InBody dict (from pdf_processor.parse_inbody), used to
        # compute real calorie/macro targets for diet-related questions.
        self.inbody_data = inbody_data
        self.activity_level = activity_level
        

    def _is_diet_question(self, question: str) -> bool:
        q = question.lower()
        return any(keyword in q for keyword in DIET_KEYWORDS)

    def answer(self, question, top_k=3):

        results = self.vector_store.search(
            question,
            top_k=top_k
        )

        context = "\n\n".join(
            result["text"] if isinstance(result, dict) else str(result)
            for result in results
        )

        max_tokens = 150

        # For diet/meal-plan questions, add computed nutrition targets so the
        # model has real numbers to build a plan from instead of guessing.
        if self.inbody_data and self._is_diet_question(question):
            targets = calculate_targets(self.inbody_data, self.activity_level)
            context += "\n\n" + targets_to_context_string(targets)
            max_tokens = 400

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

Question:
{question}

Answer:
"""

        return generate(
            self.tokenizer,
            self.model,
            prompt,
            max_new_tokens=max_tokens
        )