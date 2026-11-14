
# AIGen Interactions Log - Team CyED3 (Valeria)

**Project:** CED-III-IT2 - Sentiment Analysis on GoEmotions  
**Student:** Miguel  
**AIGen Tool Used:** Qwen  
**Level of Use:** 3 - Collaboration with GenAI (critique, adjustment and refinement of outputs)

---

## Interaction Log

| Date            | Prompt (Context + Request) | AI Output (Summary) | Adjustments and Refinements (My Conceptual Contribution) |
|:----------------| :--- | :--- | :--- |
| 14-Nov-2026 | **Context:** I need to filter GoEmotions to keep only "pure" positive/negative comments based on the official mapping. <br>**Prompt:** "Write a Python function to filter comments where all emotions belong to the same sentiment group. Remove mixed, neutral, and ambiguous comments." | The AI generated a basic filter that checked if *at least one* emotion was positive or negative, but it failed to discard comments that had emotions from *both* groups (e.g., joy + anger). It also didn't handle empty emotion lists properly. | **1. Logic correction:** I rewrote the condition to ensure `len(set(sentiments)) == 1` and that the single sentiment is strictly 'positive' or 'negative'. This guarantees we only keep "pure" sentiments. <br>**2. Edge case handling:** I added a check to drop rows with empty `emotion_ids` lists to prevent mapping errors. <br>**3. Reproducibility:** I ensured the `random_state=42` was applied strictly *after* filtering and deduplication, guaranteeing the exact same 500/500 split every time. |
