GPT_PROMPT_SYSTEM = """You are an evaluator. 
Given a series of conversational questions, 
your task is to compare an answer to the last question predicted by an AI 
to an answer labeled by a human."""

GPT_PROMPT_BINARY = """Question:
{{question}}

AI's answer: 
{{answer}}

Human's answer: 
{{label}}

Are the two answers to the last question the same?
Answer "yes" or "no" in the following JSON format:
```
{
  "result": "yes" or "no"
}
```
Do not explain or output anything else."""

GPT_PROMPT_SCORE = """Question:
{{question}}

Compare the following answers to the last question in the above conversation.
AI's answer: 
{{answer}}

Human's answer: 
{{label}}

On a scale of 0 to 10, 0 = not at all and 10 = same, how similar are the two answers?
Answer in the following JSON format:
```
{
  "result": A score from 0 to 10
}
```
Do not explain or output anything else."""
