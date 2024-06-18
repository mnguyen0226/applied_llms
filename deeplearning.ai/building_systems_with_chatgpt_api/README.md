# Building Systems with the ChatGPT API

## Language Models, the Chat Format and Tokens
- Text generation process: Give a prompt and allow the language to fill it. How it fill out? Supervised learning to traing and predict the next word.
- Two types of LLMs
  - Base LLMs: Same results
  - Instruction Tuned LLMs: How to train: First, train a base LLM on a lot of data. You then further train the model: Fine-tune on examples of where the output follows an input instruction. Then we obtain human-rating of the quelity of different LLM outputs, on criteria such as whether it is helpful, honest, and harmless. Tune LLM to increase probability that it genrates the more highly rated output (using RLHF: Reinforcement Learning from Human Feedback).


## Classification
- Evaluate inputs to ensure safety of the model reponses.

## Moderation
- Important to check if the user is using the system correctly but not abusing it. We can use the moderation API and detect prompt injection!

[Moderation API](https://platform.openai.com/docs/guides/moderation)

- Avoid Prompt Injections: If you build a bot for a product, the user might ask you to help with homeworks or fake news articles.

## Chain of Thought Reasoning
- Monologue: hiding the model reasoning to the user.

## Chaining Prompts

## Check Outputs

## Evaluation
