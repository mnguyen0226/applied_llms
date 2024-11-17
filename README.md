# Applied Large Language Models - Use Case & Automation
- Building Systems with the ChatGPT API
- ChatGPT Prompt Engineering for Developers.
- Reinforcement Learning From Human Feedbacks.

# Fine-tuning LLMs Notebooks
- Fine-tuning with specific dataset:
  - Example using AllNLI dataset for pair-based training
  - Dataset: sentence-transformers/all-nli
  - Example snippet shows loading this dataset for training and evaluation.
- Triplet-based Training
  - Example of using triplets (anchor, positive, negative) with the AllNLI dataset.
  - Dataset: sentence-transformers/all-nli, subset "triplet"
  - Example demonstrates preparing triplets for loss functions like TripletLoss.
- Semantic Textual Similarity (STS)
  - Fine-tuning and evaluation with the STS Benchmark (STSb) dataset.
  - Dataset: sentence-transformers/stsb
  - Evaluates model on semantic similarity scores between sentence pairs.
- Paraphrase Mining
  - Example of identifying semantically equivalent sentences with the Quora Question Pairs dataset.
  - Dataset: sentence-transformers/quora-duplicates
- Multi-Dataset Training
  - An advanced example combining multiple datasets for multi-task training
  - AllNLI (pair, pair-class, pair-score, triplet subsets).
  - STS Benchmark (STSb) for similarity scores.
  - Quora Question Pairs for duplicate question detection.
  - Natural Questions for question-answer pairs.
  - This example highlights training with different datasets and loss functions simultaneously.
- What other techniques for semantic analysis?
  - Read blogs and other notebooks