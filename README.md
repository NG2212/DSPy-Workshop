# DSPy Exercise Guide: From Zero to RAG (Code attached)

This guide provides a structured sequence of exercises to help students learn DSPy by progressively building a complete RAG system. Each exercise builds on previous ones, introducing new concepts while reinforcing what's already been learned.

## Exercise 1: Hello, DSPy!

**Objective:** Get familiar with basic DSPy functionality by making direct calls to language models.

**Tasks:**

- Configure DSPy with your language model of choice
- Make a direct call to the language model
- Try different parameters (temperature, max_tokens)
- Use the message format for a call

**Concept:** Understanding how DSPy wraps language model APIs

## Exercise 2: Working with Signatures

**Objective:** Learn how to create and use DSPy Signatures.

**Tasks:**

- Create a signature for text summarization
- Create a signature for question answering
- Create a signature with multiple inputs and outputs
- Use the signature with a Predict module

**Concept:** Signatures as interfaces that define inputs and outputs

## Exercise 3: Your First Module

**Objective:** Build a simple DSPy Module.

**Tasks:**

- Create a module that implements a summarizer
- Create a module that implements a question answering system
- Add docstrings to make your modules more effective
- Test your modules with different inputs

**Concept:** Modules as reusable components that implement signatures

## Exercise 4: Chain of Thought Reasoning

**Objective:** Implement step-by-step reasoning in DSPy.

**Tasks:**

- Use the ChainOfThought module
- Create your own reasoning module
- Compare results with and without reasoning
- Implement a module that shows multiple reasoning paths

**Concept:** Prompting language models to show their reasoning

## Exercise 5: Multi-Step Reasoning

**Objective:** Create more complex reasoning patterns.

**Tasks:**

- Implement a validate-and-correct pattern
- Create a module that generates multiple answers and selects the best
- Implement a module that decomposes problems into sub-problems
- Test with complex reasoning tasks

**Concept:** Composing multiple reasoning steps for better results

## Exercise 6: Introduction to Retrieval

**Objective:** Set up a simple retrieval system.

**Tasks:**

- Prepare a dataset for retrieval
- Implement a simple mock retriever
- Test retrieval with different queries
- Evaluate retrieval quality

**Concept:** Retrieval as a way to provide context to language models

## Exercise 7: Basic RAG Implementation

**Objective:** Create a simple RAG system.

**Tasks:**

- Connect a retriever with an answer generator
- Format retrieved context for the language model
- Test your RAG system with questions
- Compare RAG results with direct LM answers

**Concept:** The basic RAG pattern of retrieve-then-generate

## Exercise 8: Enhancing RAG with Query Transformation

**Objective:** Improve retrieval quality with query preprocessing.

**Tasks:**

- Implement a query transformation module
- Test how different queries affect retrieval
- Create a module that expands queries with related terms
- Measure the impact of query transformation

**Concept:** Query transformation as a way to improve retrieval

## Exercise 9: Self-Assessment and Reflection

**Objective:** Add self-evaluation to your RAG system.

**Tasks:**

- Implement a module that assesses answer quality
- Create a confidence scoring system
- Add source attribution to answers
- Implement a system that can recognize when it doesn't know

**Concept:** Self-assessment as a way to improve reliability

## Exercise 10: Advanced RAG Techniques

**Objective:** Explore more sophisticated RAG approaches.

**Tasks:**

- Implement multi-document reasoning
- Create a system that can handle follow-up questions
- Add fact-checking to your RAG pipeline
- Implement different retrieval strategies

**Concept:** Advanced patterns for more powerful RAG systems

## Final Challenge: Build Your Own RAG Application

**Objective:** Apply everything you've learned to build a complete RAG system.

**Tasks:**

- Choose a domain and dataset
- Implement a complete RAG pipeline
- Add advanced features (query optimization, self-assessment, etc.)
- Evaluate your system on domain-specific questions

## Extension Ideas

- **Conversational RAG:** Make your system handle follow-up questions
- **Knowledge Graph RAG:** Integrate structured knowledge
- **Multi-Modal RAG:** Add support for images or other modalities
- **Domain-Specific RAG:** Fine-tune for a specific domain like medical, legal, etc.
- **Optimized RAG:** Use DSPy optimizers to improve your system

## Evaluation Framework

For each exercise, evaluate your implementations using:

- **Correctness:** Does it produce the right answers?
- **Relevance:** Are the retrieved documents relevant?
- **Coherence:** Are the generated responses coherent and well-structured?
- **Reasoning:** Does the system show proper reasoning when needed?
- **Robustness:** How does it handle edge cases and difficult questions?

## Resources

- [DSPy Documentation](#)
- [Stanford DSPy Paper](#)
- [DSPy GitHub Repository](#)
- [Example Implementations](#)

---

**Nufar Gaspar**  
Executive AI consultant and trainer  
[ng@ngai.ai](mailto:ng@ngai.ai)  
+972-54-7884031
