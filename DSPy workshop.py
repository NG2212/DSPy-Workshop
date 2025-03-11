
'''
********************************************
This is a DSPy workshop and homework flow:
********************************************
'''
#installing the package
pip install -U dspy

#Other imports we will need
import dspy
import pandas as pd
from datasets import load_dataset
import random


#setup the language model - this is SONY's current API key - don't send it to others
import dspy
lm = dspy.LM('openai/gpt-4o-mini', api_key='SONY_API_KEY')
dspy.configure(lm=lm)


# Part 1: DSPy Fundamentals ------------------------------------------------
## Exercise 1: Basic LM calls
print("Exercise 1: Basic LM calls")
print("-" * 50)

# Direct call to language model
response = lm("Explain what RAG means in AI in one sentence.", temperature=0.3)
print(f"Response: {response}\n")

# Using the message format
messages = [{"role": "user", "content": "Explain what RAG means in AI in one sentence."}]
response = lm(messages=messages)
print(f"Messages format response: {response}\n")

## Exercise 2: Creating and using Signatures
# Define a basic signature for summarization using field definitions
class Summarizer(dspy.Signature):
    """Summarize the given text concisely."""
    
    text = dspy.InputField()
    summary = dspy.OutputField()

# Create your own signature for text classification
class SimpleClassifier(dspy.Signature):
    """Classify the given text into one of the provided categories."""
    
    text = dspy.InputField()
    categories = dspy.InputField()
    category = dspy.OutputField()
    confidence = dspy.OutputField()


    # Example 1: Basic summarization
print("Example 1: Basic Summarization")
summarize = dspy.Predict(Summarizer)
text_to_summarize = "Retrieval-Augmented Generation (RAG) is an AI framework that combines the knowledge access capabilities of retrieval-based systems with the generative abilities of large language models. RAG systems retrieve relevant information from a knowledge base and then use that information to generate more accurate, factual, and contextually appropriate responses. This approach addresses limitations of standalone LLMs, particularly their tendency to hallucinate or provide outdated information, by grounding responses in external, retrievable knowledge sources."
result = summarize(text=text_to_summarize)
print(f"Summary: {result.summary}\n")

# Example 2: Text classification with multiple inputs
print("Example 2: Text Classification")
classify = dspy.Predict(SimpleClassifier)
result = classify(
    text="The model achieved 95% accuracy on the test set, improving over the previous state-of-the-art by 2.3%.",
    categories="Research, Technology, Sports, Politics, Entertainment"
)
print(f"Category: {result.category}")

# Check if confidence field exists before trying to print it
if hasattr(result, 'confidence'):
    print(f"Confidence: {result.confidence}\n")
else:
    print("Note: Confidence field not returned by model\n")


# Example 3: Creating a signature with a more complex task
class Translator(dspy.Signature):
    """Translate text from one language to another."""
    
    text = dspy.InputField(desc="The text to translate")
    source_language = dspy.InputField(desc="The source language")
    target_language = dspy.InputField(desc="The target language")
    translation = dspy.OutputField(desc="The translated text")
    
print("Example 3: Translation with Field Descriptions")
translate = dspy.Predict(Translator)
result = translate(
    text="Machine learning models often require large amounts of data.",
    source_language="English",
    target_language="French"
)
print(f"Translation: {result.translation}\n")


# Example 4: Two versions of a QA signature
class SimpleQA(dspy.Signature):
    """Answer questions directly without additional context."""
    
    question = dspy.InputField()
    answer = dspy.OutputField()

class ContextQA(dspy.Signature):
    """Answer questions based on provided context."""
    
    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField()

print("Example 4: Comparing QA with and without context")
# Simple QA without context
simple_qa = dspy.Predict(SimpleQA)
result_no_context = simple_qa(question="What is the capital of France?")
print(f"Answer without context: {result_no_context.answer}")

# QA with context
context_qa = dspy.Predict(ContextQA)
result_with_context = context_qa(
    question="What is the capital of France?",
    context="France is a country in Western Europe with several overseas territories. Its capital is Paris, which is also its largest city."
)
print(f"Answer with context: {result_with_context.answer}\n")

# TODO: Create your own signature for a task of your choice
# Examples: sentiment analysis, fact checking, question generation, etc.




## Exercise 3: Build a simple Module
print("Exercise 3: Build a simple Module")
print("-" * 50)

# Define a signature for question answering
class QASignature(dspy.Signature):
    """Answer the question accurately and concisely."""
    
    question = dspy.InputField()
    answer = dspy.OutputField()

# Simple module for question answering
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use the class-based signature instead of string format
        self.qa = dspy.Predict(QASignature)
    
    def forward(self, question):
        return self.qa(question=question)

# Create and test the module
simple_qa = SimpleQA()
result = simple_qa(question="What are the advantages of using RAG over just an LLM?")
print(f"QA Answer: {result.answer}\n")

# TODO: Create your own module that extends this with reasoning

# Part 2: Building Blocks for RAG ------------------------------------------

## Exercise 4: Chain of Thought
print("Exercise 4: Chain of Thought")
print("-" * 50)

# Define a signature for chain of thought reasoning
class COTSignature(dspy.Signature):
    """Answer questions with step-by-step reasoning."""
    
    question = dspy.InputField()
    reasoning = dspy.OutputField()
    answer = dspy.OutputField()

class ReasonedQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # Using chain of thought signature
        self.qa = dspy.ChainOfThought(COTSignature)
    
    def forward(self, question):
        return self.qa(question=question)

# Test with a reasoning question
reasoned_qa = ReasonedQA()
result = reasoned_qa(question="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?")
print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.answer}\n")

## Exercise 5: Multi-step reasoning (Validation)
print("Exercise 5: Multi-step reasoning (Validation)")
print("-" * 50)

# Define signatures for generation and validation
class COTSignature(dspy.Signature):
    """Answer questions with step-by-step reasoning."""
    
    question = dspy.InputField()
    reasoning = dspy.OutputField()
    answer = dspy.OutputField()

class ValidateSignature(dspy.Signature):
    """Validate if the answer is correct and provide a correction if needed."""
    
    question = dspy.InputField()
    answer = dspy.InputField()
    is_correct = dspy.OutputField()
    correction = dspy.OutputField()

class ValidatedQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(COTSignature)
        self.validator = dspy.Predict(ValidateSignature)
    
    def forward(self, question):
        # Generate initial answer
        initial_result = self.generate(question=question)
        
        # Validate the answer
        validation = self.validator(
            question=question,
            answer=initial_result.answer
        )
        
        # Return final answer
        if validation.is_correct.lower() == "yes":
            final_answer = initial_result.answer
        else:
            final_answer = validation.correction
            
        return dspy.Prediction(
            initial_answer=initial_result.answer,
            is_correct=validation.is_correct,
            correction=validation.correction,
            final_answer=final_answer
        )

# Test with a tricky question
validated_qa = ValidatedQA()
result = validated_qa(question="What's the capital of Australia? Is it Sydney?")
print(f"Initial Answer: {result.initial_answer}")
print(f"Is Correct: {result.is_correct}")
print(f"Correction: {result.correction}")
print(f"Final Answer: {result.final_answer}\n")

# TODO: Create your own multi-step reasoning module with a different approach

# Part 3: Building a RAG System --------------------------------------------

## Exercise 6: Prepare a dataset
print("Exercise 6: Prepare a dataset")
print("-" * 50)

def load_simple_dataset():
    """Load a simple dataset for our RAG example."""
    try:
        # Try to load Squad dataset (common QA dataset)
        dataset = load_dataset("squad", split="train[:100]")
        
        # Convert to a simpler format
        corpus = []
        for i, item in enumerate(dataset):
            corpus.append({
                "id": str(i),
                "title": item["title"],
                "context": item["context"],
                "question": item["question"],
                "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
                "text": f"Title: {item['title']}\nContext: {item['context']}"
            })
        
    except Exception as e:
        print(f"Error loading Squad dataset: {e}")
        print("Creating a fallback dataset instead.")
        
        # Create a small fallback dataset
        corpus = [
            {
                "id": "1",
                "title": "Introduction to RAG",
                "context": "Retrieval-Augmented Generation (RAG) combines retrieval systems with generative AI to produce more accurate responses.",
                "question": "What is RAG?",
                "answer": "Retrieval-Augmented Generation combines retrieval systems with generative AI.",
                "text": "Title: Introduction to RAG\nContext: Retrieval-Augmented Generation (RAG) combines retrieval systems with generative AI to produce more accurate responses."
            },
            # Add more examples...
        ]
    
    return corpus

# Load the dataset
corpus = load_simple_dataset()
print(f"Loaded {len(corpus)} documents.")
print(f"Sample document: {corpus[0]}\n")

## Exercise 7: Create a mock retriever
print("Exercise 7: Create a mock retriever")
print("-" * 50)

class MockRetriever:
    def __init__(self, corpus, k=3):
        self.corpus = corpus
        self.k = k
    
    def __call__(self, query):
        # This is a very simple mock retriever that just returns random documents
        # In a real system, you would use semantic search or other retrieval methods
        return random.sample(self.corpus, min(self.k, len(self.corpus)))

# Create the retriever
retriever = MockRetriever(corpus)

# Test the retriever
test_query = "What is retrieval-augmented generation?"
retrieved_docs = retriever(test_query)
print(f"Retrieved {len(retrieved_docs)} documents for query: '{test_query}'")
print(f"First document: {retrieved_docs[0]['text']}\n")

# TODO: Implement a slightly better retriever that uses keyword matching


## Exercise 8: Create a RAG pipeline
print("Exercise 8: Create a RAG pipeline")
print("-" * 50)

# Define a signature for RAG
class RAGSignature(dspy.Signature):
    """Given the context and question, provide an accurate answer."""
    
    context = dspy.InputField()
    question = dspy.InputField()
    reasoning = dspy.OutputField()
    answer = dspy.OutputField()

class SimpleRAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generate_answer = dspy.ChainOfThought(RAGSignature)
    
    def forward(self, question):
        # Retrieve relevant documents
        retrieved_docs = self.retriever(question)
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        
        # Generate answer
        result = self.generate_answer(context=context, question=question)
        
        return dspy.Prediction(
            context=context,
            reasoning=result.reasoning,
            answer=result.answer
        )
    
    # Create and test the RAG pipeline
simple_rag = SimpleRAG(retriever)
result = simple_rag(question="What is RAG and how does it work?")
print(f"Context: {result.context[:300]}...")  # Show just a part of the context
print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.answer}\n")

# Part 4: Extending the RAG System -----------------------------------------
## Exercise 9: Query transformation
print("Exercise 9: Query transformation")
print("-" * 50)

# Define signatures for query transformation and answer generation
class QueryTransformSignature(dspy.Signature):
    """Transform the original question into a better query for retrieval."""
    
    question = dspy.InputField()
    improved_query = dspy.OutputField()

class RAGSignature(dspy.Signature):
    """Given the context and question, provide an accurate answer."""
    
    context = dspy.InputField()
    question = dspy.InputField()
    reasoning = dspy.OutputField()
    answer = dspy.OutputField()

class EnhancedRAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        
        # Module to transform the query
        self.query_transformer = dspy.Predict(QueryTransformSignature)
        
        # Module to generate the answer
        self.generate_answer = dspy.ChainOfThought(RAGSignature)
    
    def forward(self, question):
        # Transform the query
        transformed = self.query_transformer(question=question)
        improved_query = transformed.improved_query
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever(improved_query)
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        
        # Generate answer
        result = self.generate_answer(context=context, question=question)
        
        return dspy.Prediction(
            original_query=question,
            improved_query=improved_query,
            context=context,
            reasoning=result.reasoning,
            answer=result.answer
        )
    
    # Create and test the enhanced RAG
enhanced_rag = EnhancedRAG(retriever)
result = enhanced_rag(question="What does RAG do?")
print(f"Original Query: {result.original_query}")
print(f"Improved Query: {result.improved_query}")
print(f"Answer: {result.answer}\n")

## Exercise 10: Add self-assessment
print("Exercise 10: Add self-assessment")
print("-" * 50)

class AssessmentSignature(dspy.Signature):
    """Assess the quality and factuality of the answer."""
    
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.InputField()
    assessment = dspy.OutputField()
    confidence = dspy.OutputField()
Here's how to extend the RAG system with the assessment step for Exercise 10:
pythonCopy## Exercise 10: Add self-assessment
print("Exercise 10: Add self-assessment")
print("-" * 50)

# Define assessment signature
class AssessmentSignature(dspy.Signature):
    """Assess the quality and factuality of the answer."""
    
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.InputField()
    assessment = dspy.OutputField()
    confidence = dspy.OutputField()

# Define RAG with assessment
class AssessedRAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        
        # Query transformation
        self.query_transformer = dspy.Predict(QueryTransformSignature)
        
        # Answer generation
        self.generate_answer = dspy.ChainOfThought(RAGSignature)
        
        # Assessment step
        self.assessor = dspy.Predict(AssessmentSignature)
    
    def forward(self, question):
        # Transform the query
        transformed = self.query_transformer(question=question)
        improved_query = transformed.improved_query
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever(improved_query)
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        
        # Generate answer
        result = self.generate_answer(context=context, question=question)
        
        # Assess the answer
        assessment_result = self.assessor(
            context=context,
            question=question,
            answer=result.answer
        )
        
        return dspy.Prediction(
            original_query=question,
            improved_query=improved_query,
            context=context,
            reasoning=result.reasoning,
            answer=result.answer,
            assessment=assessment_result.assessment,
            confidence=assessment_result.confidence
        )

# Create and test the assessed RAG
assessed_rag = AssessedRAG(retriever)
result = assessed_rag(question="What is RAG and how does it work?")

print(f"Original Query: {result.original_query}")
print(f"Improved Query: {result.improved_query}")
print(f"Answer: {result.answer}")
print(f"Assessment: {result.assessment}")
print(f"Confidence: {result.confidence}\n")

# TODO: Extend the EnhancedRAG class to include a self-assessment step
# that evaluates the quality and factuality of the generated answer

# Part 5: Challenges -------------------------------------------------------

print("Part 5: Challenges")
print("-" * 50)
print("""
Challenge options:
1. Implement a better retriever using keyword matching or embeddings
2. Add fact-checking to your RAG pipeline
3. Implement multi-document reasoning
4. Add answer generation with cited sources
5. Create a system that can answer follow-up questions using conversation history
6. Create a RAG pipline on your docs

Choose one challenge to implement!
""")

# Your challenge implementation here...

print("\nWorkshop complete! You've built a basic RAG system with DSPy.")