
```markdown
# GPT Batcher

A simple tool to batch process messages using OpenAI's GPT models. `GPTBatcher` allows for efficient handling of multiple requests simultaneously, ensuring quick responses and robust error management.

## Installation

To get started with `GPTBatcher`, clone this repository to your local machine. Navigate to the repository directory and install the required dependencies (if any) by running:

```bash
pip install gpt_batch
```

## Quick Start

To use `GPTBatcher`, you need to instantiate it with your OpenAI API key and the model name you wish to use. Here's a quick guide:

### Handling Message Lists

This example demonstrates how to send a list of questions and receive answers:

```python
from gpt_batch.batcher import GPTBatcher

# Initialize the batcher
batcher = GPTBatcher(api_key='your_key_here', model_name='gpt-3.5-turbo-1106')

# Send a list of messages and receive answers
result = batcher.handle_message_list(['question_1', 'question_2', 'question_3', 'question_4'])
print(result)
# Expected output: ["answer_1", "answer_2", "answer_3", "answer_4"]
```

### Handling Embedding Lists

This example shows how to get embeddings for a list of strings:

```python
from gpt_batch.batcher import GPTBatcher

# Reinitialize the batcher for embeddings
batcher = GPTBatcher(api_key='your_key_here', model_name='text-embedding-3-small')

# Send a list of strings and get their embeddings
result = batcher.handle_embedding_list(['question_1', 'question_2', 'question_3', 'question_4'])
print(result)
# Expected output: ["embedding_1", "embedding_2", "embedding_3", "embedding_4"]
```

### Handling Message Lists with different API

This example demonstrates how to send a list of questions and receive answers with different api:

```python
from gpt_batch.batcher import GPTBatcher

# Initialize the batcher
batcher = GPTBatcher(api_key='sk-', model_name='deepseek-chat',api_base_url="https://api.deepseek.com/v1")


# Send a list of messages and receive answers
result = batcher.handle_message_list(['question_1', 'question_2', 'question_3', 'question_4'])

# Expected output: ["answer_1", "answer_2", "answer_3", "answer_4"]
```
## Configuration

The `GPTBatcher` class can be customized with several parameters to adjust its performance and behavior:

- **api_key** (str): Your OpenAI API key.
- **model_name** (str): Identifier for the GPT model version you want to use, default is 'gpt-3.5-turbo-1106'.
- **system_prompt** (str): Initial text or question to seed the model, default is empty.
- **temperature** (float): Adjusts the creativity of the responses, default is 1.
- **num_workers** (int): Number of parallel workers for request handling, default is 64.
- **timeout_duration** (int): Timeout for API responses in seconds, default is 60.
- **retry_attempts** (int): How many times to retry a failed request, default is 2.
- **miss_index** (list): Tracks indices of requests that failed to process correctly.

For more detailed documentation on the parameters and methods, refer to the class docstring.
