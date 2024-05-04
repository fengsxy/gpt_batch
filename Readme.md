# GPT Batcher

A simple tool to batch process messages using OpenAI's GPT models.

## Installation

Clone this repository and run:

## Usage

Here's how to use the `GPTBatcher`:

```python
from gpt_batch.batcher import GPTBatcher

batcher = GPTBatcher(key='your_key_here', model_name='gpt-3.5-turbo-1106')
result = batcher.handle_list(['your', 'list', 'of', 'messages'])
print(result)

