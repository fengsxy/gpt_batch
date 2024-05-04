import pytest
from gpt_batch import GPTBatcher
import os

def test_handle_message_list():
    # Initialize the GPTBatcher with hypothetical valid credentials
    #api_key = #get from system environment
    api_key = os.getenv('TEST_KEY')
    if not api_key:
        raise ValueError("API key must be set in the environment variables")
    batcher = GPTBatcher(api_key=api_key, model_name='gpt-3.5-turbo-1106', system_prompt="Your task is to discuss privacy questions and provide persuasive answers with supporting reasons.")
    message_list = ["I think privacy is important", "I don't think privacy is important"]

    # Call the method under test
    results = batcher.handle_message_list(message_list)

    # Assertions to verify the length of the results and the structure of each item
    assert len(results) == 2, "There should be two results, one for each message"
    assert all(len(result) >= 2 for result in results), "Each result should be at least two elements"

# Optionally, you can add a test configuration if you have specific needs
if __name__ == "__main__":
    pytest.main()
