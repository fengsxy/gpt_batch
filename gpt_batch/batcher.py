from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from tqdm import tqdm

class GPTBatcher:
    """
    A class to handle batching and sending requests to the OpenAI GPT model efficiently.

    Attributes:
        client (OpenAI): The client instance to communicate with the OpenAI API using the provided API key.
        model_name (str): The name of the GPT model to be used. Default is 'gpt-3.5-turbo-0125'.
        system_prompt (str): Initial prompt or context to be used with the model. Default is an empty string.
        temperature (float): Controls the randomness of the model's responses. Higher values lead to more diverse outputs. Default is 1.
        num_workers (int): Number of worker threads used for handling concurrent requests. Default is 64.
        timeout_duration (int): Maximum time (in seconds) to wait for a response from the API before timing out. Default is 60 seconds.
        retry_attempts (int): Number of retries if a request fails. Default is 2.
        miss_index (list): Tracks the indices of requests that failed to process correctly.

    Parameters:
        api_key (str): API key for authenticating requests to the OpenAI API.
        model_name (str, optional): Specifies the GPT model version. Default is 'gpt-3.5-turbo-0125'.
        system_prompt (str, optional): Initial text or question to seed the model with. Default is empty.
        temperature (float, optional): Sets the creativity of the responses. Default is 1.
        num_workers (int, optional): Number of parallel workers for request handling. Default is 64.
        timeout_duration (int, optional): Timeout for API responses in seconds. Default is 60.
        retry_attempts (int, optional): How many times to retry a failed request. Default is 2.
    """

    def __init__(self, api_key, model_name="gpt-3.5-turbo-0125", system_prompt="",temperature=1,num_workers=64,timeout_duration=60,retry_attempts=2,api_base_url=None):
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.num_workers = num_workers
        self.timeout_duration = timeout_duration
        self.retry_attempts = retry_attempts
        self.miss_index =[]
        if api_base_url:
            self.client.base_url = api_base_url

    def get_attitude(self, ask_text):
        index, ask_text = ask_text

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": ask_text}
            ],
            temperature=self.temperature,
        )
        return (index, completion.choices[0].message.content)

    def process_attitude(self, message_list):
        new_list = []
        num_workers = self.num_workers
        timeout_duration = self.timeout_duration
        retry_attempts=2

        executor = ThreadPoolExecutor(max_workers=num_workers)
        message_chunks = list(self.chunk_list(message_list, num_workers))
        for chunk in tqdm(message_chunks, desc="Processing messages"):
            future_to_message =  {executor.submit(self.get_attitude, message): message for message in chunk}
            for _ in range(retry_attempts):
                done, not_done = wait(future_to_message.keys(), timeout=timeout_duration)
                for future in not_done:
                    future.cancel()
                new_list.extend(future.result() for future in done if future.done())
                if len(not_done) == 0:
                    break
                future_to_message = {executor.submit(self.get_attitude, future_to_message[future]): future_to_message[future] for future, msg in not_done}
        executor.shutdown(wait=False)
        return new_list

    def complete_attitude_list(self,attitude_list, max_length):
        completed_list = []
        current_index = 0
        for item in attitude_list:
            index, value = item
            # Fill in missing indices
            while current_index < index:
                completed_list.append((current_index, None))
                current_index += 1
            # Add the current element from the list
            completed_list.append(item)
            current_index = index + 1
        while current_index < max_length:
            print("Filling in missing index", current_index)
            self.miss_index.append(current_index)
            completed_list.append((current_index, None))
            current_index += 1
        return completed_list

    def chunk_list(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def handle_message_list(self,message_list):
        indexed_list = [(index, data) for index, data in enumerate(message_list)]
        max_length = len(indexed_list)
        attitude_list = self.process_attitude(indexed_list)
        attitude_list.sort(key=lambda x: x[0])
        attitude_list = self.complete_attitude_list(attitude_list, max_length)
        attitude_list = [x[1] for x in attitude_list]
        return attitude_list
    
    def process_embedding(self,message_list):
            new_list = []
            executor = ThreadPoolExecutor(max_workers=self.num_workers)
            # Split message_list into chunks
            message_chunks = list(self.chunk_list(message_list, self.num_workers))
            fixed_get_embedding = partial(self.get_embedding)
            for chunk in tqdm(message_chunks, desc="Processing messages"):
                future_to_message = {executor.submit(fixed_get_embedding, message): message for message in chunk}
                for i in range(self.retry_attempts):
                    done, not_done = wait(future_to_message.keys(), timeout=self.timeout_duration)
                    for future in not_done:
                        future.cancel()
                    new_list.extend(future.result() for future in done if future.done())
                    if len(not_done) == 0:
                        break
                    future_to_message = {executor.submit(fixed_get_embedding, future_to_message[future]): future_to_message[future] for future in not_done}
            executor.shutdown(wait=False)
            return new_list
    def get_embedding(self,text):
        index,text = text
        response = self.client.embeddings.create(
        input=text,
        model=self.model_name)
        return (index,response.data[0].embedding)

    def handle_embedding_list(self,message_list):
        indexed_list = [(index, data) for index, data in enumerate(message_list)]
        max_length = len(indexed_list)
        attitude_list = self.process_embedding(indexed_list)
        attitude_list.sort(key=lambda x: x[0])
        attitude_list = self.complete_attitude_list(attitude_list, max_length)
        attitude_list = [x[1] for x in attitude_list]
        return attitude_list
    
    def get_miss_index(self):
        return self.miss_index

    # Add other necessary methods similar to the above, refactored to fit within this class structure.

