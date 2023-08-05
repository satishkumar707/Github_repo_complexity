import requests
import openai
import os
import tokenize
from io import BytesIO
import re


# GitHub API constants
GITHUB_API_BASE_URL = 'https://api.github.com/users'
GITHUB_API_REPOS_ENDPOINT = '/repos'

# GPT constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
GPT_ENGINE = 'text-davinci-003'  # Choose the appropriate GPT-3 engine for your needs

def fetch_user_repositories(github_user_url):
    # Extract the GitHub username from the URL
    username = github_user_url.split('/')[-1]
    # Fetch user repositories
    url = f'{GITHUB_API_BASE_URL}/{username}{GITHUB_API_REPOS_ENDPOINT}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch repositories. Status code: {response.status_code}")

# Function to tokenize the code
def tokenize_code(code):
    tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    tokenized_code = [token.string for token in tokens if token.type != tokenize.COMMENT]
    return ' '.join(tokenized_code)

# Function to split large code files into chunks
def split_large_code(code, max_tokens_per_chunk):
    # Split the code into lines
    lines = code.split('\n')
    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0

    for line in lines:
        line_tokens = tokenize_code(line)
        if current_chunk_tokens + len(line_tokens) <= max_tokens_per_chunk:
            current_chunk += line + '\n'
            current_chunk_tokens += len(line_tokens)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + '\n'
            current_chunk_tokens = len(line_tokens)

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Preprocess the code in repositories
def preprocess_code(code):
    # Define the maximum number of tokens per GPT request
    max_tokens_per_chunk = 1000
    
    # Check if the code size is within the token limits
    if len(tokenize_code(code)) <= max_tokens_per_chunk:
        return code

    # If the code size exceeds token limits, split it into chunks
    code_chunks = split_large_code(code, max_tokens_per_chunk)
    processed_code = ""

    for chunk in code_chunks:
        processed_code += chunk + '\n'
    
    return processed_code

# Evaluates complexity of each repository
def evaluate_complexity(code):
    prompt = "Assess the technical complexity of the following code and only provide a number:\n"
    prompt1 = "Assess the technical complexity of the following code and provide a score and justify your score:\n"
    preprocessed_code = preprocess_code(code)
    full_prompt = prompt + preprocessed_code
    full_prompt1 = prompt1 + preprocessed_code
    response = openai.Completion.create(
        engine=GPT_ENGINE,
        prompt=full_prompt,
        max_tokens=1000,  # Adjust this value based on GPT's token limit
        temperature=0.3,  # Adjust temperature for desired randomness
    )
    response1 = openai.Completion.create(
        engine=GPT_ENGINE,
        prompt=full_prompt1,
        max_tokens=1000,  # Adjust this value based on GPT's token limit
        temperature=0.1,  # Adjust temperature for desired randomness
    )
    return response.choices[0].text.strip(),response1.choices[0].text.strip()

# This compares and finds most complex repository
def find_most_technically_complex_repository(github_user_url):
    repositories = fetch_user_repositories(github_user_url)
    most_complex_repo = None
    max_complexity_score = float('-inf')
    
    for repo in repositories:
        preprocessed_code = preprocess_code(repo['url'])
        complexity_score,justification = evaluate_complexity(preprocessed_code)
        complexity_score = float(re.sub(r'[^\d.]', '', complexity_score))
        print(complexity_score,type(complexity_score))
        
        if complexity_score > max_complexity_score:
            max_complexity_score = complexity_score
            most_complex_repo = repo['name']
            justification = justification
    
    return most_complex_repo,justification

if __name__ == "__main__":
    user_url = "https://github.com/satishkumar707"
    most_complex_repo,justification = find_most_technically_complex_repository(user_url)
    print(f"The most technically complex repository is: {most_complex_repo} and justification is as following {justification.split('/')[-1]}")
