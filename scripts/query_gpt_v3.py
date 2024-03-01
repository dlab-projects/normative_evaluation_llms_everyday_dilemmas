import pandas as pd
import tqdm
import openai
import nltk
import os
import pickle
import sys

from pyprojroot import here
from moral_foundations_llms import utils

API_PATH = os.path.join(os.environ['HOME'], 'openai/api.txt')
IN_FILE = here('data/aita_final_v3.csv')
OUT_FILE = here('data/aita_final_v3.csv')
FAIL_FILE = here('data/exp_gpt_4_out.pkl')
# Read in API key
with open(API_PATH, 'r') as f:
    openai.api_key = f.read().strip()
# Read in processed data
df = pd.read_csv(IN_FILE)

n_query = df.shape[0]
failed = []
responses = {}

# Iterate over posts
for post in tqdm.tqdm(range(10, 1000)):
    retries = 0
    completed = False
    while not completed and retries < 3:
        # Create system message
        system_message = utils.aita_cot_prompt
        # Run GPT query
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": df['selftext'].iloc[post]}
                ],
                temperature=0.4)
        # Handle exceptions
        except openai.error.Timeout as e:
            print(f"Post {post}, OpenAI API returned a Timeout Error: {e}")
            retries += 1
            continue
        except openai.error.APIError as e:
            print(f"Post {post}, OpenAI API returned an API Error: {e}")
            retries += 1
            continue
        except openai.error.APIConnectionError as e:
            print(f"Post {post}, OpenAI API request failed to connect: {e}")
            retries += 1
            continue
        except openai.error.ServiceUnavailableError as e:
            print(f"Post {post}, OpenAI API returned a Service Unavailable Error: {e}")
            retries += 1
            continue
        # Extract answer
        answer = response['choices'][0]['message']['content']
        # Store answer, dilemma, label, and reason
        responses[post] = answer
        gpt_label = answer.split('Verdict: ')[1][:3]
        gpt_reason = answer.split('Reasoning:')[-1].strip()
        # Place in dataframe
        df.loc[post, 'gpt4_cot_label'] = gpt_label
        df.loc[post, 'gpt4_cot_reason'] = gpt_reason
        completed = True
    if not completed:
        failed.append(post)

df.to_csv(OUT_FILE, index=False)

with open(FAIL_FILE, 'wb') as file:
    pickle.dump([failed, responses], file)