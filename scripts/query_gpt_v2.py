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
IN_FILE = here('data/aita_final_v28.csv')
OUT_FILE = here('data/aita_final_v19.csv')
FAIL_FILE = here('data/exp_gpt4_out_v19.pkl')
# Read in API key
with open(API_PATH, 'r') as f:
    openai.api_key = f.read().strip()
# Read in processed data
df = pd.read_csv(IN_FILE)

n_query = df.shape[0]
failed = []
responses = {}

# Iterate over posts
for post in tqdm.tqdm([100, 3654]):
    retries = 0
    completed = False
    while not completed and retries < 3:
        # Extract number of sentences
        n_sentences = len(nltk.sent_tokenize(df.iloc[post]['top_comment']))
        # Create system message
        system_message = utils.create_system_message(identity="", length=f"{n_sentences} sentences")
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
        df.loc[post, 'gpt4_label_2'] = gpt_label
        df.loc[post, 'gpt4_reason_2'] = answer
        completed = True
    if not completed:
        failed.append(post)

df.to_csv(OUT_FILE, index=False)

with open(FAIL_FILE, 'wb') as file:
    pickle.dump([failed, responses], file)