import pandas as pd
import tqdm
import openai
import nltk
import os
import pickle
import sys

from pyprojroot import here
from moral_foundations_llms import utils

# Read in API key
api_path = os.path.join(os.environ['HOME'], 'openai/api.txt')
with open(api_path, 'r') as f:
    openai.api_key = f.read().strip()
# Read in processed data
df = pd.read_csv(here('data/aita_processed.csv'))

n_query = df.shape[0]
failed = []
responses = {}

# Iterate over posts
for post in tqdm.tqdm(range(n_query)):
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
                model='gpt-3.5-turbo',
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
        gpt_dilemma = answer.split('Verdict')[0].replace('Dilemma:', '').strip()
        gpt_label = answer.split('Verdict: ')[1][:3]
        gpt_reason = answer.split('Reasoning:')[-1].strip()
        # Place in dataframe
        df.loc[post, 'gpt_dilemma'] = gpt_dilemma
        df.loc[post, 'gpt_label'] = gpt_label
        df.loc[post, 'gpt_reason'] = gpt_reason
        completed = True
    if not completed:
        failed.append(post)

df.to_csv(here('data/aita_processed_gpt.csv'), index=False)

with open('results.pkl', 'wb') as file:
    pickle.dump(failed, file)
    pickle.dump(responses, file)