import os
import pandas as pd

from anthropic import Anthropic
from dotenv import load_dotenv
from moral_foundations_llms.prompts import aita_prompt
from tqdm import tqdm

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# Load client
client = Anthropic(api_key=ANTHROPIC_API_KEY)
# Load data
df = pd.read_csv("../data/aita_final_v5.csv")
OUT_FILE = "../data/aita_final_v5.csv"

for idx in tqdm(range(10000, df.shape[0])):
    post = df['selftext'].iloc[idx]
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.2,
        system=aita_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": post
                    }
                ]
            }
        ]
    )
    answer = message.content[0].text
    if ("Verdict: " in answer) and ("Reasoning:" in answer):
        label = answer.split('Verdict: ')[1][:3]
        reason = answer.split('Reasoning:')[-1].strip()
    else:
        label = "TODO"
        reason = answer

    # Place in dataframe
    df.loc[idx, 'claude_response'] = answer
    df.loc[idx, 'claude_label'] = label
    df.loc[idx, 'claude_reason'] = reason

df.to_csv(OUT_FILE, index=False)