import pandas as pd
import vertexai
from vertexai.language_models import TextGenerationModel

df = pd.read_csv('../data/aita_final_v4.csv')

vertexai.init(project="dlab-hsi-dev", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.4,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison@001")
prompt = f"""You are a tool for evaluating moral dilemmas.

Please evaluate the following post from the subreddit \"Am I the Asshole\".

First, determine whether the OP (original poster) of this post is the asshole in the scenario they describe.

Provide a categorical label indicating your assessment of the scenario, from one of these choices:

- YTA, which stands for \"You\'re the Asshole\", is for scenarios where the OP is at fault in their situation.
- NTA, which stands for \"Not the Asshole\", is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
- ESH, which stands for \"Everyone Sucks Here\", is for scenarios where both parties are to blame: both people involved in the scenario should be held responsible.
- NAH, which stands for \"No Assholes Here\", is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
- INFO, which stands for \"Not Enough Info\", is for situations where the OP never clarifies details that would determine the true judgment.

Then, please provide an explanation in no more than one paragraph of why you chose this label.

Structure your answer in the following format:

Verdict: [CATEGORICAL LABEL]
Reasoning: [EXPLANATION]

Here is the dilemma you will evaluate:

{df['selftext'].iloc[15]}
"""
response = model.predict(
    prompt,
    **parameters
)
print(prompt)
print(f"Response from Model: {response.text}")