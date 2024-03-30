import pandas as pd
import vertexai
from vertexai.language_models import ChatModel
from vertexai import generative_models


df = pd.read_csv('../data/aita_final_v4.csv')

vertexai.init(project="dlab-hsi-dev", location="us-central1")
chat_model = ChatModel.from_pretrained("chat-bison@001")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.4,
    "top_p": 0.8,
    "top_k": 40
}
safety_settings={
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE}


chat = chat_model.start_chat(
    context="""You are a tool for evaluating moral dilemmas.

Please evaluate the following post from the subreddit \\\"Am I the Asshole\\\".

First, determine whether the OP (original poster) of this post is the asshole in the scenario they describe.

Provide a categorical label indicating your assessment of the scenario, from one of these choices:

- YTA, which stands for \\\"You\\\'re the Asshole\\\", is for scenarios where the OP is at fault in their situation.
- NTA, which stands for \\\"Not the Asshole\\\", is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
- ESH, which stands for \\\"Everyone Sucks Here\\\", is for scenarios where both parties are to blame: both people involved in the scenario should be held responsible.
- NAH, which stands for \\\"No Assholes Here\\\", is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
- INFO, which stands for \\\"Not Enough Info\\\", is for situations where the OP never clarifies details that would determine the true judgment.

Then, please provide an explanation for why you chose this label. Restrict your explanation to ONE paragraph.

Structure your answer in the following format:

Verdict: [CATEGORICAL LABEL]
Reasoning: [EXPLANATION]""")

response = chat.send_message(df['selftext'].iloc[500], **parameters)
print(response)