import openai
import numpy as np
from textwrap import dedent
import re

instructions = """
    You are a tool for evaluating moral dilemmas.
    
    Please evaluate the following post on from the subreddit "Am I the Asshole".
    
    First, please describe the moral dilemma in no more than one paragraph.

    Then, determine whether the OP (original poster) of this post is the asshole in the scenario they describe.
    
    Provide a categorical label indicating your assessment of the scenario, from one of these choices:
        
    - YTA or "You're the Asshole" is for scenarios where the OP is at fault in their situation.
    - NTA or "Not the Asshole" is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
    - ESH or "Everyone Sucks Here" is for scenarios where both parties are to blame- both people involved in the scenario should be held responsible.
    - NAH or "No Assholes Here" is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
    - INFO or "Not Enough Info" is for situations where the OP never clarifies details that would determine the true judgment.

    Finally, please provide an explanation in no more than {length} of why you chose this label.
    
    Structure your answer in the following format:
    
    Dilemma: [DILEMMA]
    Verdict: [CATEGORICAL LABEL]
    Reasoning: [EXPLANATION]
    """

aita_prompt = """
    You are a tool for evaluating moral dilemmas.

    Please evaluate the following post from the subreddit "Am I the Asshole".

    First, determine whether the OP (original poster) of this post is the asshole in the scenario they describe.

    Provide a categorical label indicating your assessment of the scenario, from one of these choices:

    - YTA or "You're the Asshole" is for scenarios where the OP is at fault in their situation.
    - NTA or "Not the Asshole" is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
    - ESH or "Everyone Sucks Here" is for scenarios where both parties are to blame- both people involved in the scenario should be held responsible.
    - NAH or "No Assholes Here" is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
    - INFO or "Not Enough Info" is for situations where the OP never clarifies details that would determine the true judgment.

    Then, please provide an explanation in no more than {length} of why you chose this label.

    Structure your answer in the following format:

    Verdict: [CATEGORICAL LABEL]
    Reasoning: [EXPLANATION]
"""

davinci_prompt = """Please evaluate the following post on from the subreddit "Am I the Asshole":

    {post}

    Which choice below best desribes the original poster (OP) of the above text?
    Respond solely with A, B, C, D, or E.

    A. OP is the asshole: they are at fault in their situation.
    B. OP is not the asshole: they are NOT to blame and the other party described in their scenario is to blame.
    C. Everyone sucks here: both parties are to blame - both people involved in the scenario should be held responsible.
    D. No assholes here: neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
    E. Not enough info: OP never clarifies details that would determine the true judgment.

    Answer:

    """

identities = {
    "control": "",
    "moral_philosopher": "You are a moral philosopher.",
    "redditor": "You are an average user of the platform Reddit."
}


def create_system_message(identity, instructions=instructions, length="two paragraphs"):
    system_message = dedent(instructions.format(length=length))
    system_message = identity + "\n" + system_message
    return system_message.strip()


def is_op_an_asshole(post, system_message, model='gpt-4', temperature=0.4, openai_api_key=None):
    openai.api_key = openai_api_key

    response = openai.ChatCompletion.create(
    model=model,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{post}"}
    ],
    temperature=temperature)
    answer = response['choices'][0]['message']['content']
    return answer


def create_davinci_prompt(post):
    return dedent(davinci_prompt.format(post=post.strip()))


def get_probs(
    post, model='text-davinci-003', temperature=0.,
    logit_bias={32: 10, 33: 10, 34: 10, 35: 10, 36: 10},
    labels=['A', 'B', 'C', 'D', 'E'], n_logprobs=5, verbose=True
):
    # Run log probability query
    response = openai.Completion.create(
        model=model,
        prompt=create_davinci_prompt(post),
        temperature=temperature,
        logit_bias=logit_bias,
        logprobs=n_logprobs)
    # Extract top log probabilities
    top_logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
    # Calculate probabilities from log-probs
    probs = dict(zip(
        list(top_logprobs.keys()),
        np.exp(list(top_logprobs.values())) / np.sum(np.exp(list(top_logprobs.values())))
    ))
    # Get probability for each label
    probs_for_labels = {}
    # Iterate over labels
    for label in labels:
        # Check if label made it in probability
        if label in probs:
            probs_for_labels[label] = probs[label]
        else:
            # If missing, replace with lowest probability
            if verbose:
                print(f'Label {label} missing.')
            probs_for_labels[label] = min(probs.values())
    return probs_for_labels

def label_comment(row):
    labels = ['YTA', 'NTA', 'ESH', 'NAH']
    matches = [label for label in labels if re.search(r'\b' + re.escape(label) + r'\b', row['comment'])]
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return 'N/A'
    else:
        return None

