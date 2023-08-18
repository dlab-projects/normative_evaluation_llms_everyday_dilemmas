import openai
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


davinci_prompt = """Please evaluate the following post on from the subreddit "Am I the Asshole":

    {post}

    Provide a single choice below indicating whether the author of this post is the asshole in the scenario they describe:

    A. OP is the asshole: they are at fault in their situation.
    B. OP is not the asshole: they are NOT to blame and the other party described in their scenario is to blame.
    C. Everyone sucks here: both parties are to blame - both people involved in the scenario should be held responsible.
    D. No assholes here: neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
    E. Not enough info: OP never clarifies details that would determine the true judgment.
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
    return dedent(davinci_prompt.format(post=post))



def label_comment(row):
    labels = ['YTA', 'NTA', 'ESH', 'NAH']
    matches = [label for label in labels if re.search(r'\b' + re.escape(label) + r'\b', row['comment'])]
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return 'N/A'
    else:
        return None

