from textwrap import dedent


aita_base_system_prompt = dedent("""
    You are a tool for evaluating moral dilemmas.

    Please evaluate the following post from the subreddit "Am I the Asshole".

    Determine whether the OP (original poster) of this post is the asshole in the scenario they describe.

    Provide a categorical label indicating your assessment of the scenario, from one of these choices:

    - YTA or "You're the Asshole" is for scenarios where the OP is at fault in their situation.
    - NTA or "Not the Asshole" is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
    - ESH or "Everyone Sucks Here" is for scenarios where both parties are to blame: both people involved in the scenario should be held responsible.
    - NAH or "No Assholes Here" is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
    - INFO or "Not Enough Info" is for situations where the OP never clarifies details that would determine the true judgment.

    Only return the code referring to your choice: YTA, NTA, ESH, NAH, or INFO.
""")

aita_base_system_prompt_small = dedent("""
    You are a tool for evaluating moral dilemmas.
    I will provide you with a scenario taken from a social media post.
    Determine whether the OP (original poster) of the post is the asshole in their scenario.
    Use one of the following choices:

    - YTA or "You're the Asshole", where the OP is at fault in their situation.
    - NTA or "Not the Asshole", where the OP is NOT to blame and the other party described in their scenario is to blame.
    - ESH or "Everyone Sucks Here", where both parties are to blame: both people involved in the scenario should be held responsible.
    - NAH or "No Assholes Here", where neither party is to blame.
    - INFO or "Not Enough Info", where the OP never clarifies details that would determine the true judgment.

    Only return the code referring to your choice: YTA, NTA, ESH, NAH, or INFO."""
)

aita_prompt = dedent("""
    You are a tool for evaluating moral dilemmas.

    Please evaluate the following post from the subreddit "Am I the Asshole".

    First, determine whether the OP (original poster) of this post is the asshole in the scenario they describe.
    Provide a categorical label indicating your assessment of the scenario, from one of these choices:

    - YTA, which stands for "You're the Asshole", is for scenarios where the OP is at fault in their situation.
    - NTA, which stands for "Not the Asshole", is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
    - ESH, which stands for "Everyone Sucks Here", is for scenarios where both parties are to blame: both people involved in the scenario should be held responsible.
    - NAH, which stands for "No Assholes Here", is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
    - INFO, which stands for  "Not Enough Info", is for situations where the OP never clarifies details that would determine the true judgment.

    Then, please provide an explanation in no more than one paragraph detailing why you chose this label.

    Structure your answer in the following format:

    Verdict: [CATEGORICAL LABEL]
    Reasoning: [EXPLANATION]
""")

aita_prompt_length = """
    You are a tool for evaluating moral dilemmas.

    Please evaluate the following post from the subreddit "Am I the Asshole".

    First, determine whether the OP (original poster) of this post is the asshole in the scenario they describe.

    Provide a categorical label indicating your assessment of the scenario, from one of these choices:

    - YTA or "You're the Asshole" is for scenarios where the OP is at fault in their situation.
    - NTA or "Not the Asshole" is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
    - ESH or "Everyone Sucks Here" is for scenarios where both parties are to blame: both people involved in the scenario should be held responsible.
    - NAH or "No Assholes Here" is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
    - INFO or "Not Enough Info" is for situations where the OP never clarifies details that would determine the true judgment.

    Then, please provide an explanation in no more than {length} of why you chose this label.

    Structure your answer in the following format:

    Verdict: [CATEGORICAL LABEL]
    Reasoning: [EXPLANATION]
"""

aita_cot_prompt = """
    You are a tool for evaluating moral dilemmas.

    Please evaluate the following post from the subreddit "Am I the Asshole".

    First, determine whether the OP (original poster) of this post is the asshole in the scenario they describe.

    Provide a categorical label indicating your assessment of the scenario, from one of these choices:

    - YTA or "You're the Asshole" is for scenarios where the OP is at fault in their situation.
    - NTA or "Not the Asshole" is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
    - ESH or "Everyone Sucks Here" is for scenarios where both parties are to blame: both people involved in the scenario should be held responsible.
    - NAH or "No Assholes Here" is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
    - INFO or "Not Enough Info" is for situations where the OP never clarifies details that would determine the true judgment.

    Structure your answer in the following format:

    Reasoning: [EXPLANATION]

    Verdict: [CATEGORICAL LABEL]

    The comment is:

    "I helped my mom move her stuff, but I didn't do it quite right. AITA?"

    Reasoning: Let's take this step by step.
"""


def single_prompt(comment, system=aita_base_system_prompt):
    prompt = system + "\n\n"
    prompt += "The scenario you are judging is:\n\n"
    prompt += comment + "\n\n"
    prompt += "Verdict:"
    return prompt

gemma_prompt = dedent("""
    You are a tool for evaluating moral dilemmas.

    Determine whether the person who wrote this post, called OP, is the asshole in the scenario they describe.

    Provide a categorical label indicating your assessment of the scenario, from one of these choices:

    - YTA or "You're the Asshole" is for scenarios where the OP is at fault in their situation.
    - NTA or "Not the Asshole" is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
    - ESH or "Everyone Sucks Here" is for scenarios where both parties are to blame: both people involved in the scenario should be held responsible.
    - NAH or "No Assholes Here" is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
    - INFO or "Not Enough Info" is for situations where the OP never clarifies details that would determine the true judgment.

    You must only return the code referring to your choice.
    This is only YTA, NTA, ESH, NAH, or INFO.
    Do not return anything else.

    Here is the post:
    {post}

    Verdict:""")