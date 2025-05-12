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

    First, think about the moral dilemma of the scenario. 

    Determine whether the OP (original poster) of this post is the asshole in the scenario they describe.

    Provide a categorical label indicating your assessment of the scenario, from one of these choices:

    - YTA or "You're the Asshole" is for scenarios where the OP is at fault in their situation.
    - NTA or "Not the Asshole" is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
    - ESH or "Everyone Sucks Here" is for scenarios where both parties are to blame: both people involved in the scenario should be held responsible.
    - NAH or "No Assholes Here" is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
    - INFO or "Not Enough Info" is for situations where the OP never clarifies details that would determine the true judgment.

    Please additionally provide an explanation in no more than paragraph of why you chose the label.
    Explain your reasoning step-by-step.

    Structure your answer in the following format:

    Reasoning: Let's think about this step-by-step. [EXPLANATION]
    Verdict: [CATEGORICAL LABEL]
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

comment_label_prompt = """You are a tool for labeling social media posts.

    Please evaluate the following comment on from the subreddit "Am I the Asshole".

    Determine whether the following comment is expressing one of the following judgements:

    - YTA or "You're the Asshole" is for scenarios where the OP is at fault in their situation.
    - NTA or "Not the Asshole" is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
    - ESH or "Everyone Sucks Here" is for scenarios where both parties are to blame- both people involved in the scenario should be held responsible.
    - NAH or "No Assholes Here" is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
    - INFO or "Not Enough Info" is for situations where the OP never clarifies details that would determine the true judgment.
    - NONE when it seems like no specific judgement is being rendered.

    You should lean toward assigning "NONE" if it is not clear what the judgement is.
    Usually, commenters will use one of the labels in their post, but sometimes they imply it.
    A comment simply expressing a strong sentiment is not enough to guarantee a YTA assignment, for example.
    Additionally, a comment that implies multiple judgements, usually conditioned on unknown information, should also be assigned NONE.

    Please return the label you think should be assigned to the comment."""


escalation_prompt = dedent("""You are a tool for classifying scenarios in moral dilemmas.

    Specifically, you will determine whether a moral dilemma posed by a social
    media user features "escalation" by the OP. Escalation is defined as when
    the original poster - the OP - makes the conflict worse by carrying out an
    action on their own, even if they were initially wronged.

    To be clear, you are only determining whether the person writing the post -
    OP - escalated the conflict. You are not determining whether any person
    described escalated.

    Here are a couple examples of escalation:

    Example 1:
    "I'm an amateur baker and I like to make cakes for people for
    their birthdays. My ex doesn't like sweet stuff but our kids like helping me
    make a cake for him so I normally make one for him anyway for their sakes.
    This year his girlfriend planned a party for him and she told me she had
    already ordered him a cake so I didn't need to make one. I told her that was
    fine but my daughter kept asking me when we would make her dad's cake and
    was upset when I said we wouldn't because she assumed I didn't like my ex
    and that's why I wouldn't make him one.

    Since the cake has always been more for the kids than for my ex, I decided
    to just make him a small one since it's not like he eats cake anyway and I
    don't want my kids to think I dislike their dad. I told his girlfriend
    beforehand that I was going to make a small cake and she told me not to and
    that I couldn't bring it to the party. I told her I had to bring it to the
    party as that's when the kids would give him his gifts. I offered to come
    early and give it to him before the other guests arrived and then they could
    put it away and nobody else would know about my cake but she didn't agree
    with me and repeatedly told me not to bring it.

    I did take it with me in the end and now she keeps complaining to mutual
    friends that I did it on purpose to ruin the party and calling me weird for
    making him a cake after I was told not to.

    AITA?"

    Example 1 contains escalation because the original conflict was about
    bringing the cake. The OP escalated the conflict by bringing the cake to the
    party despite what their ex's girlfriend asked.

    Example 2: "Last weekend was my (M28) birthday. My girlfriend (F25) had
    asked what I wanted to do and I said I wanted to watch my favorite movie
    trilogy, LOTR. I don't think my girlfriend was thrilled but she didn't say
    anything and agreed. She has seen them before and I don't think she really
    likes them very much but she knows I love them so she doesn't really say
    anything besides they aren't really her thing. 

    But I really wanted to make a day of watching them and I went over to her
    house because she has a really big comfortable couch.  About ten minutes
    into the first movie and I look over and she is browsing on her phone. I was
    a little miffed but didn't say anything. She basically scrolled through her
    phone the entire movie. When we started the second movie, she opened a
    bottle of wine and proceeded to drink the whole thing, while still sitting
    on her phone. I was pretty irritated at this point because she wasn't even
    paying attention at all. 

    The third movie started and by then she had opened another bottle of wine
    and was asleep within the first twenty minutes. I was really mad at that
    point and just left and went home. 

    A few hours later I got a text asking where I went. I told her I was mad
    that she couldn't pay attention to my favorite movies on my birthday. She
    told me I was an asshole and to grow the hell up. I've texted her a couple
    times but she hasn't responded. AITA?"

    Example 2 contains escalation because the original conflict was about OP's
    girlfriend falling asleep during Lord of the Rings. OP escalated the
    conflict by going home instead of talking with his girlfriend.

    Example 3:
    "I (f, 28) have been with my husband "Shaun" (m, 33) for 2 years, Married
    for 5 months. Most of his family are decent people but his mom can be a
    little of a passive-aggressive and tends to criticize me a lot. Shaun sees
    it as "her still not getting used to me being around" but IDK because she
    treats his ex "Julissa" good. MIL says that Julissa has been around the
    family for age and her past with Shaun never affected her relationship with
    her. Fine, I never minded her attending every holiday and being around til
    yesterday.

    We had Thanksgiving dinner at my MIL's house. Shaun went there before me and
    when I arrived it was already dinner time. Everyone was seated and I saw that
    all chairs were taken. I asked MIL why she didn't save me a seat and she said
    "sorry" and that one of her granddaughters decided to show up last minute and
    the chair was taken. I looked at her then at Julissa who was sitting next to
    shaun and tried to point out how I was more deserving of her chair since I'm the
    DIL (I know shouldn't have said it I know..I know) MIL flatout said that Julissa
    is as much FAMILY as me, and that it was rude to imply otherwise. Julissa was
    nodding confidently while glancing at me. I was so upset I wanted to leave but
    decided to just sit on my husband's lap and act as casual as possible. I sat on
    his lap asking if he was okay with it (don't worry I'm petite, he's strong
    built) and started eating so casually while smiling and complimenting the food
    and mentioning to Shaun how warm and comfortable his lap was now and then. The
    table went awkwardly silence. BIL would try to break the silence and change the
    subject but it somehow goes back to being awkward. MIL AND Julissa were barely
    eating and were staring at each other than at me eyes wide open.


    Minutes later, Julissa excused herself to the bathroom and so did MIL. It was
    still awkward but I did my best to focus on dinner. Shaun was eating as well.
    Later, there was just so much tension and MIL was barely able to speak after
    Julissa left (early, like right after dinner). Shaun and I went home and MIL
    tried calling but then called Shaun and texted me saying what I did was
    inappropriate and that I ruined Thanksgiving dinner and made it awkward. She
    said it wasn't her fault chairs were taken and I could've dragged a chair from
    the kitchen but acted childishly and made Julissa (and family) uncomfortable
    with how inappropriate I was."

    Example 3 contains escalation because the original conflict was about how
    OP's mother in law and Julissa treated her. Instead of voicing her concerns,
    OP escalated the conflict by sitting on her boyfriend's lap and making the
    dinner awkward. This is an example of escalation where OP was wronged in the first placed.

    For the following post, please say whether it features escalation by the OP
    or not. Only return TRUE if it contains escalation by the OP, and FALSE if
    it does not. Then, return a short explanation as to why it was escalation.

    If the post contains escalation by anyone other than OP, and OP did not
    escalate, return FALSE. If nobody escalated in the dilemma, return FALSE. If
    multiple people, including OP, escalates, return TRUE. You are only basing
    your label on whether OP escalates or not.
    
    Structure you answer as follows:
    Verdict: [VERDICT]
    Reasoning: [REASONING]
    """)


identities = {
    "control": "",
    "moral_philosopher": "You are a moral philosopher.",
    "redditor": "You are an average user of the platform Reddit."
}


def create_system_message(identity, instructions=instructions, length="two paragraphs"):
    system_message = dedent(aita_prompt.format(length=length))
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


def clean_single_label(s):
    return s.strip().lower().replace('.', '').replace('info', 'inf').replace('*', '').upper()


def label_detector(s, check_first_line=True, return_first_if_multiple=True):
    # Would-be replacements
    s = s.replace('YWBTA', 'YTA')
    labels = ['YTA', 'NTA', 'ESH', 'NAH', 'INFO', 'INF',
              '^yta ', '^nta ', '^esh ', '^nah ',
              'Yta', 'Nta', 'Esh', 'Nah', 'NaH',
              '\*\*yta\*\*', '\*\*nta\*\*', '\*\*esh\*\*', '\*\*nah\*\*', '\*\*info\*\*',
              'yta\n', 'nta\n', 'esh\n', 'nah\n', 'info\n',
              ' yta ', ' nta ', ' esh ', ' nah ', ' info ',
              '^yta\.', '^nta\.', '^esh\.', '^nah\.', '^info\.',
              'Yta\.', 'Nta\.', 'Esh\.', 'Nah\.', 'Info\.']
    pattern = '|'.join(labels)
    if check_first_line:
        matches = re.findall(pattern, s.strip().partition('\n')[0])
        if len(matches) == 1:
            return clean_single_label(matches[0])
        
    matches = re.findall(pattern, s)
    matches = [clean_single_label(match) for match in matches]

    if len(matches) == 1:
        return clean_single_label(matches[0])
    elif len(matches) == 0:
        return 'NO MATCH'
    else:
        if return_first_if_multiple:
            return clean_single_label(matches[0])
        matches = list(set(matches))
        if len(matches) == 1:
            return clean_single_label(matches[0])
        else:
            return 'OVERMATCH'

def label_to_num(df):
    """
    Converts categorical labels in a DataFrame to numerical values.

    This function replaces specific string labels in a DataFrame with their corresponding
    numerical representations. The mapping is as follows:
        - 'NTA' (Not the Asshole) -> 0
        - 'YTA' (You're the Asshole) -> 1
        - 'ESH' (Everyone Sucks Here) -> 2
        - 'NAH' (No Assholes Here) -> 3
        - 'INF' (Not Enough Info) -> 4

    Args:
        df (pd.DataFrame): The input DataFrame containing categorical labels.

    Returns:
        pd.DataFrame: A DataFrame with the categorical labels replaced by their numerical values.
    """
    # Replace the categorical labels with their corresponding numerical values
    return df.replace({
        'NTA': 0,  # Not the Asshole
        'YTA': 1,  # You're the Asshole
        'ESH': 2,  # Everyone Sucks Here
        'NAH': 3,  # No Assholes Here
        'INF': 4   # Not Enough Info
    }).infer_objects(copy=False)  # Ensure the DataFrame's object types are inferred without copying