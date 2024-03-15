import vertexai
from vertexai.language_models import TextGenerationModel

vertexai.init(project="dlab-hsi-dev", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.4,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison@001")
response = model.predict(
    """You are a tool for evaluating moral dilemmas.

Please evaluate the following post from the subreddit \"Am I the Asshole\".

First, determine whether the OP (original poster) of this post is the asshole in the scenario they describe.

Provide a categorical label indicating your assessment of the scenario, from one of these choices:

- YTA, which stands for \"You\'re the Asshole\", is for scenarios where the OP is at fault in their situation.
- NTA, which stands for \"Not the Asshole\", is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
- ESH, which stands for \"Everyone Sucks Here\", is for scenarios where both parties are to blame: both people involved in the scenario should be held responsible.
- NAH, which stands for \"No Assholes Here\", is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
- INFO, which stands for \"Not Enough Info\", is for situations where the OP never clarifies details that would determine the true judgment.

Then, please provide an explanation in no more than on paragraph of why you chose this label.

Structure your answer in the following format:

Verdict: [CATEGORICAL LABEL]
Reasoning: [EXPLANATION]

Here is the dilemma you will evaluate:

\"I have had a group of four friends since elementary school. The five of us have stayed close friends through elementary, middle and high school (going to the same schools) and then through university (separate universities, but we would visit often and stayed very close). We are all 26yo guys, and have been friends for 19 years now. \\n\\nOne of the guys in the group, Brandon, is getting married in January. Brandon asked the other three guys to be groomsmen in his wedding, alongside his two brothers and two of his friends from college. this left me the only homie in the group who wasn\'t a groomsman. I was upset when I first realised, but I talked to my parents about it and they reminded me its Brandon\'s wedding and not a \'group event\', he can have who he likes up there, and just because im not a groomsman doesn\'t mean Brandon\'s doesn\'t consider me a friend. and that he does already have 7 people up there beside him, which is a lot. \\n\\nmy parents are the only one I ever told I was upset about it, and now I think im pretty well over it. they\'ve had a few grooms-party gatherings, like they went for drinks after they got fitted for suits, and went golfing together, and Brandon and his fiancé had a bbq for their wedding party - that\'s always a weird reminder for me. \\n\\nmy friends and I usually go on a trip in December to watch a football game. we started the now tradition in our first year in university, and have been going every year since. its always just been the 5 of us friends, and we go for like 3/4 days. on Tuesday my three friends came to me and wanted to know my opinion on inviting the other groomsmen on the trip as a surprise to Brandon. the three of them were clearly all for this idea, and really wanted me to say yes. \\n\\nI told them I wasn\'t sure, I had to think about it (which was awkward because it was obvious they thought I was just going to say yes). I spoke to them about it today, and said honestly I dont want to go on a trip being the only non-groomsman. I know Brandon\'s brothers, and I\'ve met his college friends, and they\'re all cool, but I dont want to be the clear odd man out. I told my friends that they should do it, I just won\'t go this year - which was fine for me because I could do with saving some money because I have a separate destination wedding to go to in February now. \\n\\nthe other guys won\'t invite the other groomsmen if it means I won\'t come. but its clear they\'re also annoyed at them not being able to invite them because of me. one of my friends spoke to me separately and he told me he really thinks im not fair or a good friend, and asked if its because I resent not being a groomsman. feels like any decision I make besides agreeing to go on the trip with the four other groomsmen is going to make them mad at me. \\n\\nAITA for backing out of the trip if I am going to be the only non-groomsman?\"""",
    **parameters
)
print(f"Response from Model: {response.text}")