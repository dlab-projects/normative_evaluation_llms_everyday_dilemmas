AITA_LABELS = [
    'NTA',
    'YTA',
    'NAH',
    'ESH',
    'INF']
n_aita_labels = len(AITA_LABELS)

MODELS = [
    'reddit',
    'gpt3.5',
    'gpt4',
    'claude',
    'bison',
    'llama',
    'mistral',
    'gemma']
n_models = len(MODELS)

MORAL_AXES = [
    'fairness',
    'feelings',
    'harms',
    'honesty',
    'relational_obligation',
    'social_norms']
n_morals = len(MORAL_AXES)

REASON_COLS = [
    'top_comment',
    'gpt3.5_reason_1',
    'gpt4_reason_1',
    'claude_reason_1',
    'bison_reason_1',
    'llama_reason_1',
    'mistral_reason_1',
    'gemma_reason_1']
models_to_reasons = dict(zip(MODELS, REASON_COLS))

LABEL_COLS = [
    'reddit_label',
    'gpt3.5_label_1',
    'gpt4_label_1',
    'claude_label_1',
    'bison_label_1',
    'llama_label_1',
    'mistral_label_1',
    'gemma_label_1']
models_to_labels = dict(zip(MODELS, LABEL_COLS))

LABEL_NUM_COLS = [label_col + '_num' for label_col in LABEL_COLS]

MORAL_LABELS_COLS = [
    'fairness_label',
    'feelings_label',
    'harm_label',
    'honesty_label',
    'relational_obligation_label',
    'social_norms_label']

MORAL_MODEL_LABEL_DICT = {
    model: [f'{reason}_{moral}_label' for moral in MORAL_AXES]
    for model, reason in models_to_reasons.items()
}

MODEL_LABELS_PLOT = [
    'Redditor',
    'GPT 3.5',
    'GPT 4',
    'Claude Sonnet',
    'Bison',
    'Llama 2 7B',
    'Mistral 7B',
    'Gemma 7B']

MORAL_AXES_LABELS_PLOT = [
    'Fairness',
    'Feelings',
    'Harms',
    'Honesty',
    'Relational Obligation',
    'Social Norms'
]

AITA_LABELS_PLOT = [
    'NTA',
    'YTA',
    'NAH',
    'ESH',
    'INFO']