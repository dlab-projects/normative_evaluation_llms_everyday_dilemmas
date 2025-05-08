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

LLMs = [
    'gpt3.5',
    'gpt4',
    'claude',
    'bison',
    'llama',
    'mistral',
    'gemma']
n_llms = len(LLMs)

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

REASON_COLS_2 = [
    'top_comment',
    'gpt3.5_reason_2',
    'gpt4_reason_2',
    'claude_reason_2',
    'bison_reason_2',
    'llama_reason_2',
    'mistral_reason_2',
    'gemma_reason_2']
models_to_reasons_2 = dict(zip(MODELS, REASON_COLS_2))

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

LLM_LABEL_COLS = LABEL_COLS[1:]

LABEL_COLS_2 = [
    'gpt3.5_label_2',
    'gpt4_label_2',
    'claude_label_2',
    'bison_label_2',
    'llama_label_2',
    'mistral_label_2',
    'gemma_label_2']
models_to_labels_2 = dict(zip(MODELS, LABEL_COLS_2))

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
    'GPT-3.5',
    'GPT-4',
    'Claude Haiku',
    'PaLM 2 Bison',
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

ALL_LABELS = [
    'reddit_label',
    'gpt3.5_label_1',
    'gpt3.5_label_2',
    'gpt3.5_label_3',
    'gpt4_label_1',
    'claude_label_1',
    'claude_label_2',
    'claude_label_3',
    'bison_label_1',
    'bison_label_2',
    'bison_label_3',
    'llama_label_1',
    'llama_label_2',
    'llama_label_3',
    'mistral_label_1',
    'mistral_label_2',
    'mistral_label_3',
    'gemma_label_1',
    'gemma_label_2',
    'gemma_label_3']

ALL_REASONS = [
    'top_comment',
    'gpt3.5_reason_1',
    'gpt3.5_reason_2',
    'gpt3.5_reason_3',
    'gpt4_reason_1',
    'claude_reason_1',
    'claude_reason_2',
    'claude_reason_3',
    'bison_reason_1',
    'bison_reason_2',
    'bison_reason_3',
    'llama_reason_1',
    'llama_reason_2',
    'llama_reason_3',
    'mistral_reason_1',
    'mistral_reason_2',
    'mistral_reason_3',
    'gemma_reason_1',
    'gemma_reason_2',
    'gemma_reason_3']