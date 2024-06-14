import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from moral_foundations_llms.loaders import RedditScenariosModule
from moral_foundations_llms.models import RedditScenarioModel
from transformers import AutoTokenizer


def run(path, config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    # DataModule
    data_module = RedditScenariosModule(config={'path': path, 'output': 'Harm'})
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        every_n_epochs=2,
        save_on_train_epoch_end=True)
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=3,
        mode='min',
        verbose=True)
    # Model
    model = RedditScenarioModel(config, tokenizer)
    logger = TensorBoardLogger("logs", name="model_Harm")
    # Trainer
    trainer = L.Trainer(
        accelerator="mps",
        max_epochs=config['epoch'],
        logger=logger,
        callbacks=[checkpoint_callback,
                   early_stopping_callback])

    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    # Configuration dictionary
    path = 'data/aita_labels_text.feather'
    config = {
        'lr': 2e-5,
        'epoch': 3,
        'accumulate_grad_batches': 1,
        'warmup_ratio': 0.1
    }
    run(path, config)
