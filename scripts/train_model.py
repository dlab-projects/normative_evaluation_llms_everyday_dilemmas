import os
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from moral_foundations_llms.loaders import RedditScenariosModule
from moral_foundations_llms.models import RedditScenarioModel
from transformers import AutoTokenizer


def run(path, config):
    TOKENIZER = 'roberta-large'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    # 'Relational_Obligation', 'Feelings', 'Fairness', 'Honesty', 'Social_Norms', 'Harm'
    axis = 'Social_Norms'
    # Configurations
    data_module_cfg = {
        'path': path,
        'output': axis,
        'batch_size': 16,
        'tokenizer': TOKENIZER}
    config['base'] = TOKENIZER

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename=f'checkpoint_{axis}_' + '{epoch}-{val_loss:.2f}',
        every_n_epochs=2,
        save_on_train_epoch_end=True)
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=3,
        mode='min',
        verbose=True)
    # Logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name=f"model_{axis}",
        version=0)

    # Data Module
    data_module = RedditScenariosModule(config=data_module_cfg)
    # Model
    model = RedditScenarioModel(config, tokenizer)

    # Trainer
    trainer = L.Trainer(
        accelerator="cuda",
        max_epochs=config['epoch'],
        devices=1,
        logger=logger,
        precision="bf16-mixed",
        callbacks=[checkpoint_callback,
                   early_stopping_callback])

    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    # Configuration dictionary
    path = 'data/aita_labels_text.feather'
    config = {
        'lr': 2e-6,
        'epoch': 5,
        'accumulate_grad_batches': 1,
        'warmup_ratio': 0.1
    }
    run(path, config)
