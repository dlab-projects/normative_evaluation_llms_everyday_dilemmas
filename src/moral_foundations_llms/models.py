import torch
import lightning as L

from torch.nn import Dropout, Linear, Module, ReLU, Sigmoid
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torchmetrics.classification import BinaryAccuracy
from transformers import get_linear_schedule_with_warmup


def masked_average_pooling(hidden_states, attention_mask):
    """
    Perform average pooling on the hidden states, taking the attention mask into account.

    Args:
        hidden_states (torch.Tensor): The hidden states of shape (batch_size, sequence_length, hidden_size).
        attention_mask (torch.Tensor): The attention mask of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: The pooled hidden states of shape (batch_size, hidden_size).
    """
    # Apply the attention mask to the hidden states
    masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)

    # Sum the hidden states and the mask
    sum_hidden_states = masked_hidden_states.sum(dim=1)
    sum_mask = attention_mask.sum(dim=1, keepdim=True)

    # Compute the average hidden states
    average_hidden_states = sum_hidden_states / sum_mask
    return average_hidden_states


class RedditScenarioPredictor(Module):
    def __init__(
        self, base='roberta-base', n_dense=128, dropout_rate=0.4,
        outputs=None
    ):
        super(RedditScenarioPredictor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        self.base_model = AutoModel.from_pretrained(base)
        self.dense_layer = Linear(
            in_features=self.base_model.config.hidden_size,
            out_features=n_dense)
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(p=dropout_rate)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.output = Linear(
            in_features=n_dense,
            out_features=1)

    def forward(self, input_ids, attention_mask):
        # Run inputs through base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain hidden state
        x = outputs.last_hidden_state
        # Perform average pooling across the token dimension, with masking
        x = masked_average_pooling(x, attention_mask)
        # Dense layer, with dropout, and ReLU activation
        x = self.dense_layer(x)
        x = self.dropout(x)
        x = self.relu(x)
        # Dense layer to produce score
        x = self.output(x)
        output = self.sigmoid(x) 
        return output

    def tokenize(self, texts):
        encoding = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        return encoding['input_ids'], encoding['attention_mask']

    def predict(self, texts):
        input_ids, attention_mask = self.tokenize(texts)
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
        return output


class RedditScenarioModel(L.LightningModule):
    def __init__(self, config, tokenizer):
        """method used to define our model parameters"""
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer
        self.model = RedditScenarioPredictor()
        self.save_hyperparameters()
        self.loss = torch.nn.BCELoss()
        self.accuracy = BinaryAccuracy()


    def configure_optimizers(self):
        # Optimizer
        optimizer = AdamW(self.parameters(), lr=float(self.cfg['lr']))

        # Scheduler with warmup
        #num_devices = (
        #    torch.cuda.device_count()
        #    if self.trainer.devices == -1
        #    else int(self.trainer.devices)
        #)
        num_devices = 1
        total_steps = (
            len(self.trainer.datamodule.train_dataloader())
            // self.cfg['accumulate_grad_batches']
            // num_devices
            * self.cfg['epoch']
        )
        warmup_steps = int(total_steps * self.cfg['warmup_ratio'])
        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            ),
            "interval": "step",  # runs per batch rather than per epoch
            "frequency": 1,
            "name": "learning_rate",
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        output = self(batch['input_ids'], batch['attention_mask']).squeeze()
        loss = self.loss(output, batch['outputs'])
        accuracy = self.accuracy(output, batch['outputs'])
        self.log("train_loss", loss)
        self.log("acc", accuracy)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self(batch['input_ids'], batch['attention_mask']).squeeze()
        loss = self.loss(output, batch['outputs'])
        accuracy = self.accuracy(output, batch['outputs'])
        self.log("val_loss", loss)
        self.log("val_acc", accuracy)
        return {"loss": loss}

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

