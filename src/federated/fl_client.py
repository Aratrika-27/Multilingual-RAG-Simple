"""
Federated learning client for the multilingual code-mix chatbot.
"""
import flwr as fl
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..config import GENERATOR_MODEL, FL_SERVER_ADDRESS

class MultilingualFLClient(fl.client.NumPyClient):
    """Federated learning client for the multilingual code-mix chatbot."""
    
    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        model_name: str = GENERATOR_MODEL,
        client_id: str = "client"
    ):
        """
        Initialize the federated learning client.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            model_name: Name of the model to use
            client_id: Client identifier
        """
        self.client_id = client_id
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        
        # Prepare datasets
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on the local dataset."""
        self.set_parameters(parameters)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/{self.client_id}",
            num_train_epochs=config.get("epochs", 1),
            per_device_train_batch_size=config.get("batch_size", 8),
            logging_dir=f"./logs/{self.client_id}",
            logging_steps=10,
            save_strategy="no",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
        )
        
        # Train the model
        trainer.train()
        
        # Return updated model parameters and metrics
        return self.get_parameters({}), len(self.train_dataset), {"client_id": self.client_id}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model on the local dataset."""
        self.set_parameters(parameters)
        
        if self.eval_dataset is None:
            return 0.0, 0, {"client_id": self.client_id}
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/{self.client_id}_eval",
            per_device_eval_batch_size=config.get("batch_size", 8),
            logging_dir=f"./logs/{self.client_id}_eval",
            save_strategy="no",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=self.eval_dataset,
        )
        
        # Evaluate the model
        metrics = trainer.evaluate()
        
        return metrics.get("eval_loss", 0.0), len(self.eval_dataset), {"client_id": self.client_id}

def prepare_dataset(
    df: pd.DataFrame,
    tokenizer: MT5Tokenizer,
    text_column: str = "text",
    target_column: str = "target",
    max_length: int = 512
) -> Dataset:
    """
    Prepare a dataset for training/evaluation.
    
    Args:
        df: DataFrame containing the data
        tokenizer: Tokenizer to use
        text_column: Column containing the input text
        target_column: Column containing the target text
        max_length: Maximum sequence length
        
    Returns:
        Dataset for training/evaluation
    """
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples[text_column],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[target_column],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Convert DataFrame to Dataset
    dataset = Dataset.from_pandas(df)
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def start_federated_client(
    train_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    model_name: str = GENERATOR_MODEL,
    client_id: str = "client",
    server_address: str = FL_SERVER_ADDRESS
) -> None:
    """
    Start a federated learning client.
    
    Args:
        train_df: Training data
        eval_df: Evaluation data (optional)
        model_name: Name of the model to use
        client_id: Client identifier
        server_address: Address of the federated learning server
    """
    # Load tokenizer
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_df, tokenizer)
    eval_dataset = prepare_dataset(eval_df, tokenizer) if eval_df is not None else None
    
    # Initialize client
    client = MultilingualFLClient(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_name=model_name,
        client_id=client_id
    )
    
    # Start client
    fl.client.start_numpy_client(server_address=server_address, client=client)
