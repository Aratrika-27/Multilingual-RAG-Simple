"""
Federated learning server for the multilingual code-mix chatbot.
"""
import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..config import FL_SERVER_ADDRESS, FL_NUM_ROUNDS, FL_MIN_CLIENTS

class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Strategy for federated learning with model saving."""
    
    def __init__(
        self,
        save_path: str = "./models/federated_model",
        *args,
        **kwargs
    ):
        """
        Initialize the strategy.
        
        Args:
            save_path: Path to save the model
            *args: Additional arguments for FedAvg
            **kwargs: Additional keyword arguments for FedAvg
        """
        super().__init__(*args, **kwargs)
        self.save_path = save_path
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Parameters]:
        """Aggregate model parameters and save the model."""
        # Call aggregate_fit from base class (FedAvg)
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Convert parameters to NumPy arrays
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Save aggregated parameters
            print(f"Saving round {server_round} aggregated parameters...")
            np.save(f"{self.save_path}_round_{server_round}.npy", aggregated_ndarrays)
            
            # Save the latest model
            print(f"Saving latest model to {self.save_path}_latest.npy")
            np.save(f"{self.save_path}_latest.npy", aggregated_ndarrays)
        
        return aggregated_parameters

def start_federated_server(
    server_address: str = FL_SERVER_ADDRESS,
    num_rounds: int = FL_NUM_ROUNDS,
    min_clients: int = FL_MIN_CLIENTS,
    save_path: str = "./models/federated_model"
) -> None:
    """
    Start a federated learning server.
    
    Args:
        server_address: Address of the federated learning server
        num_rounds: Number of federated learning rounds
        min_clients: Minimum number of clients required for training
        save_path: Path to save the model
    """
    # Define strategy
    strategy = SaveModelStrategy(
        save_path=save_path,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
    )
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
