"""
Script to run the federated learning server.
"""
import argparse
import os
from pathlib import Path

from src.federated.fl_server import start_federated_server
from src.config import FL_SERVER_ADDRESS, FL_NUM_ROUNDS, FL_MIN_CLIENTS, MODELS_DIR

def main():
    """Run the federated learning server."""
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--address", type=str, default=FL_SERVER_ADDRESS, help="Server address")
    parser.add_argument("--rounds", type=int, default=FL_NUM_ROUNDS, help="Number of federated learning rounds")
    parser.add_argument("--min_clients", type=int, default=FL_MIN_CLIENTS, help="Minimum number of clients")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the model")
    
    args = parser.parse_args()
    
    # Determine save path
    save_path = args.save_path
    if save_path is None:
        save_path = os.path.join(MODELS_DIR, "federated_model")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Starting federated learning server at {args.address}...")
    print(f"Number of rounds: {args.rounds}")
    print(f"Minimum number of clients: {args.min_clients}")
    print(f"Model save path: {save_path}")
    
    # Start server
    start_federated_server(
        server_address=args.address,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        save_path=save_path
    )

if __name__ == "__main__":
    main()
