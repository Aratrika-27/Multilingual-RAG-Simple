"""
Script to run a federated learning client.
"""
import argparse
import os
import pandas as pd
from pathlib import Path

from src.federated.fl_client import start_federated_client
from src.config import FL_SERVER_ADDRESS, GENERATOR_MODEL, DATA_DIR

def main():
    """Run a federated learning client."""
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--client_id", type=str, required=True, help="Client identifier")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--eval_data", type=str, default=None, help="Path to evaluation data CSV")
    parser.add_argument("--model", type=str, default=GENERATOR_MODEL, help="Model name")
    parser.add_argument("--server_address", type=str, default=FL_SERVER_ADDRESS, help="Server address")
    
    args = parser.parse_args()
    
    print(f"Starting federated learning client {args.client_id}...")
    print(f"Training data: {args.train_data}")
    print(f"Evaluation data: {args.eval_data if args.eval_data else 'None'}")
    print(f"Model: {args.model}")
    print(f"Server address: {args.server_address}")
    
    # Load data
    train_df = pd.read_csv(args.train_data)
    eval_df = pd.read_csv(args.eval_data) if args.eval_data else None
    
    # Start client
    start_federated_client(
        train_df=train_df,
        eval_df=eval_df,
        model_name=args.model,
        client_id=args.client_id,
        server_address=args.server_address
    )

if __name__ == "__main__":
    main()
