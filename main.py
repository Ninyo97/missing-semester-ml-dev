import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import argparse
import logging
import os
from datetime import datetime

from models.mlp import MLP
from datasets import load_mnist_datasets

from sklearn.metrics import accuracy_score


def setup_logging(log_level='INFO'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('mnist_training')

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, log_dir


def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train MLP on MNIST dataset')
    
    parser.add_argument('--batch-size-train', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--batch-size-test', type=int, default=1000,
                        help='Batch size for testing (default: 1000)')
    
    parser.add_argument('--hidden-features', type=int, default=128,
                        help='Number of hidden features in MLP (default: 128)')
    
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    
    # Logging arguments
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    
    return parser.parse_args()

def train(m, dl, max_epochs, lr, device, logger=None):
    if logger is None:
        logger = logging.getLogger('mnist_training')
    
    logger.info(f"Starting training for {max_epochs} epochs with lr={lr}")
    logger.debug(f"Using device: {device}")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=m.parameters(), lr=lr)
    
    m.to(device)
    logger.debug("Model moved to device")
    
    for epoch in range(max_epochs):
        epoch_loss = []
        logger.debug(f"Starting epoch {epoch+1}/{max_epochs}")
        
        for batch_idx, batch in enumerate(tqdm(dl, desc=f'Epoch {epoch+1}/{max_epochs}')):
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            preds = m(x)
            loss = criterion(preds, y)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            # Log batch loss occasionally for debugging
            if batch_idx % 1000 == 0:
                logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = torch.mean(torch.Tensor(epoch_loss)).item()
        logger.info(f"Epoch {epoch+1}/{max_epochs} completed - Average Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    logger.info("Training completed successfully")
    return m


def test(m, dl, device, logger=None):
    if logger is None:
        logger = logging.getLogger('mnist_training')
    
    logger.info("Starting model evaluation")
    logger.debug(f"Using device: {device}")
    
    predictions = []
    y_true = []
    
    m.eval()
    logger.debug("Model set to evaluation mode")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dl, desc='Testing')):
            x, y = batch
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(m(x), dim=1)
            predictions.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
            # Log progress occasionally
            if batch_idx % 500 == 0 and batch_idx > 0:
                logger.debug(f"Processed {batch_idx} test batches")

    accuracy = accuracy_score(y_true=np.array(y_true), y_pred=np.array(predictions))
    logger.info(f"Model evaluation completed - Accuracy: {accuracy:.4f}")
    
    return accuracy


def main():
    args = parse_arguments()
    
    logger, log_dir = setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("MNIST MLP Training Started")
    logger.info("=" * 60)
    
    logger.info(f"Configuration: {vars(args)}")
    logger.info(f"Log directory: {log_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)


    traindl, testdl, trainset, testset = load_mnist_datasets(
        batch_size_train=args.batch_size_train, 
        batch_size_test=args.batch_size_test 
    )

    model = MLP(in_features=784, hidden_features=args.hidden_features, out_features=10)
    
    logger.info("Starting training phase...")
    trained_model = train(model, traindl, args.epochs, args.lr, device, logger)
    
    logger.info("Starting testing phase...")
    acc = test(trained_model, testdl, device, logger)
    
    result_msg = f'Test Accuracy after training {args.epochs} epochs: {acc:.4f}'
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info(result_msg)
    logger.info("=" * 60)
    print(result_msg)
    
    model_info_path = os.path.join(log_dir, 'model_info.txt')
    with open(model_info_path, 'w') as f:
        f.write(f"Model Architecture: MLP\n")
        f.write(f"Input Features: 784\n")
        f.write(f"Hidden Features: {args.hidden_features}\n")
        f.write(f"Output Features: 10\n")
        f.write(f"Final Test Accuracy: {acc:.4f}\n")
        f.write(f"Training Configuration: {vars(args)}\n")
    
    logger.info(f"Model information saved to: {model_info_path}")
    

if __name__ == '__main__':
    main()
