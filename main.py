import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import argparse
import logging
import os
import json
from datetime import datetime

from models.mlp import MLP
from datasets import load_mnist_datasets

from sklearn.metrics import accuracy_score
import copy


def setup_logging(log_level='INFO'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('mnist_training')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, log_dir, timestamp


def setup_tensorboard(log_dir, args, timestamp):
    """
    Setup TensorBoard logging with experiment-specific directory structure.
    
    Args:
        log_dir: Base log directory
        args: Training arguments
        timestamp: Timestamp string
    
    Returns:
        SummaryWriter: TensorBoard writer instance
    """
    # Create TensorBoard experiment name with key hyperparameters
    experiment_name = f"mlp_h{args.hidden_features}_bs{args.batch_size_train}_lr{args.lr}_ep{args.epochs}_{timestamp}"
    tensorboard_dir = os.path.join(log_dir, "tensorboard", experiment_name)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # Log hyperparameters
    hparams = {
        'hidden_features': args.hidden_features,
        'batch_size_train': args.batch_size_train,
        'batch_size_test': args.batch_size_test,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'log_level': args.log_level
    }
    
    # Log hyperparameters to TensorBoard
    writer.add_hparams(hparams, {})
    
    return writer, experiment_name

def save_model_checkpoint(model, args, accuracy, log_dir, timestamp, logger=None):
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_filename = f"mlp_h{args.hidden_features}_{timestamp}.pth"
    config_filename = f"config_h{args.hidden_features}_{timestamp}.json"
    
    model_path = os.path.join(checkpoint_dir, model_filename)
    config_path = os.path.join(checkpoint_dir, config_filename)
    
    logger.info(f"Saving model checkpoint to: {model_path}")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'in_features': 784,
            'hidden_features': args.hidden_features,
            'out_features': 10
        },
        'training_config': vars(args),
        'final_accuracy': accuracy,
        'timestamp': timestamp,
        'total_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    torch.save(checkpoint, model_path)
    logger.info(f"Model checkpoint saved successfully")
    
    # Save configuration as JSON
    logger.info(f"Saving configuration to: {config_path}")
    config_data = {
        'training_configuration': vars(args),
        'model_architecture': {
            'type': 'MLP',
            'in_features': 784,
            'hidden_features': args.hidden_features,
            'out_features': 10,
            'total_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        'training_results': {
            'final_accuracy': accuracy,
            'epochs_trained': args.epochs
        },
        'metadata': {
            'timestamp': timestamp,
            'model_file': model_filename,
            'log_directory': log_dir
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4, default=str)
    
    logger.info(f"Configuration saved successfully")
    
    return model_path, config_path

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
    
    # Checkpointing arguments
    parser.add_argument('--save-checkpoint', action='store_true',
                        help='Save model checkpoint after training')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to checkpoint file to load and evaluate')
    
    # TensorBoard arguments
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard logging (default: False)')
    parser.add_argument('--tensorboard-comment', type=str, default='',
                        help='Comment to add to TensorBoard experiment name')
    
    return parser.parse_args()

def train(m, dl, valdl, device, args, logger, writer=None, log_dir=None, timestamp=None):
    """Train the model with TensorBoard logging."""
    max_epochs = args.epochs
    lr = args.lr
    
    logger.info(f"Starting training for {max_epochs} epochs with lr={lr}")
    logger.debug(f"Using device: {device}")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=m.parameters(), lr=lr)
    
    m.to(device)
    logger.debug("Model moved to device")
    
    # Log model architecture to TensorBoard
    if writer is not None:
        # Create a dummy input to trace the model
        dummy_input = torch.randn(1, 784).to(device)
        try:
            writer.add_graph(m, dummy_input)
            logger.debug("Model graph added to TensorBoard")
        except Exception as e:
            logger.warning(f"Could not add model graph to TensorBoard: {e}")
    
    best_val_acc = 0.0
    best_model = None
    
    # Track training metrics
    train_losses = []
    val_accuracies = []

    for epoch in range(max_epochs):
        m.train()  # Set to training mode
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
            
            # Log batch loss to TensorBoard (every 100 batches)
            if writer is not None and batch_idx % 100 == 0:
                global_step = epoch * len(dl) + batch_idx
                writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            
            # Log batch loss occasionally for debugging
            if batch_idx % 1000 == 0:
                logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = torch.mean(torch.Tensor(epoch_loss)).item()
        train_losses.append(avg_loss)
        
        # Log epoch metrics to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
            writer.add_scalar('Learning_Rate', lr, epoch)
        
        logger.info(f"Epoch {epoch+1}/{max_epochs} completed - Average Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # Validation every 2 epochs
        if (epoch + 1) % 2 == 0:
            val_acc = val(m, valdl, device, logger)
            val_accuracies.append(val_acc)
            
            # Log validation accuracy to TensorBoard
            if writer is not None:
                writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            
            logger.info(f"Validation Accuracy after epoch {epoch+1}: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(m)
                if log_dir and timestamp:
                    save_model_checkpoint(
                        best_model, args, val_acc, log_dir, timestamp, logger
                    )
                logger.info(f"New best validation accuracy: {best_val_acc:.4f} - Model checkpoint saved")
    
    # Log final metrics to TensorBoard
    if writer is not None:
        writer.add_scalar('Metrics/Best_Validation_Accuracy', best_val_acc, 0)
        
        # Log hyperparameters with final results
        hparams = {
            'hidden_features': args.hidden_features,
            'batch_size_train': args.batch_size_train,
            'learning_rate': args.lr,
            'epochs': args.epochs
        }
        metrics = {
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'best_val_accuracy': best_val_acc,
            'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0
        }
        writer.add_hparams(hparams, metrics)

    logger.info("Training completed successfully")
    return best_model

def val(m, dl, device, logger=None):
    logger.info("Starting model validation")
    
    predictions = []
    y_true = []
    
    m.eval()    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dl, desc='Validating')):
            x, y = batch
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(m(x), dim=1)
            predictions.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            
    accuracy = accuracy_score(y_true=np.array(y_true), y_pred=np.array(predictions))
    logger.info(f"Model validation completed - Accuracy: {accuracy:.4f}")
    
    return accuracy

def test(m, dl, device, args, writer=None, logger=None):    
    """Test the model and return accuracy with TensorBoard logging."""
    if logger is None:
        logger = logging.getLogger('mnist_training')
    
    logger.info("Starting model evaluation")
    
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
    
    # Log test accuracy to TensorBoard
    if writer is not None:
        writer.add_scalar('TestAccuracy/train-batch-size', accuracy, args.batch_size_train)
        writer.add_scalar('TestAccuracy/hidden_features', accuracy, args.hidden_features)
    
    logger.info(f"Model evaluation completed - Accuracy: {accuracy:.4f}")
    
    return accuracy

def main():
    args = parse_arguments()
    
    logger, log_dir, timestamp = setup_logging(args.log_level)
    
    # Setup TensorBoard if enabled
    writer = None
    experiment_name = None
    if args.tensorboard:
        writer, experiment_name = setup_tensorboard(log_dir, args, timestamp)
        logger.info(f"TensorBoard enabled - Experiment: {experiment_name}")
        print(f"TensorBoard logging enabled. Run: tensorboard --logdir {os.path.join(log_dir, 'tensorboard')}")
    
    logger.info("=" * 60)
    logger.info("MNIST MLP Training Started")
    logger.info("=" * 60)
    
    logger.info(f"Configuration: {vars(args)}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Timestamp: {timestamp}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)

    logger.info("Loading MNIST datasets...")
    traindl, valdl, testdl, trainset, valset, testset = load_mnist_datasets(
        batch_size_train=args.batch_size_train, 
        batch_size_test=args.batch_size_test
    )
    logger.info(f"Datasets loaded - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")

    logger.info(f"Creating MLP model with {args.hidden_features} hidden features")
    model = MLP(in_features=784, hidden_features=args.hidden_features, out_features=10)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} trainable parameters")
    
    # Log model info to TensorBoard
    if writer is not None:
        writer.add_text('Model/Architecture', f'MLP with {args.hidden_features} hidden features')
        writer.add_text('Model/Parameters', f'{total_params:,} trainable parameters')
        writer.add_text('Training/Configuration', str(vars(args)))
    
    logger.info("Starting training phase...")
    trained_model = train(model, traindl, valdl, device, args, logger, writer, log_dir, timestamp)
    
    logger.info("Starting testing phase...")
    acc = test(trained_model, testdl, device, args, writer, logger)
    
    result_msg = f'Test Accuracy after training {args.epochs} epochs: {acc:.4f}'
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info(result_msg)
    logger.info("=" * 60)
    print(result_msg)
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        logger.info("TensorBoard logging completed")
    
    model_info_path = os.path.join(log_dir, 'model_info.txt')
    with open(model_info_path, 'w') as f:
        f.write(f"Model Architecture: MLP\n")
        f.write(f"Input Features: 784\n")
        f.write(f"Hidden Features: {args.hidden_features}\n")
        f.write(f"Output Features: 10\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Final Test Accuracy: {acc:.4f}\n")
        f.write(f"Training Configuration: {vars(args)}\n")
        f.write(f"Timestamp: {timestamp}\n")
        if args.tensorboard and experiment_name:
            f.write(f"TensorBoard Experiment: {experiment_name}\n")
            f.write(f"TensorBoard Directory: {os.path.join(log_dir, 'tensorboard', experiment_name)}\n")
    
    logger.info(f"Model information saved to: {model_info_path}")
    
if __name__ == '__main__':
    main()
