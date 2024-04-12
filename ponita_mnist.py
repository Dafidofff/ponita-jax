import argparse
import hydra
import omegaconf
import wandb
from torch.utils.data import DataLoader

from ponita.datasets.mnist import MNISTPointCloud, collate_fn as collate_fn_mnist
# from ponita.datasets.mnist_superpixel import MNISTSuperPixelPointCloud, collate_fn as collate_fn_mnist
from ponita.trainers.mnist_trainer import MNISTTrainer


@hydra.main(version_base=None, config_path="./ponita/configs", config_name="mnist_classification")
def train(config):

    # Set log dir
    # if not config.logging.log_dir:
    #     hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    #     config.logging.log_dir = hydra_cfg['runtime']['output_dir']

    # Define the datasets
    print('Using fully connected model')
    train_dataset = MNISTPointCloud(split='train')
    val_dataset = MNISTPointCloud(split='val')
    collate_fn = collate_fn_mnist

    # Define the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Load and initialize the model
    trainer = MNISTTrainer(config, train_dataloader, val_dataloader, seed=config.optimizer.seed)
    trainer.create_functions()

    # Initialize wandb
    wandb.init(
        entity="equivariance",
        project="ponita-jax",
        dir=config.logging.log_dir,
        config=omegaconf.OmegaConf.to_container(config),
        mode='disabled' if config.logging.debug else 'online',
    )

    # Train model
    trainer.train_model(config.training.num_epochs)


if __name__ == "__main__":

    train()