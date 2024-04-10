import argparse
import hydra
import omegaconf
import wandb
from torch.utils.data import DataLoader

from ponita.datasets.qm9 import QM9Dataset, collate_fn as collate_fn_nfc
from ponita.datasets.qm9_fc import  QM9DatasetFC, collate_fn_fc
from ponita.trainers.qm9_trainer import QM9Trainer


@hydra.main(version_base=None, config_path="./ponita/configs", config_name="qm9_regression")
def train(config):

    # Set log dir
    # if not config.logging.log_dir:
    #     hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    #     config.logging.log_dir = hydra_cfg['runtime']['output_dir']

    # Define the datasets
    # datasets = {split: QM9Dataset(split=split, target=config.training.target) for split in ['train', 'val', 'test']}
    # dataloaders = {
    #     split: DataLoader(dataset, batch_size=config.training.batch_size, shuffle=(split == 'train'), num_workers=config.training.num_workers, pin_memory=True, collate_fn=collate_fn)
    #     for split, dataset in datasets.items()}

    # Define the datasets
    if config.training.fully_connected:
        print('Using fully connected model')
        train_dataset = QM9DatasetFC(split='train', target=config.training.target)
        val_dataset = QM9DatasetFC(split='val', target=config.training.target)
        test_dataset = QM9DatasetFC(split='test', target=config.training.target)
        collate_fn = collate_fn_fc
    else:
        train_dataset = QM9Dataset(split='train', target=config.training.target)
        val_dataset = QM9Dataset(split='val', target=config.training.target)
        test_dataset = QM9Dataset(split='test', target=config.training.target)
        collate_fn = collate_fn_nfc

    # Define the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Load and initialize the model
    trainer = QM9Trainer(config, train_dataloader, val_dataloader, seed=config.optimizer.seed)
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