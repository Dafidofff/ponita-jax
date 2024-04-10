import os
import wandb
import tqdm
import numpy as np
import jax.numpy as jnp

# Checkpointing
from orbax import checkpoint


class  BaseJaxTrainer:

    def __init__(
            self,
            config,
            train_loader,
            val_loader,
            seed,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.seed = seed

        # Keep track of training state
        self.global_step = 0
        self.epoch = 0

        # Keep track of state of validation
        self.val_epoch = 0
        self.global_val_step = 0
        self.total_val_epochs = 0

        # Description strings for train and val progress bars
        self.train_mse_epoch, self.val_mse_epoch = np.inf, np.inf
        self.prog_bar_desc = """{state} :: epoch - {epoch}/{total_epochs} | step - {step}/{global_step} :: mse step {loss:.4f} -- train mse epoch {train_mse_epoch:.4f} -- val mse epoch {val_mse_epoch:.4f}"""
        self.prog_bar = tqdm.tqdm(
            desc=self.prog_bar_desc.format(
                state='Training',
                epoch=self.epoch,
                total_epochs=self.config.training.num_epochs,
                step=0,
                global_step=len(self.train_loader),
                loss=jnp.inf,
                train_mse_epoch=self.train_mse_epoch,
                val_mse_epoch=self.val_mse_epoch
            ),
            total=len(self.train_loader)
        )
        
        # Set checkpoint options
        if self.config.logging.checkpoint:
            checkpoint_options = checkpoint.CheckpointManagerOptions(
                save_interval_steps=config.logging.checkpoint_every_n_epochs,
                max_to_keep=config.logging.keep_n_checkpoints,
            )
            orbax_checkpointer = checkpoint.PyTreeCheckpointer()
            self.checkpoint_manager = checkpoint.CheckpointManager(
                directory=os.path.abspath(config.logging.log_dir + '/checkpoints'),
                checkpointers=orbax_checkpointer,
                options=checkpoint_options,
            )

    def save_checkpoint(self, state):
        """ Save the current state to a checkpoint

        Args:
            state: The current training state.
        """
        if self.config.logging.checkpoint:
            self.checkpoint_manager.save(step=self.epoch, items={'state': state, 'cfg': self.config})

    def load_checkpoint(self):
        """ Load the latest checkpoint"""
        return self.checkpoint_manager.restore(self.checkpoint_manager.latest_step())

    def update_prog_bar(self, loss, step, train=True):
        """ Update the progress bar.

        Args:
            desc: The description string.
            loss: The current loss.
            epoch: The current epoch.
            step: The current step.
        """
        # If we are at the beginning of the epoch, reset the progress bar
        if step == 0:
            # Depending on whether we are training or validating, set the total number of steps
            if train:
                self.prog_bar.total = len(self.train_loader)
            else:
                self.prog_bar.total = len(self.val_loader)
            self.prog_bar.reset()
        else:
            self.prog_bar.update(self.config.logging.log_every_n_steps)

        if train:
            global_step = self.global_step
            epoch = self.epoch
            total_epochs = self.config.training.num_epochs
        else:
            global_step = self.global_val_step
            epoch = self.val_epoch
            total_epochs = self.total_val_epochs

        self.prog_bar.set_description_str(
            self.prog_bar_desc.format(
                state='Training' if train else 'Validation',
                epoch=epoch,
                total_epochs=total_epochs,
                step=step,
                global_step=len(self.train_loader),
                loss=loss,
                train_mse_epoch=self.train_mse_epoch,
                val_mse_epoch=self.val_mse_epoch
            ),
        )
