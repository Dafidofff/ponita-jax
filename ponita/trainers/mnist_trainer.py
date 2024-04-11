from tqdm import tqdm
from typing import Any, Callable
from functools import partial

import wandb
import jax
import optax
import jax.numpy as jnp
from flax import struct, core
from flax import linen as nn

from ponita.trainers._base_trainer import BaseJaxTrainer
from ponita.nn.ponita_model import Ponita, FullyConnectedPonita
from ponita.utils.geometry.rotations import RandomSOd



class SNeFTrainState(struct.PyTreeNode):
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    rng: jnp.ndarray = struct.field(pytree_node=True)
    opt_state: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


class MNISTTrainer(BaseJaxTrainer):

    def __init__(
            self,
            config,
            train_loader,
            val_loader,
            seed,
    ):
        super().__init__(config, train_loader, val_loader, seed)

        # set ponita model vars
        self.in_channels_scalar = 1  # One-hot encoding molecules
        in_channels_vec = 0  # 
        out_channels_scalar = 10  # The target
        out_channels_vec = 0  # 

        # Transform
        self.train_aug = config.training.train_augmentation
        self.rotation_generator = RandomSOd(2)

        # Model
        model_func = FullyConnectedPonita if config.training.fully_connected else Ponita
        self.model = model_func(
            input_dim         = self.in_channels_scalar + in_channels_vec,
            hidden_dim        = config.ponita.hidden_dim,
            output_dim        = out_channels_scalar,
            num_layers        = config.ponita.num_layers,
            output_dim_vec    = out_channels_vec,
            radius            = config.ponita.radius,
            num_ori           = config.ponita.num_ori,
            basis_dim         = config.ponita.basis_dim,
            degree            = config.ponita.degree,
            widening_factor   = config.ponita.widening_factor,
            layer_scale       = config.ponita.layer_scale,
            task_level        = 'graph',
            multiple_readouts = config.ponita.multiple_readouts,
            batch_size        = config.training.batch_size
        )

        self.shift = 0
        self.scale = 1

        # Set dataset statistics
        # self.set_dataset_statistics(train_loader)

    def set_dataset_statistics(self, dataloader):
        print('Computing dataset statistics...')
        ys = []
        for data in tqdm(dataloader):
            ys.append(data['y'])
        ys = jnp.array(ys)
        # ys = np.concatenate(ys)
        self.shift = jnp.mean(ys)
        self.scale = jnp.std(ys)
        print('Mean and std of target are:', self.shift, '-', self.scale)

    def init_train_state(self):
        """Initializes the training state.

        Returns:
            TrainState: The training state.
        """
        # Initialize optimizer and scheduler
        self.optimizer = optax.adam(self.config.optimizer.learning_rate)

        # Random key
        key = jax.random.PRNGKey(self.config.optimizer.seed)

        # Split key
        key, model_key = jax.random.split(key)

        # Initialize model
        x = jnp.ones((4,28*28,1))
        pos = jnp.ones((4,28*28,2))
        edge_index = None
        batch = jnp.array([0, 1, 2, 3])

        # x, pos, edge_index, batch
        model_params = self.model.init(model_key, x, pos, edge_index, batch)

        # Create train state
        train_state = SNeFTrainState(
            params=model_params,
            opt_state=self.optimizer.init(model_params),
            rng=key
        )
        return train_state

    def create_functions(self):

        def step(state, batch):
            """Performs a single training step.

            Args:
                state (TrainState): The current training state.
                batch (dict): The current batch of data.
                train (bool): Whether we're training or validating. If training, we optimize both autodecoder and nef,
                    otherwise only autodecoder.

            Returns:
                TrainState: The updated training state.
            """

            # Split random key
            rng, key = jax.random.split(state.rng)

            if self.train_aug:
                rot = self.rotation_generator()
                if len(batch['pos'].shape) > 2:
                    batch['pos'] = jnp.einsum('ij, bnj->bnj', rot, batch['pos'])
                else:
                    batch['pos'] = jnp.einsum('ij, bj->bi', rot, batch['pos'])
            
            # Define loss and calculate gradients
            def loss_fn(params):
                logits, _ = self.model.apply(params, batch['x'], batch['pos'], None, batch['batch'])
                # one hot encoding of the target
                label = jax.nn.one_hot(batch['y'], 10)

                loss = jnp.mean(optax.softmax_cross_entropy(logits, label))
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(state.params)

            # Update autodecoder
            updates, opt_state = self.optimizer.update(grads, state.opt_state)
            params = optax.apply_updates(state.params, updates)

            return loss, state.replace(
                params=params,
                opt_state=opt_state,
                rng=key
            )

        # Jit functions
        self.apply_nef_jitted = jax.jit(self.model.apply)
        self.train_step = jax.jit(step)
        # self.train_step = step
        self.val_step = self.train_step

    def train_model(self, num_epochs, state=None):
        """Trains the model for the given number of epochs.

        Args:
            num_epochs (int): The number of epochs to train for.

        Returns:
            state: The final training state.
        """

        # Keep track of global step
        self.global_step = 0
        self.epoch = 0

        if state is None:
            state = self.init_train_state()

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            state = self.train_epoch(state, epoch)

            # Save checkpoint (ckpt manager takes care of saving every n epochs)
            self.save_checkpoint(state)

            # Validate every n epochs
            if epoch % self.config.test.test_interval == 0:
                self.validate_epoch(state)
        return state

    def train_epoch(self, state, epoch):
        # Loop over batches
        losses = 0
        for batch_idx, batch in enumerate(self.train_loader):
 
            loss, state = self.train_step(state, batch)
            losses += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                wandb.log({'train_mse_step': loss})
                self.update_prog_bar(loss, step=batch_idx)

            # if self.global_step % self.config.logging.visualize_every_n_steps == 0:
            #     self.visualize_batch(state, batch, name='train/recon')

            # Increment global step
            self.global_step += 1

        # Update epoch loss
        self.train_mse_epoch = losses / len(self.train_loader)
        wandb.log({'train_mse_epoch': self.train_mse_epoch})
        wandb.log({'epoch': epoch})
        return state
    
    def validate_epoch(self, state):
        """ Validates the model.

        Args:
            state: The current training state.
        """
        # Loop over batches
        losses = 0
        for batch_idx, batch in enumerate(self.val_loader):
            loss, _ = self.val_step(state, batch)
            losses += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                wandb.log({'val_mse_step': loss})
                self.update_prog_bar(loss, step=batch_idx, train=False)

            # Increment global step
            self.global_val_step += 1

        # Update epoch loss
        self.val_mse_epoch = losses / len(self.val_loader)
        wandb.log({'val_mse_epoch': self.val_mse_epoch}, commit=False)