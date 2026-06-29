import random
import numpy as np
import torch
from tqdm import tqdm
from .logger import logger
import csv
import warnings

from hatorch.models import Model


class Trainer:
    """High‑level utility to train, evaluate, save and resume any **hatorch** model.

    Parameters
    ----------
    model : hatorch.models.Model
        Wrapper around a ``torch.nn.Module`` supplying the forward pass.
    optimizer : torch.optim.Optimizer
        Optimiser instance that updates ``model`` parameters.
    criterion : torch.nn.Module
        Loss function used for both training and validation.
    device : torch.device
        Target device on which the network will run (e.g. ``torch.device('cuda')``).
    seed : int, optional
        Reproducibility seed. When supplied, it fixes Python, NumPy and
        PyTorch RNG states.
    scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        Learning‑rate scheduler stepped *every training iteration* (after each
        optimiser step). If ``None`` no scheduling is performed.
    saving_uri : str, optional
        **Base path** – *without extension* – where checkpoints are written.
        Each time a checkpoint is triggered the trainer appends the pattern
        ``"_epoch_<E>.state"`` (with ``<E>=epoch+1``) to this base path and
        serialises a full training snapshot via :pyfunc:`torch.save`.  For
        example, with ``saving_uri="runs/checkpoint/model"`` the file created
        at epoch 10 will be ``runs/checkpoint/model_epoch_10.state``.
    saving_freq : int, optional
        Save a checkpoint every *N* epochs. *Must* be specified together with
        ``saving_uri`` and vice‑versa; otherwise a ``ValueError`` is raised.
    resume_from_previous_state : str, optional
        Path to an existing ``*.state`` file produced by :py:meth:`save`. When
        given, the trainer restores model weights, optimiser state, scheduler
        state (if any) and RNG states so training resumes seamlessly from the
        next epoch.

    Notes
    -----
    A checkpoint contains the following keys::

        {
            'epoch': int,                       # last completed epoch
            'model_state_dict': OrderedDict,    # nn.Module parameters
            'optimizer_state_dict': dict,       # optimiser buffers
            'python_rng_state': tuple,
            'numpy_rng_state': tuple,
            'torch_rng_state': Tensor,
            'scheduler_state_dict': dict,       # present only if scheduler
        }

    Restoring all RNG states ensures that any subsequent call that relies on
    ``random``, ``numpy.random`` or ``torch.rand`` produces identical results
    to the original run.
    """    
    def __init__(
            self, model :
            Model, optimizer : torch.optim.Optimizer, criterion : torch.nn.Module,
            device : torch.device,
            seed : int = None,
            scheduler : torch.optim.lr_scheduler.LRScheduler = None,
            saving_uri : str | None = None,
            saving_freq : int | None = None,
            resume_from_previous_state : str | None = None,
    ):
        self.model = model
        self.epoch = 0
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.saving_uri = saving_uri
        self.saving_freq = saving_freq
        self.model.model.to(self.device)

        if seed:
            self.set_seed(seed)

        if saving_uri is None and saving_freq is not None:
            logger.critical(f"Trainer called with saving_freq as {saving_freq} and saving_uri as {saving_uri}.")
            raise ValueError("If saving_freq is set, saving_uri must be set too.")

        if saving_uri is not None and saving_freq is None:
            logger.critical(f"Trainer called with saving_freq as {saving_freq} and saving_uri as {saving_uri}.")
            raise ValueError("If saving_uri is set, saving_freq must be set too.")

        if resume_from_previous_state:
            logger.info(f"Resuming from {resume_from_previous_state}.")
            self.resume(resume_from_previous_state)

        warnings.warn("Use hatorch.train.Trainer instead of hatorch.utils.Trainer", DeprecationWarning)

    @staticmethod
    def set_seed(seed):
        """Make results fully deterministic on the current machine.

        Affects Python, NumPy and PyTorch (CPU & CUDA) RNGs and configures
        cuDNN to deterministic mode.
        """        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def eval(self, valid_loader) -> ((str, str), float):
        """Run *evaluation only* pass over ``valid_loader``.

        Returns
        -------
        (top1, top5, val_loss)
            * ``top1`` : str – Top‑1 accuracy formatted to two decimals.
            * ``top5`` : str – Top‑5 accuracy formatted to two decimals.
            * ``val_loss`` : float – Mean validation loss.
        """        
        logger.info("Entering evaluation mode...")
        correct = 0
        top5_correct = 0
        total = 0
        val_loss = 0
        self.model.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(valid_loader, desc="Validation")):
                inputs, labs = images.to(self.device), labels.to(self.device)
                outputs = self.model.model(inputs)

                # Top-1
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == labs).sum().item()

                # Top-5
                # outputs.topk(k=5) returns a tuple (values, indices)
                _, top5_pred = outputs.topk(k=5, dim=1, largest=True, sorted=True)
                # Check if the correct label is among the top 5 predictions for each sample
                top5_correct += (top5_pred.eq(labs.view(-1, 1).expand_as(top5_pred)).sum(dim=1) > 0).sum().item()

                total += labels.size(0)
                val_loss += self.criterion(outputs, labs).item() * labels.size(0)

        acc = 100 * correct / total
        top5_acc = 100 * top5_correct / total
        val_loss /= len(valid_loader.dataset)

        print(f"Accuracy of the network on test images: {acc:.2f} %")
        print(f"Top-5 Accuracy of the network on test images: {top5_acc:.2f} %")
        # Return a tuple with top1 and top5 accuracies as strings, and the loss
        return f"{acc:.2f}", f"{top5_acc:.2f}", val_loss

    def save(self, path):
        """Serialise model, optimiser, scheduler and RNG states to disk.

        The file will be written to ``f"{path}_epoch_{self.epoch+1}.state"``.
        """        
        state_dict = {
            'epoch': self.epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'python_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
        }
        if self.scheduler:
            state_dict['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(state_dict, path + "_epoch_" + str(self.epoch + 1) + ".state")
        logger.info(f"Saved model to {path}_epoch_{self.epoch + 1}.state")

    def resume(self, path):
        """Restore training state created via :py:meth:`save`."""        
        checkpoint = torch.load(path, weights_only=False)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        random.setstate(checkpoint['python_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        self.epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {self.epoch}")

    def train(self, train_loader, valid_loader, epochs, update_freq=200, valid_freq=0, do_file_logging=False):
        """Main training loop.

        Parameters
        ----------
        train_loader, valid_loader : torch.utils.data.DataLoader
            Data loaders for training and validation datasets.
        epochs : int
            Additional epochs to train **from current ``self.epoch``**.
        update_freq : int, default 200
            Update tqdm bar & logger every *N* iterations.
        valid_freq : int, default 0
            Validate every *N* epochs (0 disables periodic validation).
        do_file_logging : bool, default False
            If *True* writes epoch‑wise metrics to ``testing.csv``.
        """        
        logger.info(f"Entering training mode... for {epochs} epochs")
        loss_int = None
        self.model.model.train(True)
        epochs = self.epoch + epochs
        log_file_name = "testing.csv"

        if do_file_logging:
            logger.info(f"Creating file {log_file_name}")
            with open(log_file_name, "w", newline='') as file:
                logger.info(f"File {log_file_name} already exists. Overwriting it.")
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Training_loss", "Validation_loss", "Accuracy"])

        for epoch in range(self.epoch, epochs):
            logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
            self.epoch = epoch
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}. Loss: {loss_int}")
            for i, (images, labels) in enumerate(progress_bar):
                inputs, labs = images.to(self.device), labels.to(self.device)
                outputs = self.model.model(inputs)
                loss = self.criterion(outputs, labs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update tqdm every 'update_freq' iterations
                if i % update_freq == 0:
                    loss_int = loss.item()
                    progress_bar.set_description(f"Epoch {epoch + 1}/{epochs}. Loss: {loss_int:.4f}")
                    logger.debug("Loss updated")

            if self.scheduler:
                self.scheduler.step()

            if self.saving_freq is not None and (epoch + 1) % self.saving_freq == 0:
                self.save(self.saving_uri)

            if valid_freq != 0 and (epoch + 1) % valid_freq == 0:
                accuracy, top_5, val_loss = self.eval(valid_loader)
                if do_file_logging:
                    with open(log_file_name, "a") as file:
                        writer = csv.writer(file)
                        writer.writerow([epoch + 1, loss_int, val_loss, accuracy, top_5])
                self.model.model.train(True)
                logger.info("Reentering training mode...")

        self.eval(valid_loader)
