"""
Class for performing the training loop

"""

import time

import torch
from torch.utils.data import DataLoader
import wandb

import data.Constants as Constants
from model.UAVModel import UAVModel
from train.Hyperparams import Hyperparams
from train.TrainStopper import init_stopper


class Trainer:
    def __init__(self, hparams: Hyperparams):
        self.hparams = hparams
        self.optimizer_class = self.hparams.optimizer_class
        self.loss_fn_class = self.hparams.loss_class
        self.stopper_class = self.hparams.stopper_class
        self.scheduler_class = self.hparams.scheduler_class
        self.logger = self.hparams.logger

        try:
            self.save_preds = self.hparams.save_preds
        except AttributeError:
            self.save_preds = False

    @property
    def __name__(self):
        return self.__class__.__name__

    def train_model(self, model: UAVModel, train_loader: DataLoader, val_loader=None) -> UAVModel:
        cname = self.__name__
        device = self.hparams.device

        ################################################################################################################
        # Training class object initializations
        ################################################################################################################
        model = model.to(device)
        self.logger.log({"name": cname, "msg": f"Training model using device {device.type}"})

        # Initialize the optimizer object
        optimizer = self.optimizer_class(model.parameters(), **self.hparams.optimizer_hparams)
        lr = self.hparams.optimizer_hparams['lr']
        self.logger.log({"name": cname,
                        "msg": f"Using {optimizer.__class__.__name__} optimizer with "
                               f"LR={lr}"})

        # Initialize learning rate scheduler (if one provided)
        scheduler = None
        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer, **self.hparams.scheduler_hparams)
            self.logger.log({"name": cname,
                             "msg": f"Using {scheduler.__class__.__name__} learning rate scheduler with "
                                    f"params {','.join([f'{k}={v}' for k,v in self.hparams.scheduler_hparams.items()])}"
                             })

        # Initialize the loss function
        loss_fn = self.loss_fn_class(**self.hparams.loss_hparams)
        self.logger.log({"name": cname, "msg": f"Using {loss_fn.__class__.__name__} loss function."})

        # Initialize the training stopper
        stopper = init_stopper(self.hparams)

        # Perform benchmark (pre-training NN performance) on training set
        init_loss = self.eval_model(model, loss_fn, train_loader)
        self.logger.log({'name': cname, 'msg': f'Pre-training train-set loss {init_loss:.3f}', 'data': {
                'epoch': 0,
                f'train_{loss_fn.__name__}': init_loss
        }})

        # Perform benchmark (pre-training NN performance) on validation set
        if val_loader is not None:
            init_val_loss = self.eval_model(model, loss_fn, val_loader)
            self.logger.log({'name': cname, 'msg': f'Pre-training val-set loss {init_val_loss:.3f}', 'data': {
                'epoch': 0,
                f'train_{loss_fn.__name__}': init_val_loss
            }})

        ################################################################################################################
        # Begin main training loop
        ################################################################################################################
        start_time = time.time()
        epoch = 1
        while not stopper.stop():
            epoch_loss_data = {"epoch": epoch}

            # Set model into training mode
            model.train()
            self.logger.log({"name": cname, "msg": f"Training epoch {epoch} beginning"})

            # Go through training set batches
            epoch_start_time = time.time()
            cum_epoch_loss = 0
            for inp, tgt in train_loader:

                # Pass data batch through the model
                model_out = model.forward(inp)

                # Calculate the loss
                loss = loss_fn(model_out, tgt)
                cum_epoch_loss += loss.item()

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_end_time = time.time()
            epoch_time_s = epoch_end_time - epoch_start_time

            # Average the training loss over batches
            avg_epoch_loss = cum_epoch_loss / len(train_loader)
            epoch_loss_data[f"train_{loss_fn.__name__}"] = avg_epoch_loss

            epoch += 1

            # Update the stopper state, if no val set
            if val_loader is None:
                stopper.step(epoch, avg_epoch_loss)

            # Update scheduler state
            if scheduler is not None:
                scheduler.step()

                llr = scheduler.get_last_lr()[0]
                if lr != llr:
                    self.logger.log({"name": cname, "msg": f"Learning rate updated from {lr} to {llr}"})
                    lr = llr

            # At end of epoch, evaluate the performance on validation set
            if val_loader is not None:
                avg_val_loss = self.eval_model(model, loss_fn, val_loader)
                epoch_loss_data[f"val_{loss_fn.__name__}"] = avg_val_loss
                self.logger.log({"name": cname,
                                 "msg": f"Validation loss: {avg_val_loss:.2f}"})

                stopper.step(epoch, avg_val_loss)

            # Log the loss
            self.logger.log({"name": cname,
                             "msg": f"Training epoch {epoch - 1} complete ({epoch_time_s:.3f}s). "
                                    f"Train Loss: {avg_epoch_loss:.2f}",
                             "data": epoch_loss_data,
                             "metrics": epoch_loss_data
                             })

            # See if we should save current version of model
            if stopper.save_model():
                self.logger.log({"name": cname, "msg": "Saving model checkpoint."})
                model.save(mname=f"checkpoint_")

        ################################################################################################################
        # End main training loop
        ################################################################################################################

        # Post training procedures

        end_time = time.time()

        # Perform benchmark (post-training NN performance) on training set
        avg_loss = self.eval_model(model, loss_fn, train_loader)
        self.logger.log({'name': cname, 'msg': f'Final train-set loss {avg_loss:.3f}'})

        # Post training benchmark on validation set
        if val_loader is not None:
            avg_val_loss = self.eval_model(model, loss_fn, val_loader)
            self.logger.log({'name': cname, 'msg': f'Final validation-set loss {avg_val_loss:.3f}'})

        time_elapsed_s = end_time - start_time

        self.logger.log({"name": self.__name__,
                         "msg": f"Training complete. Model trained for {epoch} epochs. "
                                f"Training time: {time_elapsed_s/60:.2f} min"})

        return model
    
    @staticmethod
    def eval_model(model: UAVModel, loss_fn, data_loader: DataLoader):
        with torch.no_grad():
            model.eval()
            avg_loss = 0
            for inp, tgt in data_loader:
                m_out = model.forward(inp)
                loss = loss_fn(m_out, tgt)
                avg_loss += loss.item()
            avg_loss = avg_loss / len(data_loader)

        return avg_loss
    
    
class GCNTrainer(Trainer):
    def __init__(self, hparams: Hyperparams):
        super(GCNTrainer, self).__init__(hparams)
        
    def train_generative_model(self, model, train_loader=None, dataset=None):
        from torchdrug import core, tasks
        from train.Task import UAVGenerationGCPN

        cname = self.__name__
        device = self.hparams.device

        ################################################################################################################
        # Training class object initializations
        ################################################################################################################
        model = model.to(device)
        self.logger.log({"name": cname, "msg": f"Training model using device {device.type}"})

        # Initialize the optimizer object
        optimizer = self.optimizer_class(model.parameters(), **self.hparams.optimizer_hparams)
        lr = self.hparams.optimizer_hparams['lr']
        self.logger.log({"name": cname,
                         "msg": f"Using {optimizer.__class__.__name__} optimizer with "
                                f"LR={lr}"})
        
        # https://torchdrug.ai/docs/api/tasks.html#gcpngeneration
        task = UAVGenerationGCPN(model, [i for i in range(0, (len(Constants.COMPONENT_IDS)))], criterion="nll",
                                    max_node=len(Constants.COMPONENT_IDS), max_edge_unroll=len(Constants.COMPONENT_IDS))
        
        engine_kwargs = {}
        
        # Decide if logging via wandb
        if self.hparams.use_wandb:
            engine_kwargs["logger"] = "wandb"
            
        if 'cuda' in str(self.hparams.device):
            engine_kwargs["gpus"] = (0, )
        
        solver = core.Engine(task, dataset, None, None, optimizer, batch_size=self.hparams.batch_size,
                             log_interval=10, **engine_kwargs)
        
        solver.train(num_epoch=self.hparams.n_epochs)
        
        model = task.model
        
        # for x in dataset.data:
        #    px = model.forward(x, x.node_feature.float())
        # results = task.generate(num_sample=32, max_resample=5)

        # results.to_uav_string()
        # print(results)
        
    
    @staticmethod
    def eval_model():
        pass
