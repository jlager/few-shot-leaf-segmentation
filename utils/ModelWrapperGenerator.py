import torch, time, sys, pdb
import numpy as np

from utils.TimeRemaining import *
from torch.utils.data import DataLoader

class ModelWrapper():
   
    '''
    Utility function that wraps around a PyTorch model. It allows for easy
    training and saving/loading feed-forward models using a data generator. 
   
    Args:
        model        (callable): Model.
        optimizer    (callable): Optimizer.
        loss         (callable): Loss function that inputs (pred, true).
        regularizer  (callable): Regularization that inputs (model, inputs, 
                                  true, pred).
        augmentation (callable): Augmentation that inputs x and y.
        scheduler    (callable): Learning rate scheduler.
        save_name      (string): Model name for saving model/opt weights.
        save_best_train  (bool): Indicator for saving on best train loss.
        save_best_val    (bool): Indicator for saving on best val loss.
        save_opt         (bool): Indicator for saving optimizer weights.
   
    Inputs:
        x       (tensor/generator): Input data.
        y       (tensor/generator): Target data.
        batch_size           (int): Batch size.
        epochs               (int): Total number of epochs.
        verbose              (int): How much info to print to the screen:
                                        0: No updates
                                        1: Update after every epoch
                                        2: Update after every batch
        validation_data     (list): Input and target validation data.
        shuffle             (bool): Whether to shuffle data each epoch.
        class_weight        (list): NOT IMPLEMENTED
        sample_weight       (list): NOT IMPLEMENTED
        initial_epoch        (int): Initial epoch to start training.
        steps_per_epoch      (int): Train batches before moving to next epoch.
        validation_steps     (int): Val batches before moving to next epoch.
        validation_freq      (int): NOT IMPLEMENTED
        early_stopping       (int): Number of epochs since validation improved.
        best_train_loss    (float): Best loss on training set.
        best_val_loss      (float): Best loss on validation set.
        include_val_aug     (bool): Inclusion of augmentation in val step.
        include_val_reg     (bool): Inclusion of regularizer in val loss.
        lr_dec_epoch         (int): Decrease lr after this many epochs.
        lr_dec_prop        (float): Value <= 1 to multiply learning rate.
        rel_save_thresh    (float): Rel. diff. btwn losses before saving.
        
    Returns:
        train_loss_list (list): Training errors per epoch.
        val_loss_list   (list): Validation errors per epoch.
    '''
   
    def __init__(self, 
                 model,
                 optimizer,
                 loss,
                 regularizer=None,
                 augmentation=None,
                 device=None,
                 scheduler=None,
                 save_name=None,
                 log_name=None,
                 save_best_train=False,
                 save_best_val=True,
                 save_opt=False,
                 save_reg=False):
        
        # assign values
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.regularizer = regularizer
        self.augmentation = augmentation
        self.scheduler = scheduler
        self.save_name = save_name
        self.log_name = log_name
        self.save_best_train = save_best_train
        self.save_best_val = save_best_val
        self.save_opt = save_opt
        self.save_reg = save_reg
        self.train_loss_list = []
        self.val_loss_list = []
        self.train = False
        self.val = False
        self.device = device if device is not None else torch.device('cpu')
        
        # if no save name specified, don't save weights
        if self.save_name is None:
            self.save_best_train = False
            self.save_best_val = False
            self.save_opt = False
            self.save_reg = False
        
    def fit(self,
            train_dataset=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_dataset=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            early_stopping=None,
            best_train_loss=None,
            best_val_loss=None,
            include_val_aug=False,
            include_val_reg=False,
            lr_dec_epoch=None,
            lr_dec_prop=1.0,
            rel_save_thresh=0.0,
            workers=1,
            collate_fn=None,
            synchronize=True):
        
        # initialize book keeping
        train_length = len(train_dataset)
        train_batches_per_epoch = int(train_length/batch_size)
        if validation_dataset is not None:
            val_length = len(validation_dataset) 
            val_batches_per_epoch = int(len(validation_dataset)/batch_size)
        start_time = time.time()
        last_improved_train, last_improved_val = 0, 0
        best_train_loss = 1e12 if best_train_loss is None else best_train_loss
        best_val_loss = 1e12 if best_val_loss is None else best_val_loss
        
        # callback at beginning of training
        if callbacks is not None:
            for c in callbacks:
                if c.on_train_begin:
                    c(self)

        # initialize log file
        if self.log_name is not None:
            with open(self.log_name, 'w') as f:
                f.writelines('Epoch,Train Loss,Val Loss\n')
        
        # loop over epochs
        for epoch in range(initial_epoch, initial_epoch + epochs):

            #
            # training step
            #
            
            self.train = True
            self.val = False
            
            # callback at beginning of epoch
            if callbacks is not None:
                for c in callbacks:
                    if c.on_epoch_begin:
                        c(self)
            
            self.model.train()
            train_losses = 0
            epoch_start_time = time.time()
            
            # compile train data loader
            if collate_fn is None:
                train_loader = DataLoader(train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=shuffle, 
                                        num_workers=workers)
            else:
                train_loader = DataLoader(train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=shuffle, 
                                        num_workers=workers,
                                        collate_fn=collate_fn)
            
            # loop over training batches
            idx = 0
            for x_true, y_true in train_loader:
                
                # callback at beginning of batch
                if callbacks is not None:
                    for c in callbacks:
                        if c.on_batch_begin:
                            c(self)
                
                # stop loop if steps_per_epoch exceeded
                if steps_per_epoch is not None:
                    if idx >= steps_per_epoch:
                        break
                
                batch_start_time = time.time()
                
                # assign to device
                x_true = x_true.to(self.device).contiguous()
                y_true = y_true.to(self.device).contiguous()
                
                # computes loss
                def closure():
                    
                    # zero out gradients
                    self.optimizer.zero_grad()
                    
                    # zero-initialize losses
                    self.train_loss = 0
                    self.train_reg_loss = 0

                    # require gradients
                    if torch.is_tensor(x_true):
                        x_true.requires_grad = True

                    # run the model
                    y_pred = self.model(x_true)

                    
                    # otherwise proceed normally
                    self.train_loss = self.train_loss + self.loss(y_pred, y_true)
                    if self.regularizer is not None:
                        self.train_reg_loss = self.train_reg_loss + self.regularizer(
                            self.model, 
                            x_true,
                            y_true,
                            y_pred)
                    self.train_loss = self.train_loss + self.train_reg_loss
                    self.train_loss.backward() 
                    
                    return self.train_loss
                
                # update model parameters
                if self.scheduler is None:
                    self.optimizer.step(closure=closure)
                else:
                    self.scheduler.step(closure())
                
                # update book keeping for this batch
                self.train_loss = self.train_loss.cpu().detach().numpy()
                train_losses += self.train_loss
                
                # wait for GPU computations to finish
                if self.device != torch.device('cpu') and synchronize:
                    torch.cuda.synchronize()
                
                # print batch statistics
                if verbose == 2:
                    elapsed, remaining, ms_per_iter = TimeRemaining(
                        current_iter=idx+1, 
                        total_iter=train_batches_per_epoch, 
                        start_time=epoch_start_time, 
                        previous_time=batch_start_time, 
                        ops_per_iter=batch_size)
                    sys.stdout.write(('\r\x1b[KEpoch {0} {1}/{2}' + 
                                      ' | {3} ms ' + 
                                      ' | Train loss = {4:1.4e}' + 
                                      ' | Remaining = ').format(
                        str(epoch+1).rjust(len(str(epochs))),
                        str(idx+1).rjust(len(str(train_batches_per_epoch))),
                        train_batches_per_epoch,
                        ms_per_iter,
                        train_losses/(idx+1)) + remaining + '          ')
                    sys.stdout.flush()
                    
                # callback at ending of batch
                if callbacks is not None:
                    for c in callbacks:
                        if c.on_batch_end:
                            c(self)
                            
                idx += 1
                
            # update book keeping for this epoch
            self.train_loss_list.append(train_losses/train_batches_per_epoch)
            
            # if train error improved
            rel_diff = (best_train_loss - self.train_loss_list[-1])
            rel_diff /= best_train_loss
            if rel_diff > rel_save_thresh:
                
                # update best training loss
                best_train_loss = self.train_loss_list[-1]
                
                # optionally save model and optimizer
                if self.save_best_train:
                    self.save(self.save_name+'_best_train')
                    
                # update early stopper
                last_improved_train = epoch
                    
            # print readout
            if verbose == 2:
                sys.stdout.write(('\r\x1b[KEpoch {0}' + 
                                  ' {1}/{2}' + 
                                  ' | {3} ms ' + 
                                  ' | Train loss = {4:1.4e}' + 
                                  ' | Elapsed = {5}').format(
                    str(epoch+1).rjust(len(str(epochs))),
                    train_batches_per_epoch,
                    train_batches_per_epoch,
                    ms_per_iter,
                    np.mean(self.train_loss_list[-1]),
                    elapsed) + '       ')
                sys.stdout.flush()
                
            # optional early stopping
            if early_stopping is not None and self.save_best_train:
                if epoch - last_improved_train >= early_stopping:
                    break
            
            #
            # validation step
            #
            
            if validation_dataset is not None:
                
                self.train = False
                self.val = True
                
                self.model.eval()
                
                # zero-initialize losses
                self.val_loss = 0
                self.val_reg_loss = 0
                
                # compile validation data loader
                if collate_fn is None:
                    val_loader = DataLoader(validation_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=workers)
                else:
                    val_loader = DataLoader(validation_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=workers,
                                            collate_fn=collate_fn)
                
                # prevent cuda memory from blowing up on validation
                with torch.no_grad():
                
                    # loop over validation batches
                    idx = 0
                    for x_true, y_true in val_loader:

                        # callback at beginning of batch
                        if callbacks is not None:
                            for c in callbacks:
                                if c.on_batch_begin:
                                    c(self)

                        # stop loop if validation_steps exceeded
                        if validation_steps is not None:
                            if idx >= validation_steps:
                                break

                        # assign to device
                        x_true = x_true.to(self.device).contiguous()
                        y_true = y_true.to(self.device).contiguous()

                        # require gradients
                        if torch.is_tensor(x_true):
                            x_true.requires_grad = True

                        # run the model
                        y_pred = self.model(x_true)

                        # compute loss 
                        if isinstance(self.loss, list): 
                            for loss_fun in self.loss:
                                self.val_loss += loss_fun(y_pred, y_true)
                        else:
                            self.val_loss += self.loss(y_pred, y_true)

                        # optionally include regularization in val loss
                        if include_val_reg and self.regularizer is not None:
                            self.val_reg_loss += self.regularizer(self.model, 
                                                                  x_true,
                                                                  y_true,
                                                                  y_pred)

                        # wait for GPU computations to finish
                        if self.device != torch.device('cpu') and synchronize:
                            torch.cuda.synchronize()

                        # callback at ending of batch
                        if callbacks is not None:
                            for c in callbacks:
                                if c.on_batch_end:
                                    c(self)

                        idx += 1
                    
                # update book keeping for this epoch
                loss = self.val_loss/(idx + 1) + self.val_reg_loss/(idx + 1)
                self.val_loss_list.append(loss.cpu().detach().numpy())
                
                # if validation error improved
                rel_diff = (best_val_loss - self.val_loss_list[-1])
                rel_diff /= best_val_loss
                if rel_diff > rel_save_thresh:
                    
                    # update best validation loss
                    best_val_loss = self.val_loss_list[-1]
                    
                    # optionally save model and optimizer
                    if self.save_best_val:
                        self.save(self.save_name+'_best_val')
                    
                    # update early stopper
                    last_improved_val = epoch
                    
                    improved = ' *'
                    
                else:
                    
                    improved = ''
                    
            # log progress
            if self.log_name is not None:
                with open(self.log_name, 'a') as f:
                    f.writelines('{0},{1},{2}\n'.format(
                        epoch+1, 
                        self.train_loss_list[-1], 
                        self.val_loss_list[-1]))
            
            # update user
            if verbose == 1:
                
                # times
                elapsed, remaining, ms = TimeRemaining(
                    current_iter=epoch+1,
                    total_iter=initial_epoch+epochs,
                    start_time=start_time,
                    previous_time=epoch_start_time,
                    ops_per_iter=batch_size)
                
                # prints
                p = '\rEpoch {0}'.format(epoch+1)
                p += ' | Train loss = {0:1.4e}'.format(self.train_loss_list[-1])
                if validation_dataset is not None:
                    p += ' | Val loss = {0:1.4e}'.format(self.val_loss_list[-1])
                p += ' | Remaining = ' + remaining + '           '
                sys.stdout.write(p)
                
            # final print readout for verbose 2
            if verbose == 2:
                sys.stdout.write(('\r\x1b[KEpoch {0} {1}/{2}' + 
                                  ' | Train loss = {3:1.4e}' + 
                                  ' | Val loss = {4:1.4e}').format(
                    str(epoch+1).rjust(len(str(epochs))),
                    train_batches_per_epoch,
                    train_batches_per_epoch,
                    self.train_loss_list[-1],
                    self.val_loss_list[-1]))
                print(improved + '                 ')
                
            # optional early stopping
            if early_stopping is not None and self.save_best_val:
                if epoch - last_improved_val >= early_stopping:
                    break
                    
            # optional learning rate annealing
            if lr_dec_epoch is not None:
                if np.mod(epoch, lr_dec_epoch) == 0 and epoch != 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= lr_dec_prop
                        
            # callback at ending of epoch
            if callbacks is not None:
                for c in callbacks:
                    if c.on_epoch_end:
                        c(self)
                
        # final print readout for verbose 1
        if verbose == 1:
            
            # times
            elapsed, remaining, ms = TimeRemaining(
                current_iter=epoch+1,
                total_iter=initial_epoch+epochs,
                start_time=start_time,
                previous_time=epoch_start_time,
                ops_per_iter=batch_size)
            
            # prints
            if self.save_best_val:
                idx = np.argmin(self.val_loss_list)
            elif self.save_best_train:
                idx = np.argmin(self.train_loss_list)
            else:
                idx = -1
            p = '\rEpoch {0}'.format(epoch+1)
            p += ' | Train loss = {0:1.4e}'.format(self.train_loss_list[idx])
            if validation_dataset is not None:
                p += ' | Val loss = {0:1.4e}'.format(self.val_loss_list[idx])
            p += ' | Elapsed = ' + elapsed + '           '
            sys.stdout.write(p)
            print()
                
        # final print readout for verbose 2
        if verbose == 2:
            
            # times
            elapsed, remaining, ms = TimeRemaining(
                current_iter=epoch+1,
                total_iter=initial_epoch+epochs,
                start_time=start_time,
                previous_time=epoch_start_time,
                ops_per_iter=batch_size)
            
            # prints
            if self.save_best_val:
                idx = np.argmin(self.val_loss_list)
            elif self.save_best_train:
                idx = np.argmin(self.train_loss_list)
            else:
                idx = -1
            p = '\rTrain loss = {0:1.4e}'.format(self.train_loss_list[idx])
            if validation_dataset is not None:
                p += ' | Val loss = {0:1.4e}'.format(self.val_loss_list[idx])
            p += ' | Elapsed = ' + elapsed + '           '
            sys.stdout.write(p)
            print()
            
        # callback at ending of training
        if callbacks is not None:
            for c in callbacks:
                if c.on_train_end:
                    c(self)
    
    def predict(self, inputs):
        
        '''
        Runs the model on a given set of inputs.
        '''
        
        # run model in eval mode (for batchnorm, dropout, etc.)
        self.model.eval()
        
        return self.model(inputs)
    
    def save(self, save_name):
        
        '''
        Saves model weights and optionally optimizer weights.
        '''
        
        # save model weights
        torch.save(self.model.state_dict(), save_name+'_model.save')
        
        # save optimizer weights
        if self.save_opt and self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), save_name+'_opt.save')
        
        # save regularizer weights
        if self.save_reg and self.regularizer is not None:
            torch.save(self.regularizer.state_dict(), save_name+'_reg.save')
    
    def load(self, 
             model_weights, 
             opt_weights=None, 
             reg_weights=None, 
             device=None):
        
        '''
        Loads model weights and optionally optimizer weights.
        '''
        
        # load model weights
        weights = torch.load(model_weights, map_location=device)
        self.model.load_state_dict(weights)
        self.model.eval()
        
        # load optimizer weights
        if opt_weights is not None:
            params = torch.load(opt_weights, map_location=device)
            self.optimizer.load_state_dict(params)
        
        # load regularizer weights
        if reg_weights is not None:
            params = torch.load(reg_weights, map_location=device)
            self.regularizer.load_state_dict(params)
    
    def load_best_train(self, device=None):
        
        '''
        Loads model weights that yielded best training error.
        '''
        
        # load model weights
        name = self.save_name+'_best_train_model.save'
        weights = torch.load(name, map_location=device)
        self.model.load_state_dict(weights)
        self.model.eval()
        
        # load optimizer weights
        if self.save_opt and self.optimizer is not None:
            name = self.save_name+'_best_train_opt.save'
            params = torch.load(name, map_location=device)
            self.optimizer.load_state_dict(params)
        
        # load optimizer weights
        if self.save_reg and self.regularizer is not None:
            name = self.save_name+'_best_train_reg.save'
            params = torch.load(name, map_location=device)
            self.regularizer.load_state_dict(params)
    
    def load_best_val(self, device=None):
        
        '''
        Loads model weights that yielded best validation error.
        '''
        
        # load model weights
        name = self.save_name+'_best_val_model.save'
        weights = torch.load(name, map_location=device)
        self.model.load_state_dict(weights)
        self.model.eval()
        
        # load optimizer weights
        if self.save_opt and self.optimizer is not None:
            name = self.save_name+'_best_val_opt.save'
            params = torch.load(name, map_location=device)
            self.optimizer.load_state_dict(params)
        
        # load optimizer weights
        if self.save_reg and self.regularizer is not None:
            name = self.save_name+'_best_val_reg.save'
            params = torch.load(name, map_location=device)
            self.regularizer.load_state_dict(params)
