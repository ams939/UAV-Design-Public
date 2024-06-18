"""
Custom loss functions

"""
import torch
from torch.nn.modules.loss import CrossEntropyLoss, NLLLoss, MSELoss, BCEWithLogitsLoss, L1Loss

from data.Constants import PAD_VALUE


class SCELoss(CrossEntropyLoss):
    """
    Sequential cross-entropy loss (SCE loss), calculates the cross-entropy loss for every element in a sequence

    """
    def __init__(self, ignore_index=PAD_VALUE):
        super(SCELoss, self).__init__(ignore_index=ignore_index, reduction='none')

    @property
    def __name__(self):
        return self.__class__.__name__

    def forward(self, pred_seq_token_logits: torch.Tensor, tgt_seq_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
        Expects pred_seq_token_logits in the form (sequence_length, batch_size, n_tokens)
        and tgt_seq_tokens in the form (sequence_length, batch_size)

        """

        # Reshaping for the torch loss fn, which expects inputs as (batch_size, n_tokens, sequence_length)
        logits_re = torch.transpose(pred_seq_token_logits, dim0=2, dim1=0)
        logits_re = torch.transpose(logits_re, dim0=0, dim1=1)

        # Reshape the target sequences into batch first format (batch_size, sequence_length)
        tgt_seqs = torch.transpose(tgt_seq_tokens, dim0=1, dim1=0)

        loss = super(SCELoss, self).forward(logits_re, tgt_seqs)

        assert loss.shape == tgt_seqs.shape, "Unexpected loss shape"

        # Sum losses together over the sequence dimension
        loss = torch.sum(loss, dim=-1)

        assert loss.shape == torch.Size([tgt_seqs.shape[0]])

        # Average losses over batch dimension
        loss = torch.mean(loss, dim=0)

        assert loss.shape == torch.Size([]), "Unexpected loss shape"

        return loss


class SNLLLoss(NLLLoss):
    """
    Sequential cross-entropy loss (SCE loss), calculates the cross-entropy loss for every element in a sequence

    """
    def __init__(self, ignore_index=PAD_VALUE):
        super(SNLLLoss, self).__init__(ignore_index=ignore_index, reduction='none')

    @property
    def __name__(self):
        return self.__class__.__name__

    def forward(self, pred_seq_token_probs: torch.Tensor, tgt_seq_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
        Expects pred_seq_token_logits in the form (sequence_length, batch_size, n_tokens)
        and tgt_seq_tokens in the form (sequence_length, batch_size)

        """

        # Reshaping for the torch loss fn, which expects inputs as (batch_size, n_tokens, sequence_length)
        probs_re = torch.transpose(pred_seq_token_probs, dim0=2, dim1=0)
        probs_re = torch.transpose(probs_re, dim0=0, dim1=1)

        # Reshape the target sequences into batch first format (batch_size, sequence_length)
        tgt_seqs = torch.transpose(tgt_seq_tokens, dim0=1, dim1=0)

        loss = -super(SNLLLoss, self).forward(torch.log(probs_re), tgt_seqs)

        assert loss.shape == tgt_seqs.shape, "Unexpected loss shape"

        # Sum losses together over the sequence dimension
        loss = torch.sum(loss, dim=-1)

        assert loss.shape == torch.Size([tgt_seqs.shape[0]])

        # Average losses over batch dimension
        loss = torch.mean(loss, dim=0)

        assert loss.shape == torch.Size([]), "Unexpected loss shape"

        return loss


class MSEBCELoss(MSELoss):
    def __init__(self):
        super(MSEBCELoss, self).__init__(reduction='mean')
        self.reg_loss_fn = super(MSEBCELoss, self).forward
        self.cls_loss_fn = BCEWithLogitsLoss()

    @property
    def __name__(self):
        return self.__class__.__name__

    def forward(self, pred, tgt):

        # Unpack the prediction and target tuples
        reg_pred, cls_pred = pred

        if reg_pred is None:
            reg_tgt = None

            device = cls_pred.device
            cls_tgt = tgt.to(device)

        elif cls_pred is None:
            cls_tgt = None

            device = reg_pred.device
            reg_tgt = tgt.to(device)

        else:
            device = reg_pred.device
            reg_tgt, cls_tgt = tgt[:-1, :].to(device), tgt[-1, :].to(device)

        # Calculate the losses
        if reg_pred is not None:
            reg_tgt = torch.transpose(reg_tgt, dim0=1, dim1=0)
            reg_loss = self.reg_loss_fn(reg_pred, reg_tgt)
        else:
            reg_loss = None

        if cls_pred is not None:
            cls_pred = cls_pred.squeeze()
            cls_tgt = cls_tgt.squeeze()
            cls_loss = self.cls_loss_fn(cls_pred, cls_tgt)
        else:
            cls_loss = None

        # Return the loss
        if reg_loss is not None and cls_loss is not None:
            return reg_loss + cls_loss

        elif reg_loss is not None:
            return reg_loss
        elif cls_loss is not None:
            return cls_loss
        else:
            print("Warning: No loss calculated")
            return None


class MAEBCELoss(L1Loss):
    def __init__(self):
        super(MAEBCELoss, self).__init__(reduction='mean')
        self.reg_loss_fn = super(MAEBCELoss, self).forward
        self.cls_loss_fn = BCEWithLogitsLoss()

    @property
    def __name__(self):
        return self.__class__.__name__

    def forward(self, pred, tgt):

        # Unpack the prediction and target tuples
        reg_pred, cls_pred = pred

        if reg_pred is None:
            reg_tgt = None

            device = cls_pred.device
            cls_tgt = tgt.to(device)

        elif cls_pred is None:
            cls_tgt = None

            device = reg_pred.device
            reg_tgt = tgt.to(device)

        else:
            device = reg_pred.device
            reg_tgt, cls_tgt = tgt[:-1, :].to(device), tgt[-1, :].to(device)

        # Calculate the losses
        if reg_pred is not None:
            reg_tgt = torch.transpose(reg_tgt, dim0=1, dim1=0)
            reg_loss = self.reg_loss_fn(reg_pred, reg_tgt)
        else:
            reg_loss = None

        if cls_pred is not None:
            cls_pred = cls_pred.squeeze()
            cls_tgt = cls_tgt.squeeze()
            cls_loss = self.cls_loss_fn(cls_pred, cls_tgt)
        else:
            cls_loss = None

        # Return the loss
        if reg_loss is not None and cls_loss is not None:
            return reg_loss + cls_loss

        elif reg_loss is not None:
            return reg_loss
        elif cls_loss is not None:
            return cls_loss
        else:
            print("Warning: No loss calculated")
            return None


class CustomMSELoss(MSELoss):
    def __init__(self):
        super(CustomMSELoss, self).__init__(reduction='mean')

    @property
    def __name__(self):
        return self.__class__.__name__

    def forward(self, pred, tgt):
        # Move targets to device
        device = pred.device
        tgt = tgt.to(device)

        # Calculate the losses
        reg_pred = pred
        reg_tgt = torch.transpose(tgt, dim0=1, dim1=0)

        reg_loss = super(CustomMSELoss, self).forward(reg_pred, reg_tgt)

        return reg_loss


class WeightedMAEBCELoss(BCEWithLogitsLoss):
    def __init__(self, pos_weight, reg_weight=1.0, cls_weight=1.0):
        if not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.Tensor([pos_weight])
        super(WeightedMAEBCELoss, self).__init__(pos_weight=pos_weight)
        self.reg_loss_fn = WeightedL1Loss()
        self.cls_loss_fn = super(WeightedMAEBCELoss, self).forward
        self.pos_weight = pos_weight
        
        self.reg_weight = torch.Tensor([reg_weight])
        self.cls_weight = torch.Tensor([cls_weight])
    
    @property
    def __name__(self):
        return self.__class__.__name__
    
    def forward(self, pred, tgt):
        
        # Unpack the prediction and target tuples
        reg_pred, cls_pred = pred
        
        assert reg_pred is not None, "This loss function requires regression predictions"
        assert cls_pred is not None, "This loss function requires classification predictions"
        
        device = reg_pred.device
        reg_tgt, cls_tgt = tgt[:-1, :].to(device), tgt[-1, :].to(device)
        weights = torch.zeros_like(cls_pred)
        weights[cls_tgt != 1.0] = 1.0
        weights[cls_tgt == 1.0] = self.pos_weight.item()
        weights = weights.to(device)
        
        assert torch.all(weights).item() is True, "Error in loss weights"
        
        # Calculate the losses
        reg_tgt = torch.transpose(reg_tgt, dim0=1, dim1=0)
        assert reg_tgt.shape == reg_pred.shape, f"Shape mismatch in regression predictions {reg_pred.shape} " \
                                                f"and targets {reg_tgt.shape}"
        reg_loss = self.reg_loss_fn(reg_pred, reg_tgt, weights)

        cls_pred = cls_pred.squeeze()
        cls_tgt = cls_tgt.squeeze()
        assert cls_pred.shape == cls_tgt.shape, f"Shape mismatch in classification predictions {cls_pred.shape} " \
                                                f"and targets {cls_tgt.shape}"
        cls_loss = self.cls_loss_fn(cls_pred, cls_tgt)

        return reg_loss + cls_loss
        

class WeightedL1Loss(L1Loss):
    def __init__(self):
        super(WeightedL1Loss, self).__init__(reduction='none')

    @property
    def __name__(self):
        return self.__class__.__name__
    
    def forward(self, pred, tgt, weights=None):
        # Move targets to device
        device = pred.device
        tgt = tgt.to(device)
    
        # Calculate the losses
        reg_pred = pred
        reg_tgt = tgt
    
        reg_loss = super(WeightedL1Loss, self).forward(reg_pred, reg_tgt)
        
        # Sum over the metric dimension (sum of L1 distances of each axis)
        reg_loss = torch.sum(reg_loss, dim=1)
        
        # A "weighted average" of loss terms s.t loss terms are repeated "weights" times
        reg_loss = torch.sum(weights * reg_loss) / torch.sum(weights)
    
        return reg_loss


class CustomBCELoss(BCEWithLogitsLoss):
    """
    'This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable
    than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.'
    - torch docs, https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    """
    def __init__(self):
        super(CustomBCELoss, self).__init__()

    @property
    def __name__(self):
        return self.__class__.__name__

    def forward(self, pred_logits, tgt_class_labels):
        device = pred_logits.device
        tgt = tgt_class_labels.to(device)

        # Calculate the losses
        pred = pred_logits
        tgt = torch.transpose(tgt, dim0=1, dim1=0)

        loss = super(CustomBCELoss, self).forward(pred, tgt)

        return loss
