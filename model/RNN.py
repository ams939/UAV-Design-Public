"""
This file contains RNN-based realizations of the UAVModel class

"""
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

from data.Constants import VOCAB_SIZE, SOS_VALUE, EOS_VALUE, PAD_VALUE, SIM_METRICS, SIM_OUTCOME
from train.Hyperparams import Hyperparams
from model.UAVModel import UAVModel
from model.NN import FCNet
from utils.utils import idx_to_onehot, make_mask


class CharRNN(UAVModel):
    """
    Implementation of the Char-RNN model as described by Andrej Karpathy:
    https://github.com/karpathy/char-rnn
    http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    https://gist.github.com/karpathy/d4dee566867f8291f086

    """
    def __init__(self, hparams: Hyperparams):
        super(CharRNN, self).__init__(hparams)

        self.rnn = None
        self.out_layer = None

        self.rnn_hparams = self.hparams.model_hparams
        self.num_layers = self.rnn_hparams["num_layers"]
        self.num_hidden = self.rnn_hparams["hidden_size"]
        self.num_features = VOCAB_SIZE

        self.build()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Tensor of shape (seq_len, batch_size, num_features)

        Returns:
            out: Tensor of RNN predicted probabilities for each vocab token (num_features probabilities) at each "time
            step" (letter position in sequence) for a batch of inputs
                Tensor of shape (seq_len, batch_size, num_features)
        """
        x = x.to(self.device)

        # Get outputs from RNN for each step in the sequence
        hidden_out_seq, _ = self.rnn.forward(x)

        # Get token logits for each step in the sequence
        logit_out_seq = self.out_layer.forward(hidden_out_seq)

        # Note absence of softmax activation, this is done by the loss function

        return logit_out_seq

    def sample(self, n_samples=10, seq_len=128) -> torch.Tensor:
        """
        Generates sample sequences from the RNN

        Returns tensor of shape (sample_seq_len, batch_size)

        """
        sample_batch_size = n_samples
        sample_seq_len = seq_len

        # The first input token for beginning generation, i.e sequence of length 1
        in_token_idx = torch.full((1, sample_batch_size), fill_value=SOS_VALUE)
        in_token = idx_to_onehot(in_token_idx, max_idx=VOCAB_SIZE).type(torch.float32)

        # Initialize the memory state and hidden state to zeros
        c_n = torch.zeros((self.num_layers, sample_batch_size, self.num_hidden))
        h_n = torch.zeros((self.num_layers, sample_batch_size, self.num_hidden))

        out_seq = torch.zeros(sample_seq_len, sample_batch_size)

        out_seq[0, :] = in_token_idx

        # Generate outputs from the RNN, one token at a time
        for i in range(1, sample_seq_len):

            # Pass through the net
            out, (c_n, h_n) = self.rnn.forward(in_token, (c_n, h_n))
            out = self.out_layer.forward(out)

            # Sample the next token index from the categorical distribution with probs from the NN
            token_dist = Categorical(logits=out)
            in_token_idx = token_dist.sample()
            in_token = idx_to_onehot(in_token_idx, max_idx=VOCAB_SIZE).type(torch.float32)

            # Store the sampled token index into the output sequence
            out_seq[i, :] = in_token_idx

        return out_seq

    def sample_design(self, batch_size=1, max_seq_len=10000) -> torch.Tensor:
        """
        Generates tokens from the model until EOS token is generated.

        """

        batch_size = 1

        # The first input token for beginning generation, i.e sequence of length 1
        in_token_idx = torch.full((1, batch_size), fill_value=SOS_VALUE)
        in_token = idx_to_onehot(in_token_idx, max_idx=VOCAB_SIZE).type(torch.float32)

        # Initialize the memory state and hidden state to zeros
        c_n = torch.zeros((self.num_layers, batch_size, self.num_hidden))
        h_n = torch.zeros((self.num_layers, batch_size, self.num_hidden))

        out_seq = torch.zeros(1, batch_size)

        out_seq[0, :] = in_token_idx

        seq_lens = np.zeros(batch_size)
        eos_reached = False
        seq_idx = 1
        while not eos_reached:
            # Pass through the net
            out, (c_n, h_n) = self.rnn.forward(in_token, (c_n, h_n))
            out = self.out_layer.forward(out)

            # Sample the next token index from the categorical distribution with probs from the NN
            token_dist = Categorical(logits=out)
            in_token_idx = token_dist.sample()
            in_token = idx_to_onehot(in_token_idx, max_idx=VOCAB_SIZE).type(torch.float32)

            # Store the sampled token index into the output sequence
            out_seq[seq_idx, :] = in_token_idx

            # Store first encountered EOS location for each sequence
            in_token_idx_np = in_token_idx.cpu().numpy()
            eos_indices = np.intersect1d(np.argwhere(in_token_idx_np == EOS_VALUE), np.argwhere(seq_lens == 0))
            if len(eos_indices) > 0:
                seq_lens[eos_indices] = seq_idx + 1

            seq_idx += 1
            eos_reached = np.all(seq_lens)

    def build(self):
        """
        Uses hyperparameters given to build  the model
        """

        # The LSTM layer
        self.rnn = nn.LSTM(input_size=self.num_features, **self.rnn_hparams)

        # NN layer for outputting VOCAB_SIZE probabilities
        self.out_layer = nn.Linear(in_features=self.num_hidden, out_features=self.num_features)

        self.load()


class SimRNN(UAVModel):
    """
    Simulator surrogate RNN model, can predict all the simulator outputs (range, cost, velocity, result), i.e
    performs a regression task of uav_string --> (range, cost, velocity) and classification task uav_string --> result

    """
    def __init__(self, hparams: Hyperparams):
        super(SimRNN, self).__init__(hparams)

        self.rnn = None
        self.fc_hidden = None
        self.clf_layer = None
        self.reg_layer = None

        self.rnn_hparams = self.hparams.model_hparams["rnn_hparams"]
        self.num_hidden = self.rnn_hparams["hidden_size"]
        self.num_features = VOCAB_SIZE
        self.num_targets = len(set(self.hparams.dataset_hparams["target_cols"]).intersection(set(SIM_METRICS)))
        self.dropout = self.hparams["dropout"]
        self.fc_hidden_sizes = self.hparams.model_hparams["fc_hparams"]["fc_hidden_sizes"]
        self.num_classes = self.hparams["num_outcomes"]  # Number of sim result classes
        self.agg_seq_states = self.hparams.model_hparams["agg_seq_states"]

        self.build()

    def build(self):
        # Construct the RNN layer
        self.rnn = nn.LSTM(input_size=self.num_features, batch_first=False, **self.rnn_hparams)

        # Construct fully connected hidden layers taking the RNN outputs
        self.fc_hidden = nn.Sequential()
        fc_hidden_in = self.fc_hidden_sizes[:-1]
        fc_hidden_in.insert(0, self.num_hidden)
        for idx, size in enumerate(self.fc_hidden_sizes):
            self.fc_hidden.append(nn.Linear(in_features=fc_hidden_in[idx], out_features=size))
            self.fc_hidden.append(nn.ReLU())
            self.fc_hidden.append(nn.Dropout(self.dropout))

        fc_hidden_out = self.fc_hidden_sizes[-1]

        # Output layers
        self.reg_layer = nn.Linear(in_features=fc_hidden_out, out_features=self.num_targets)  # predicts metric values
        self.clf_layer = nn.Linear(in_features=fc_hidden_out, out_features=self.num_classes)  # predicts outcome class

        # Load the model from file, if it exists
        self.load()

    def forward(self, x_in: Tuple[torch.Tensor, torch.Tensor]):
        """
            Args:
                x_in: Tuple of tensors:
                    x : Padded input tensor of shape (max_seq_len, batch_size, num_features)
                    x_lens: Tensor of the true sequence lengths of sequences in x

            Returns:
                reg_out: Output of the regression layer
                clf_out: Output of the classifier layer ()
        """
        x, x_lens = x_in

        x = x.to(self.device)

        # Pack the padded sequences
        x_packed = pack_padded_sequence(x, x_lens)

        # Pass through LSTM layers, take the last hidden layer output
        out, (h_n, _) = self.rnn(x_packed)

        # Aggregate the RNN hidden states, or just take the last one
        if self.agg_seq_states:
            # Unpack the packed sequences
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(out)

            # Mask out padded value outputs
            mask = make_mask(output_lengths, outputs.shape)
            masked_output = outputs * mask.to(self.device)

            # Sum together every hidden state for each step
            h_n = torch.sum(masked_output, dim=0)
        else:
            # Take last hidden state of last layer
            h_n = h_n[-1, :, :]

        # Pass through the hidden FC layer
        h_n = self.fc_hidden(h_n)

        # Pass through output layers
        clf_out = self.clf_layer(h_n)
        reg_out = self.reg_layer(h_n)

        return reg_out, clf_out

    def predict(self, x: Tuple[torch.Tensor, torch.Tensor]):
        with torch.no_grad():
            # self.device = torch.device('cpu')
            reg_out, clf_out = self.forward(x)

        return reg_out.detach().cpu(), torch.round(torch.sigmoid(clf_out.detach())).type(torch.long).cpu()

    def sample(self, n_samples):
        """ Not supported by this model """
        pass


class MetricSimRNN(UAVModel):
    """
    Second iteration of the simulator surrogate NN. This time using a more modular structure, with an "encoder" RNN
    and multiple hidden layers following it. Supports using the Embedding layer, bidirectional LSTM's

    """
    def __init__(self, hparams: Hyperparams):
        super(MetricSimRNN, self).__init__(hparams)

        self.rnn_encoder = None
        self.fc_hidden = None
        self.out_layer = None
        self.num_targets = len(self.hparams.dataset_hparams["target_cols"])  # Number of sim metric predictions
        self.fc_hidden_sizes = self.hparams.model_hparams["fc_hparams"]["fc_hidden_sizes"]
        self.dropout = self.hparams.dropout

        self.build()

    def build(self):
        self.rnn_encoder = self.hparams.model_hparams["rnn_encoder_class"](self.hparams)

        # Construct fully connected hidden layers taking the RNN outputs
        self.fc_hidden = nn.Sequential()
        fc_hidden_in = self.fc_hidden_sizes[:-1]
        fc_hidden_in.insert(0, self.rnn_encoder.num_output)
        for idx, size in enumerate(self.fc_hidden_sizes):
            self.fc_hidden.append(nn.Linear(in_features=fc_hidden_in[idx], out_features=size))
            self.fc_hidden.append(nn.ReLU())
            self.fc_hidden.append(nn.Dropout(self.dropout))

        fc_hidden_out = self.fc_hidden_sizes[-1]
        self.out_layer = nn.Linear(in_features=fc_hidden_out, out_features=self.num_targets)

        self.load()

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        """
            Args:
                x : Padded input tensor of shape (max_seq_len, batch_size, num_features)
                x_lens: Tensor of the true sequence lengths of sequences in x

            Returns:
                reg_out: Output of the regression layer
        """
        h_n = self.rnn_encoder.forward(x, x_lens)

        # Pass through the hidden FC layer(s)
        h_n = self.fc_hidden(h_n)

        # Pass through output layers
        out = self.out_layer(h_n)

        return out

    def predict(self, x: torch.Tensor, x_lens: torch.Tensor):
        with torch.no_grad():
            # self.set_device(torch.device('cpu'))
            out = self.forward(x, x_lens)

        return out.detach().cpu()

    def set_device(self, device):
        self.rnn_encoder.set_device(device)
        self.device = device


class OutcomeSimRNN(UAVModel):
    """ Classifier RNN - Predicts only the simulator outcome """
    def __init__(self, hparams: Hyperparams):
        super(OutcomeSimRNN, self).__init__(hparams)

        self.rnn_encoder = None
        self.fc_hidden = None
        self.out_layer = None

        self.fc_hidden_sizes = self.hparams.model_hparams["fc_hparams"]["fc_hidden_sizes"]
        self.num_classes = self.hparams["num_outcomes"]  # Number of sim metric predictions
        self.dropout = self.hparams.dropout

        self.build()

    def build(self):
        self.rnn_encoder = self.hparams.model_hparams["rnn_encoder_class"](self.hparams)

        # Construct fully connected hidden layers taking the RNN outputs
        self.fc_hidden = nn.Sequential()
        fc_hidden_in = self.fc_hidden_sizes[:-1]
        fc_hidden_in.insert(0, self.rnn_encoder.num_output)
        for idx, size in enumerate(self.fc_hidden_sizes):
            self.fc_hidden.append(nn.Linear(in_features=fc_hidden_in[idx], out_features=size))
            self.fc_hidden.append(nn.ReLU())
            self.fc_hidden.append(nn.Dropout(self.dropout))

        fc_hidden_out = self.fc_hidden_sizes[-1]

        self.out_layer = nn.Linear(in_features=fc_hidden_out, out_features=self.num_classes)

    def forward(self, x_in: Tuple[torch.Tensor, torch.Tensor]):
        """
            Args:
                x_in: Tuple of tensors (x, x_lens)
                    x : Padded input tensor of shape (max_seq_len, batch_size, num_features)
                    x_lens: Tensor of the true sequence lengths of sequences in x

            Returns:
                out: Output of the classifier layer
        """
        x, x_lens = x_in

        h_n = self.rnn_encoder.forward(x, x_lens)

        # Pass through the hidden FC layer(s)
        h_n = self.fc_hidden(h_n)

        # Pass through output layers
        out = self.out_layer(h_n)

        return out

    def predict(self, x: Tuple[torch.Tensor, torch.Tensor]):
        with torch.no_grad():
            # self.set_device(torch.device('cpu'))
            out = self.forward(x)

        return torch.round(torch.sigmoid(out.detach())).type(torch.long).cpu()

    def set_device(self, device):
        self.rnn_encoder.set_device(device)
        self.device = device


class FullSimRNN(UAVModel):
    """
    Simulator surrogate RNN model, can predict all the simulator outputs (range, cost, velocity, result), i.e
    performs a regression task of uav_string --> (range, cost, velocity) and classification task uav_string --> result

    """
    def __init__(self, hparams: Hyperparams):
        super(FullSimRNN, self).__init__(hparams)

        self.rnn_hparams = self.hparams.model_hparams["rnn_encoder_hparams"]
        self.num_hidden = self.rnn_hparams["hidden_size"]
        self.num_features = VOCAB_SIZE
        self.targets = self.hparams.dataset_hparams["target_cols"]
        self.num_target_metrics = len(set(self.targets).intersection(set(SIM_METRICS)))
        self.dropout = self.hparams["dropout"]
        self.fc_hidden_sizes = self.hparams.model_hparams["fc_hparams"]["hidden_sizes"]
        self.fc_reg_sizes = self.hparams.model_hparams["reg_hparams"]["hidden_sizes"]
        self.fc_clf_sizes = self.hparams.model_hparams["clf_hparams"]["hidden_sizes"]
        self.num_classes = self.hparams["num_outcomes"]
        self.is_classifier = SIM_OUTCOME in self.targets
        self.is_regressor = self.num_target_metrics > 0

        # Initialize hidden layers
        self.rnn = None
        self.fc_hidden = None
        self.class_layer = None
        self.regression_layer = None

        self.build()

    def build(self):
        # Construct the RNN layer - handled by RNN-Encoder
        self.rnn = self.hparams.model_hparams["rnn_encoder_class"](self.hparams)

        # Construct fully connected hidden layers taking the RNN outputs
        self.fc_hidden = FCNet(in_features=self.rnn.num_output, hidden_sizes=self.fc_hidden_sizes, dropout=self.dropout)
        fc_hidden_out = self.fc_hidden_sizes[-1]

        # Construct heads for targets

        # Metrics head
        if self.is_regressor:
            self.regression_layer = FCNet(in_features=fc_hidden_out, out_features=self.num_target_metrics,
                                          hidden_sizes=self.fc_reg_sizes, act_out=False)

        # Outcome head
        if self.is_classifier:
            self.class_layer = FCNet(in_features=fc_hidden_out, out_features=self.num_classes,
                                     hidden_sizes=self.fc_clf_sizes, act_out=False)

        # Load the model from file, if it exists
        self.load()

    def forward(self, x_in: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
            Args:
                x_in: Tuple of tensors (x, x_lens), or tensor x
                    x : Padded input tensor of shape (max_seq_len, batch_size, num_features)
                    x_lens: Tensor of the true sequence lengths of sequences in x

            Returns:
                reg_out: Output of the regression layer
                clf_out: Output of the classifier layer ()
        """
        # Pass through LSTM layers, take the last hidden layer output
        out = self.rnn(x_in)

        # Pass through the hidden FC layer
        h_n = self.fc_hidden(out)

        # Pass through output heads
        if self.is_classifier:
            clf_out = self.class_layer(h_n)
        else:
            clf_out = None

        if self.is_regressor:
            reg_out = self.regression_layer(h_n)
        else:
            reg_out = None

        return reg_out, clf_out

    def predict(self, x_in: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
        with torch.no_grad():
            # self.set_device(torch.device('cpu'))  # Inference done on CPU
            reg_out, clf_out = self.forward(x_in)

        if reg_out is not None:
            reg_out = reg_out.detach().cpu()

        if clf_out is not None:
            clf_out = torch.round(torch.sigmoid(clf_out.detach())).type(torch.long).cpu()

        return reg_out, clf_out

    def set_device(self, device):
        self.rnn.set_device(device)
        self.device = device


class UAVRNNEncoder(UAVModel):
    def __init__(self, hparams: Hyperparams):
        super(UAVRNNEncoder, self).__init__(hparams)

        self.rnn = None
        self.num_features = VOCAB_SIZE
        self.rnn_hparams = self.hparams.model_hparams["rnn_encoder_hparams"]
        self.num_layers = self.rnn_hparams["num_layers"]
        self.num_hidden = self.rnn_hparams["hidden_size"]
        self.bidirectional = self.rnn_hparams["bidirectional"]
        self.agg_seq_states = self.hparams.model_hparams["agg_seq_states"]

        if self.bidirectional:
            self.D = 2
        else:
            self.D = 1

        try:
            self.one_hot = hparams.dataset_hparams["one_hot"]
        except TypeError:
            self.one_hot = False

        try:
            self.use_embed = hparams.model_hparams["use_embed"]
        except TypeError:
            self.use_embed = False

        assert self.one_hot != self.use_embed, "Please specify either 'one_hot' or 'use_embed' as true, not both."

        if self.use_embed:
            self.embedding_size = hparams.model_hparams["embedding_size"]
            self.rnn_feature_dim = self.embedding_size
            self.embed_layer = None
        else:
            self.rnn_feature_dim = self.num_features

        self.num_output = self.num_hidden * self.D

        self.build()

    def build(self):
        if self.use_embed:
            self.embed_layer = nn.Embedding(num_embeddings=self.num_features, embedding_dim=self.embedding_size,
                                            padding_idx=PAD_VALUE)

        self.rnn = nn.LSTM(input_size=self.rnn_feature_dim, **self.rnn_hparams)

        # TODO: Separate out saving and loading of model elements
        # self.load()

    def forward(self, x_in: Tuple[torch.Tensor, torch.Tensor]):
        """
            Args:
                x_in: Tuple of tensors (x, x_lens)
                    x : Padded input tensor of shape (max_seq_len, batch_size, num_features)
                    x_lens: Tensor of the true sequence lengths of sequences in x

            Returns:
                reg_out: Output of the regression layer
        """
        x, x_lens = x_in

        x = x.to(self.device)

        if self.use_embed:
            x = self.embed_layer(x)

        # Pack the padded sequences
        x_packed = pack_padded_sequence(x, x_lens)

        # Pass through LSTM layers, take the last hidden layer output
        out, (h_n, _) = self.rnn(x_packed)

        # Aggregate the RNN hidden states, or just take the last one
        if self.agg_seq_states:
            # Unpack the packed sequences
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(out)

            # Mask out padded value outputs
            mask = make_mask(output_lengths, outputs.shape)
            masked_output = outputs * mask.to(self.device)

            # Sum together every hidden state for each step
            h_n = torch.sum(masked_output, dim=0)
        else:
            # Take the output of the last rnn layer, h_n: (D*num_layers, batch_size, num_hidden)
            if self.bidirectional:
                # If bidirectional, concatenate the forward and reverse LSTM final hidden states
                # https://stackoverflow.com/questions/50856936/taking-the-last-state-from-bilstm-bigru-in-pytorch
                h_n = torch.concat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
            else:
                h_n = h_n[-1, :, :]

        return h_n


class SimpleRNN(UAVModel):
    """
    SimpleRNN - expects batches of fixed length inputs

    """

    def __init__(self, hparams: Hyperparams):
        super(SimpleRNN, self).__init__(hparams)

        self.rnn_hparams = hparams.model_hparams["rnn_encoder_hparams"]
        self.rnn_feature_dim = hparams.feature_dim
        self.num_layers = self.rnn_hparams["num_layers"]
        self.num_hidden = self.rnn_hparams["hidden_size"]
        self.batch_first = self.rnn_hparams["batch_first"]
        if self.batch_first is None:
            self.batch_first = False  # by convention of this project
        
        self.bidirectional = self.rnn_hparams["bidirectional"]
        self.agg_seq_states = self.hparams.model_hparams["agg_seq_states"]

        if self.bidirectional:
            self.D = 2
        else:
            self.D = 1
        
        if self.batch_first:
            self.ts_dim = 1
        else:
            self.ts_dim = 0

        hparams.logger.log(
            {"name": "SimpleRNN", "msg": f"Sanity check time! Is the time step dimension {self.ts_dim}?"})
        
        self.num_output = self.num_hidden * self.D

        self.rnn = None
        self.build()

    def build(self):
        self.rnn = nn.LSTM(input_size=self.rnn_feature_dim, **self.rnn_hparams)

    def forward(self, x: torch.Tensor):

        x = x.to(self.device)

        # Pass through LSTM
        out, (h_n, _) = self.rnn(x)

        # Aggregate the RNN hidden states, or just take the last one
        if self.agg_seq_states:

            # Sum together every hidden state for each step
            h_n = torch.sum(out, dim=self.ts_dim)
        else:
            # Take the output of the last rnn layer, h_n: (D*num_layers, batch_size, num_hidden)
            if self.bidirectional:
                # If bidirectional, concatenate the forward and reverse LSTM final hidden states
                # https://stackoverflow.com/questions/50856936/taking-the-last-state-from-bilstm-bigru-in-pytorch
                h_n = torch.concat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
            else:
                h_n = h_n[-1, :, :]

        return h_n
    
    
class PolicyRNN(UAVModel):
    def __init__(self, hparams: Hyperparams):
        super(PolicyRNN, self).__init__(hparams)
        
        self.in_fc_sizes = hparams.model_hparams.in_fc_hparams.hidden_sizes
        self.hidden_fc_sizes = hparams.model_hparams.hidden_fc_hparams.hidden_sizes
        self.rnn_hidden = hparams.model_hparams.rnn_encoder_hparams.hidden_size
        
        self.in_rnn = None
        self.in_fc = None
        
        self.hidden = None
        self.out = None
        
        self.build()
        
    def build(self):
        # Layer taking in the sequence of states/actions
        _in_rnn = SimpleRNN(self.hparams)
        _in_fc = FCNet(in_features=_in_rnn.num_output, hidden_sizes=self.in_fc_sizes)
        
        self.in_rnn = torch.nn.Sequential(
            _in_rnn,
            _in_fc
        )
        
        # Layer taking in an action
        self.in_fc = FCNet(in_features=self.hparams.feature_dim, hidden_sizes=self.in_fc_sizes)
        
        # Hidden FC layer, note in_features has factor 2x due to concatenation of in_fc and in_rnn outputs
        self.hidden = FCNet(in_features=2*self.in_fc_sizes[-1], hidden_sizes=self.hidden_fc_sizes,
                            dropout=self.hparams.dropout)
        
        # Out FC layer
        self.out = FCNet(in_features=self.hidden_fc_sizes[-1], hidden_sizes=[self.hparams.out_dim], act_out=False)
        
    def forward(self, state_seq: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        Network forward pass method, takes two arguments:
            state_seq: A batch of sequences of tensors, has shape (batch_size, sequence length, num features)
            act: A batch of action tensors, one action for each sequence, has shape (batch_size, num_features)
            
        Returns a tensor of shape (batch_size, out_dim) where out_dim is specified in hparams
        
        """
        batch_size = state_seq.shape[0]
        
        # Sanity checks on input
        assert state_seq.shape[0] == act.shape[0], f"Batch size mismatch for state_seq and act tensors! " \
                                                   f"({state_seq.shape[0]} != {act.shape[0]})"
        assert state_seq.shape[-1] == self.hparams.feature_dim, f"Sequence tensors of incorrect feature size passed! " \
                                                          f"(Expected {self.hparams.feature_dim}, got {state_seq.shape[-1]})"
        
        # Move tensors to device
        state_seq = state_seq.to(self.device)
        act = act.to(self.device)
        
        # In layer pass
        rnn_in_out = self.in_rnn(state_seq)
        fc_in_out = self.in_fc(act)
        
        # Concatenate input layers' outputs
        in_out = torch.concat([rnn_in_out, fc_in_out], dim=1)
        assert in_out.shape == torch.Size((batch_size, 2*self.in_fc_sizes[-1])), "DEBUG: Check concat op"
        
        # Pass through hidden layer(s)
        hidden_out = self.hidden(in_out)
        assert hidden_out.shape == torch.Size((batch_size, self.hidden_fc_sizes[-1])), "DEBUG: Check hidden"
        
        # Pass through out layer
        out = self.out(hidden_out)
        assert out.shape == torch.Size((batch_size, self.hparams.out_dim)), "DEBUG: Check out"
        
        return out
    
    def predict(self, seq, act):
        with torch.no_grad():
            out = self.forward(seq, act)
        return out

        
if __name__ == "__main__":
    pass
