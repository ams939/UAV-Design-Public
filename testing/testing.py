"""
Assortment of random tests

"""
import torch
import pandas as pd

from train.Hyperparams import Hyperparams, DummyHyperparams
from train.TrainStopper import PlateauStopper
from train.Logging import DummyLogger
from train.Loss import SCELoss, CustomMSELoss
from data.DataLoader import UAVSequenceLoader
from data.DataPreprocessor import UAVDesignPreprocessor
from data.DataPostprocessor import UAVDataPostprocessor
from data.DataLoader import UAVDataLoader
from data.UAVDataset import UAVRegressionDataset
from data.Constants import VOCAB_SIZE
from utils.utils import idx_to_onehot
from model.RNN import CharRNN

hparams = Hyperparams("./testing/test_crnn_hparams.json")


def test_sequence_dataloader():
    dataset_hparams = hparams.dataset_hparams
    dataset = hparams.dataset_class(**dataset_hparams)
    batch_size = hparams.dataloader_hparams["batch_size"]
    seq_len = hparams.dataset_hparams["seq_len"]

    dl = UAVSequenceLoader(dataset, batch_size, shuffle=False)

    for idx, (inp_seq, tgt_seq) in enumerate(dl):

        # Last batch will be incomplete
        if idx == len(dl) - 1:
            break

        # Check the shapes of the return values
        assert (inp_seq.shape == torch.Size(
            [seq_len, batch_size, VOCAB_SIZE])), f"Input sequence unexpected size {inp_seq.shape}"
        assert (tgt_seq.shape == torch.Size([seq_len, batch_size])), f"Target sequence unexpected size {inp_seq.shape}"

        # Check the onehot encoding
        assert (torch.allclose(torch.sum(inp_seq, dim=2), torch.ones((seq_len, batch_size), dtype=torch.float64))), \
            f"Last dimension (one-hot) does not sum to 1"

        # Check input-target correspondence
        inp_idx_tensor = torch.nonzero(inp_seq[1:, 0, :] == 1.0, as_tuple=True)[1]

        tgt_idx_tensor = tgt_seq[:-1, 0]

        assert (torch.allclose(inp_idx_tensor, tgt_idx_tensor)), "Input-target mismatch"

    print("Success.")


def test_charnn():
    model = CharRNN(hparams)

    dataset_hparams = hparams.dataset_hparams
    dataset = hparams.dataset_class(**dataset_hparams)

    batch_size = hparams.dataloader_hparams["batch_size"]
    seq_len = hparams.dataset_hparams["seq_len"]

    dummy_input = torch.zeros((seq_len, batch_size, VOCAB_SIZE))

    m_out = model.forward(dummy_input)

    assert m_out.shape == torch.Size((seq_len, batch_size, VOCAB_SIZE)), f"Net out size {m_out.shape}"

    print("Success")


def test_sce_loss():
    loss_fn = SCELoss()

    batch_size = hparams.dataloader_hparams["batch_size"]
    seq_len = hparams.dataset_hparams["seq_len"]

    # Create fake outputs
    dummy_out_idx = torch.floor(torch.zeros((seq_len, batch_size)).uniform_(0, VOCAB_SIZE)).type(torch.long)
    dummy_out = idx_to_onehot(dummy_out_idx, max_idx=VOCAB_SIZE).type(torch.float64)

    dummy_tgt = torch.floor(torch.zeros((seq_len, batch_size)).uniform_(0, VOCAB_SIZE)).type(torch.long)

    loss = loss_fn(dummy_out, dummy_tgt)

    # TODO write a function that iteratively calculates the cross entropy loss

    # assert loss == expected_loss, f"Loss expected: {expected_loss}, Actual: {loss.item()}"


def test_design_preprocessor():
    # Quick sanity check to make sure everything is working as expected
    default_uav = "*aMM0+++*bNM2+++*cMN1+++*dLM2+++*eML1+++^ab^ac^ad^ae,5,3"

    pp = UAVDesignPreprocessor(one_hot=True)

    # Parse the string
    _idx_tensor = pp.parse_design(default_uav)

    # Convert to one hot
    oh_tensor = idx_to_onehot(_idx_tensor, VOCAB_SIZE)

    _uav_str = postprocess([oh_tensor])[0]

    assert (_uav_str == default_uav), "Strings not equal"

    print("Success")


def test_plateau_stopper():
    stopper = PlateauStopper(DummyLogger(), patience=10)
    stopper.prev_loss = 0.1

    i = 0
    for i in range(10):
        if stopper.stop():
            break

        stopper.step(i, 0.01)

    assert i == 6, f"i={i}"
    print("Success")


def test_scale_rescale():
    """
    Sanity check that the preprocess and postprocess operations are inverses of one another


    """

    from sklearn.preprocessing import RobustScaler

    hparams = DummyHyperparams({
        "dataset_hparams": {
            "datafile": "data/datafiles/preprocessed/simresults_preprocessed_validunique.csv",
            "target_cols": ["range", "cost", "velocity", "result"],
            "scale": True,
            "scaler_class": RobustScaler,
            "one_hot": True
        },
        "dataloader_hparams": {
            "batch_size": 1,
            "shuffle": False
        }
    })

    control_df = pd.read_csv(hparams.dataset_hparams["datafile"])

    dataset = UAVRegressionDataset(hparams)
    dataloader = UAVDataLoader(dataset, **hparams.dataloader_hparams)

    postprocessor = UAVDataPostprocessor(hparams)
    reconstructed_df = None

    for (input_idxs, input_lens), tgt in dataloader:
        metrics = torch.transpose(tgt[:-1, :], dim0=1, dim1=0)
        outcomes = tgt[-1, :]

        post_data = postprocessor.postprocess(input_idxs, metrics, outcomes)

        if reconstructed_df is None:
            reconstructed_df = post_data
        else:
            reconstructed_df = pd.concat([reconstructed_df, post_data], ignore_index=True)

    reconstructed_df.to_csv("./testing/test.csv", index=False)


def test_mse():
    from sklearn.preprocessing import RobustScaler

    hparams = DummyHyperparams({
        "dataset_hparams": {
            "datafile": "data/datafiles/preprocessed/simresults_preprocessed_validunique.csv",
            "target_cols": ["range", "velocity", "result"],
            "scale": True,
            "scaler_class": RobustScaler,
            "one_hot": True
        },
        "dataloader_hparams": {
            "batch_size": 8,
            "shuffle": False
        }
    })

    dataset = UAVRegressionDataset(hparams)
    dataloader = UAVDataLoader(dataset, **hparams.dataloader_hparams)

    loss_fn = CustomMSELoss()

    batch_loss = 0
    for _, tgt in dataloader:
        tgt = tgt[:-1, :]
        pred = torch.transpose(tgt, dim0=1, dim1=0)
        outcomes = tgt[-1, :]

        noise = torch.normal(0, 1, pred.shape)

        # pred = pred + noise

        loss = loss_fn(noise, tgt)
        batch_loss += loss.item()

    total_loss = batch_loss / len(dataloader)

    print(f"Total dataset loss equals {total_loss}")


if __name__ == "__main__":
    # test_sequence_dataloader()
    # test_design_preprocessor()
    # test_charnn()
    # test_sce_loss()
    # test_plateau_stopper()
    # test_scale_rescale()
    test_mse()
