import torch

from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
from torch import nn

import numpy as np

from sklearn.model_selection import train_test_split

class ChromatogramModel:

    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train_step(self, chroms, peakgroups, labels):

        self.optimizer.zero_grad()

        yhat = self.model(chroms, peakgroups)

        loss = self.criterion(
            yhat.reshape(-1),
            labels.double()
        )

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train_val_split(self, dataset, val_split=0.10):

        train_idx, val_idx = train_test_split(
            list(range(len(dataset))),
            test_size=val_split
        )

        training_set = Subset(dataset, train_idx)
        validation_set = Subset(dataset, val_idx)

        return training_set, validation_set

    def validation_step(self, chroms, peakgroups, labels):

        yhat = self.model(chroms, peakgroups)

        val_loss = self.criterion(yhat.reshape(-1), labels)

        return val_loss.item()

    def load(self, saved_model_path):

        self.model.load_state_dict(
            torch.load(saved_model_path)
        )

    def eval_test_accuracy(self, testing_dataset):

        testing_loader = DataLoader(
            testing_dataset,
            batch_size=32,
            num_workers=5,
            drop_last=True
        )

        accuracies = []

        for i, (peakgroups, chroms, labels) in enumerate(testing_loader):

            peakgroups = peakgroups.to(device)
            chroms = chroms.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = self.model.predict(chroms.double(), peakgroups)

                accuracy = ((predictions.detach() == labels.reshape((-1, 1)).detach()).sum() / labels.shape[0]).item()

                accuracies.append(accuracy)

            accuracies.append(accuracy)

        return np.mean(accuracies)

    def train(self, training_data, val_split=0.10, batch_size=32, n_epochs=50):

        training_split, validation_split= self.train_val_split(training_data, val_split)

        training_loader = DataLoader(
            training_split,
            batch_size=batch_size,
            shuffle=True,
            num_workers=5
        )

        validation_loader = DataLoader(
            validation_split,
            batch_size=batch_size,
            shuffle=True,
            num_workers=5
        )

        try:

            for epoch in range(1, n_epochs + 1):

                losses = []
                validation_losses = []

                accuracies = []

                for i, (peakgroups, chroms, labels) in enumerate(training_loader):

                    peakgroups = peakgroups.to(self.device)
                    chroms = chroms.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    yhat = self.model(chroms, peakgroups)

                    loss = self.criterion(yhat.reshape(-1), labels.double())

                    loss.backward()

                    self.optimizer.step()

                    losses.append(loss.item())

                    percentage = (i / len(training_loader)) * 100.0

                    print("Epoch percentage: ", percentage, end='\r')

                training_loss = np.mean(losses)

                losses.append(training_loss)

                with torch.no_grad():

                    val_losses = []

                    for i, (peakgroups, chroms, labels) in enumerate(validation_loader):

                        peakgroups = peakgroups.to(self.device)
                        chroms = chroms.to(self.device)
                        labels = labels.to(self.device)

                        val_loss = self.validation_step(chroms, peakgroups, labels)

                        val_losses.append(val_loss)

                        predictions = self.model.predict(
                            chroms.double(),
                            peakgroups
                        )

                        accuracy = ((predictions.detach() == labels.reshape((-1, 1)).detach()).sum() / predictions.shape[0]).item()

                        accuracies.append(accuracy)

                    batch_val_loss = np.mean(val_losses)

                    validation_losses.append(val_loss)

                epoch_loss = np.mean(losses)
                epoch_val_accuracy = np.mean(accuracies)
                epoch_val_loss = np.mean(validation_losses)

                self.train_losses.append(epoch_loss)
                self.val_losses.append(epoch_val_loss)

                print(f'epoch {epoch}, loss {epoch_loss}, val loss {epoch_val_loss}, val accuracy {epoch_val_accuracy}')

        except Exception as e:

            torch.save(self.model.state_dict(), './chrom.pth')

            raise e

        torch.save(self.model.state_dict(), './chrom.pth')


class ChromatogramProbabilityModel(nn.Module):

    def __init__(self, n_features, sequence_length):

        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=n_features,
            out_channels=7,
            kernel_size=3,
            stride=1,
            padding='same'
        )

        self.n_features = n_features
        self.sequence_length = sequence_length

        self.layer_dim = 2
        self.hidden_dim = 20

        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.layer_dim,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        self.linear = nn.Linear((2 * self.hidden_dim) * sequence_length + 3, 42)
        self.linear_2 = nn.Linear(42, 42)
        self.linear_3 = nn.Linear(42, 42)
        self.linear_4 = nn.Linear(42, 1)


    def forward(self, chromatogram, peakgroup):

        batch_size, seq_len, _ = chromatogram.size()

        out = self.conv1d(chromatogram.double())

        out = out.permute(0, 2, 1)

        out, _ = self.rnn(out.double())

        out = out.contiguous().view(batch_size, -1)

        out = torch.cat((out, peakgroup), 1)

        out = self.linear(out)

        out = F.relu(out)

        out = self.linear_2(out)

        out = F.relu(out)

        out = self.linear_3(out)

        out = F.relu(out)

        out = self.linear_4(out)

        return out

    def predict(self, data, peakgroup):

        out = self.forward(data, peakgroup)

        probabilities = torch.sigmoid(out)

        return (probabilities > 0.5).double()


class ChromatogramDataset(Dataset):

    def __init__(self, peakgroups, chromatograms, peakgroup_graph):

        self.peakgroups = peakgroups

        print("Scaling Retention Times...")

        peakgroup_graph.scale_peakgroup_retention_times()

        self.chromatograms = chromatograms

        self.peakgroup_graph = peakgroup_graph
        self.min_chromatogram_length = chromatograms.min_chromatogram_length()

        self.interfering_chroms = []

    def __len__(self):

        return len(self.peakgroups)

    def __getitem__(self, idx):

        peakgroup = self.peakgroups[idx]

        peptide_id = ''

        for edge_node in peakgroup.iter_edges(self.peakgroup_graph):

            if edge_node.color == 'peptide':
                peptide_id = f"{edge_node.sequence}_{edge_node.charge}"

        peakgroup_boundaries = np.array(
            [
                peakgroup.scaled_rt_start,
                peakgroup.scaled_rt_apex,
                peakgroup.scaled_rt_end
            ],
            dtype=np.float64
        )

        label = peakgroup.target

        transition_chromatograms = list()

        rt_steps = list()
        chrom_ids = list()

        for native_id, chromatogram_records in self.chromatograms[peptide_id].items():

            precursor_difference = abs(chromatogram_records.precursor_mz - peakgroup.mz)
            #             print(precursor_difference)

            rt_min = chromatogram_records.rts.min()
            rt_max = chromatogram_records.rts.max()

            if precursor_difference == 0.0:

                if chromatogram_records.type == 'fragment':

                    if not rt_steps:
                        scaled_chrom_rt = chromatogram_records.scaled_rts(
                            min_val=self.peakgroup_graph.min_rt_val,
                            max_val=self.peakgroup_graph.max_rt_val
                        )

                        rt_steps = [scaled_chrom_rt[chrom_idx] for chrom_idx in range(self.min_chromatogram_length)]

                        transition_chromatograms.append(np.asarray(rt_steps))

                    transition_chromatogram = list()

                    normalized_intensities = chromatogram_records.normalized_intensities(add_min_max=(0.0, 10.0))

                    if np.isfinite(normalized_intensities).all():

                        for chrom_idx in range(self.min_chromatogram_length):

                            norm_intensity = normalized_intensities[chrom_idx]

                            if np.isnan(norm_intensity):
                                norm_intensity = 0.0

                            transition_chromatogram.append(normalized_intensities[chrom_idx])
                    else:

                        for chrom_idx in range(self.min_chromatogram_length):
                            transition_chromatogram.append(0)

                    transition_chromatograms.append(np.asarray(transition_chromatogram))

                    chrom_ids.append(native_id)

        chromatograms_transformed = list()

        if len(transition_chromatograms) > 7:
            peptide_node = self.peakgroup_graph[peakgroup.get_edges()[0]]

            self.interfering_chroms.append([chrom_ids, len(transition_chromatograms), peptide_id, idx, peakgroup.mz,
                                            peptide_node.modified_sequence])

            transition_chromatograms = transition_chromatograms[:7]

        for row_transform in zip(*transition_chromatograms):
            chromatogram_row = np.asarray(row_transform, dtype='double')

            chromatograms_transformed.append(chromatogram_row)

        chromatograms_transformed = torch.tensor(chromatograms_transformed, dtype=torch.double)

        return torch.tensor(peakgroup_boundaries, dtype=torch.double), chromatograms_transformed.double().T, torch.tensor(label).double()


if __name__ == '__main__':

    from gscore.utils.connection import Connection
    import pickle

    import pandas as pd

    from gscore.parsers import osw, queries

    from gscore.denoiser import BaggedDenoiser
    from sklearn.model_selection import train_test_split

    from gscore.chromatograms import Chromatograms

    from torch.utils.data import Dataset, DataLoader

    from sklearn.model_selection import train_test_split

    from torch.utils.data import Subset

    import torch
    from torch import nn


    osw_path = '/home/aaron/projects/ghost/data/spike_in/openswath/AAS_P2009_172.osw'
    chrom_path = '/home/aaron/projects/ghost/data/spike_in/chromatograms/AAS_P2009_172.sqMass'

    chromatograms = Chromatograms().from_sqmass_file(chrom_path)

    peakgroup_graph, none_peak_groups = osw.fetch_peakgroup_graph(
        osw_path=osw_path,
        query=queries.SelectPeakGroups.FETCH_TRAIN_CHROMATOGRAM_SCORING_DATA
    )

    highest_ranked = peakgroup_graph.filter_ranked_peakgroups(
        rank=1,
        score_column='probability',
        value=0.5,
        user_operator='>',
        target=1
    )

    decoy_ranked = peakgroup_graph.get_ranked_peakgroups(rank=1, target=0)

    combined_data = highest_ranked + decoy_ranked

    print(len(combined_data))

    chromatogram_dataset = ChromatogramDataset(combined_data, chromatograms, peakgroup_graph)

    train_idx, val_idx = train_test_split(
        list(range(len(chromatogram_dataset))),
        test_size=0.2
    )

    training_dataset = Subset(chromatogram_dataset, train_idx)
    testing_dataset = Subset(chromatogram_dataset, val_idx)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ChromatogramProbabilityModel(
        7,
        172
    ).double().to(device)

    chromatogram_model = ChromatogramModel(
        model=model,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.005),
        device=device
    )

    chromatogram_model.train(
        training_data=training_dataset,
        val_split=0.10,
        batch_size=32,
        n_epochs=2
    )

