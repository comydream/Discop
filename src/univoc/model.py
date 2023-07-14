from math import log2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import librosa

import importlib_resources
from omegaconf import OmegaConf

from config import Settings, audio_default_settings
from utils import set_seed, SingleExampleOutput


def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell


class Vocoder(nn.Module):

    def __init__(
        self,
        n_mels,
        conditioning_size,
        embedding_dim,
        rnn_size,
        fc_size,
        bits,
        hop_length,
        sr,
    ):
        super().__init__()
        self.rnn_size = rnn_size
        self.bits = bits
        self.hop_length = hop_length
        self.sr = sr

        self.rnn1 = nn.GRU(
            n_mels,
            conditioning_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.embedding = nn.Embedding(2**self.bits, embedding_dim)
        self.rnn2 = nn.GRU(embedding_dim + 2 * conditioning_size, rnn_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_size, fc_size)
        self.fc2 = nn.Linear(fc_size, 2**self.bits)

    @classmethod
    def from_pretrained(cls, url, map_location=None, cfg_path=None):
        r"""
        Loads the Torch serialized object at the given URL (uses torch.hub.load_state_dict_from_url).

        Parameters:
            url (string): URL of the weights to download
            cfg_path (Path): path to config file. Defaults to univoc/config/config.yaml
        """
        cfg_ref = (importlib_resources.files("univoc.config").joinpath("config.yaml") if cfg_path is None else cfg_path)
        with cfg_ref.open() as file:
            cfg = OmegaConf.load(file)
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location=map_location)
        model = cls(**cfg.model)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    def forward(self, x, mels):
        mels, _ = self.rnn1(mels)

        mels = F.interpolate(mels.transpose(1, 2), scale_factor=self.hop_length)
        mels = mels.transpose(1, 2)

        x = self.embedding(x)

        x, _ = self.rnn2(torch.cat((x, mels), dim=2))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @torch.no_grad()
    def generate(self, mel):
        r"""
        Generates an audio waverform from a log-Mel spectrogram.

        Parameters:
            mel (Tensor): of shape (1, seq_len, n_mels) containing the log-Mel spectrogram.

        Returns:
            Tuple[np.array, int]: The resulting waveform of shape (seq_len * hop_length) and sample rate in Hz.
        """
        wav = []
        cell = get_gru_cell(self.rnn2)

        mel, _ = self.rnn1(mel)

        mel = F.interpolate(mel.transpose(1, 2), scale_factor=self.hop_length)
        mel = mel.transpose(1, 2)

        h = torch.zeros(mel.size(0), self.rnn_size, device=mel.device)
        x = torch.zeros(mel.size(0), device=mel.device, dtype=torch.long)
        x = x.fill_(2**(self.bits - 1))

        for m in tqdm(torch.unbind(mel, dim=1), leave=False):
            x = self.embedding(x)
            h = cell(torch.cat((x, m), dim=1), h)

            x = F.relu(self.fc1(h))
            logits = self.fc2(x)

            posterior = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(posterior)

            x = dist.sample()
            wav.append(x.item())

        wav = np.asarray(wav, dtype=np.int)
        wav = librosa.mu_expand(wav - 2**(self.bits - 1), mu=2**self.bits - 1)
        return wav, self.sr

    @torch.no_grad()
    def encode_speech(self, mel, message: str, settings: Settings = audio_default_settings, tqdm_desc: str = 'Enc '):
        from stega_cy import encode_step
        algo, temp, top_p, _, seed = settings()
        set_seed(seed)

        wav = []
        cell = get_gru_cell(self.rnn2)

        mel, _ = self.rnn1(mel)

        mel = F.interpolate(mel.transpose(1, 2), scale_factor=self.hop_length)
        mel = mel.transpose(1, 2)

        # mel = mel[:, :48000, :]  # generate only first 3 seconds

        h = torch.zeros(mel.size(0), self.rnn_size, device=mel.device)
        x = torch.zeros(mel.size(0), device=mel.device, dtype=torch.long)
        x = x.fill_(2**(self.bits - 1))

        total_capacity = 0
        total_entropy = 0
        total_log_probs = 0  # to calculate the perplexity
        total_kld = 0
        total_minimum_entropy = 0
        max_kld = 0

        message_encoded = ''
        start = time.time()
        for m in tqdm(torch.unbind(mel, dim=1), desc=tqdm_desc, ncols=70):
            x = self.embedding(x)
            h = cell(torch.cat((x, m), dim=1), h)

            x = F.relu(self.fc1(h))
            logits = self.fc2(x)
            logits, indices = logits[0, :].sort(descending=True)
            logits = logits.double()

            logits = logits / temp

            probs = F.softmax(logits, dim=-1)

            if top_p < 1.0:
                cum_probs = probs.cumsum(0)

                k = (cum_probs > top_p).nonzero()[0].item() + 1
                probs = probs[:k]
                indices = indices[:k]
                probs = 1 / cum_probs[k - 1] * probs  # Normalization
            probs = probs.tolist()
            indices = indices.tolist()

            sampled_index, capacity_t, entropy_t, kld_step, min_entropy_t = encode_step(settings, indices, probs, message)()

            indices_idx = indices.index(sampled_index)
            total_entropy += entropy_t
            total_log_probs += log2(probs[indices_idx])
            total_kld += kld_step
            total_minimum_entropy += -log2(probs[0])
            if kld_step > max_kld:
                max_kld = kld_step

            x = torch.tensor([sampled_index], device=mel.device)

            if capacity_t > 0:
                total_capacity += capacity_t
                message_encoded += message[:capacity_t]
                message = message[capacity_t:]  # remove the encoded part of `message`

            wav.append(x.item())
        end = time.time()
        perplexity = 2**(-1 / len(wav) * total_log_probs)
        wav = np.asarray(wav, dtype=np.int)
        # print(wav)
        wav = librosa.mu_expand(wav - 2**(self.bits - 1), mu=2**self.bits - 1)
        # return wav, self.sr
        return SingleExampleOutput(None, wav, total_capacity, total_entropy, total_kld / len(wav), max_kld, perplexity,
                                   end - start, settings, total_minimum_entropy), self.sr

    @torch.no_grad()
    def decode_speech(self, mel, stego_speech: np.ndarray, settings: Settings = audio_default_settings, tqdm_desc: str = 'Dec '):
        from stega_cy import decode_step
        algo, temp, top_p, _, seed = settings()
        set_seed(seed)

        cell = get_gru_cell(self.rnn2)

        mel, _ = self.rnn1(mel)

        mel = F.interpolate(mel.transpose(1, 2), scale_factor=self.hop_length)
        mel = mel.transpose(1, 2)

        h = torch.zeros(mel.size(0), self.rnn_size, device=mel.device)
        x = torch.zeros(mel.size(0), device=mel.device, dtype=torch.long)
        x = x.fill_(2**(self.bits - 1))

        # mu_compress
        stego_speech = librosa.mu_compress(stego_speech, mu=2**self.bits - 1) + 2**(self.bits - 1)

        t = 0
        message_decoded = ''
        start = time.time()
        for m in tqdm(torch.unbind(mel, dim=1), desc=tqdm_desc, ncols=70):
            x = self.embedding(x)
            h = cell(torch.cat((x, m), dim=1), h)

            x = F.relu(self.fc1(h))
            logits = self.fc2(x)
            logits, indices = logits[0, :].sort(descending=True)
            logits = logits.double()

            logits = logits / temp

            probs = F.softmax(logits, dim=-1)

            if top_p < 1.0:
                cum_probs = probs.cumsum(0)

                k = (cum_probs > top_p).nonzero()[0].item() + 1
                probs = probs[:k]
                indices = indices[:k]
                probs = 1 / cum_probs[k - 1] * probs  # Normalization
            probs = probs.tolist()
            indices = indices.tolist()

            message_decoded_t = decode_step(settings, indices, probs, stego_speech[t])

            if message_decoded_t == 'x':
                raise ValueError('Failed to decode!')
            message_decoded += message_decoded_t

            x = torch.tensor([stego_speech[t]], device=mel.device)
            t += 1
        end = time.time()
        print('Decode time: {:.2f}s'.format(end - start))
        return message_decoded

    @torch.no_grad()
    def random_sample_speech(self, mel, message, settings: Settings = audio_default_settings, tqdm_desc: str = 'Enc '):
        from random_sample_cy import encode_step
        algo, temp, top_p, _, seed = settings()
        set_seed(seed)

        wav = []
        cell = get_gru_cell(self.rnn2)

        mel, _ = self.rnn1(mel)

        mel = F.interpolate(mel.transpose(1, 2), scale_factor=self.hop_length)
        mel = mel.transpose(1, 2)

        # mel = mel[:, :48000, :]  # generate only first 3 seconds

        h = torch.zeros(mel.size(0), self.rnn_size, device=mel.device)
        x = torch.zeros(mel.size(0), device=mel.device, dtype=torch.long)
        x = x.fill_(2**(self.bits - 1))

        total_capacity = 0
        total_entropy = 0
        total_log_probs = 0  # to calculate the perplexity
        total_kld = 0
        total_minimum_entropy = 0
        max_kld = 0

        message_encoded = ''
        start = time.time()
        for m in tqdm(torch.unbind(mel, dim=1), desc=tqdm_desc, ncols=70):
            x = self.embedding(x)
            h = cell(torch.cat((x, m), dim=1), h)

            x = F.relu(self.fc1(h))
            logits = self.fc2(x)
            logits, indices = logits[0, :].sort(descending=True)
            logits = logits.double()

            logits = logits / temp

            probs = F.softmax(logits, dim=-1)

            if top_p < 1.0:
                cum_probs = probs.cumsum(0)

                k = (cum_probs > top_p).nonzero()[0].item() + 1
                probs = probs[:k]
                indices = indices[:k]
                probs = 1 / cum_probs[k - 1] * probs  # Normalization
            probs = probs.tolist()
            indices = indices.tolist()

            sampled_index, capacity_t, entropy_t, kld_step, min_entropy_t = encode_step(indices, probs, message)()

            indices_idx = indices.index(sampled_index)
            total_entropy += entropy_t
            total_log_probs += log2(probs[indices_idx])
            total_kld += kld_step
            total_minimum_entropy += -log2(probs[0])
            if kld_step > max_kld:
                max_kld = kld_step

            x = torch.tensor([sampled_index], device=mel.device)

            # if capacity_t > 0:
            #     total_capacity += capacity_t
            #     message_encoded += message[:capacity_t]
            #     message = message[capacity_t:]  # remove the encoded part of `message`

            wav.append(x.item())
        end = time.time()
        perplexity = 2**(-1 / len(wav) * total_log_probs)
        wav = np.asarray(wav, dtype=np.int)
        wav = librosa.mu_expand(wav - 2**(self.bits - 1), mu=2**self.bits - 1)
        # return wav, self.sr
        return SingleExampleOutput(None, wav, total_capacity, total_entropy, total_kld / len(wav), max_kld, perplexity,
                                   end - start, settings, total_minimum_entropy), self.sr
