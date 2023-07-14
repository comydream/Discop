import torch
from tacotron import load_cmudict, Tacotron, text_to_id
from univoc import Vocoder

from config import Settings, audio_default_settings


def get_tts_model(settings: Settings):
    assert settings.task == 'text-to-speech' and settings.model_name == 'univoc'
    vocoder = Vocoder.from_pretrained(
        "https://github.com/bshall/UniversalVocoding/releases/download/v0.2/univoc-ljspeech-7mtpaq.pt",
        map_location=settings.device).to(settings.device)
    tacotron = Tacotron.from_pretrained("https://github.com/bshall/Tacotron/releases/download/v0.1/tacotron-ljspeech-yspjx3.pt",
                                        map_location=settings.device).to(settings.device)
    cmudict = load_cmudict()
    return vocoder, tacotron, cmudict


def encode_speech(vocoder: Vocoder,
                  tacotron: Tacotron,
                  cmudict,
                  message_bits: str,
                  text: str,
                  settings: Settings = audio_default_settings,
                  verbose: bool = False,
                  tqdm_desc: str = 'Enc '):
    x = torch.tensor(text_to_id(text, cmudict), dtype=torch.long, device=settings.device).unsqueeze(0)
    mel, _ = tacotron.generate(x)
    mel = mel.transpose(1, 2)

    single_encode_step_output, sr = vocoder.encode_speech(mel, message_bits, settings=settings, tqdm_desc=tqdm_desc)

    return single_encode_step_output, sr


def decode_speech(vocoder: Vocoder,
                  tacotron: Tacotron,
                  cmudict,
                  speech,
                  text: str,
                  settings: Settings = audio_default_settings,
                  verbose: bool = False,
                  tqdm_desc: str = 'Dec '):
    x = torch.tensor(text_to_id(text, cmudict), dtype=torch.long, device=settings.device).unsqueeze(0)
    mel, _ = tacotron.generate(x)
    mel = mel.transpose(1, 2)

    message_decoded = vocoder.decode_speech(mel, speech, settings=settings, tqdm_desc=tqdm_desc)

    return message_decoded


def random_sample_speech(vocoder,
                         tacotron,
                         cmudict,
                         message_bits,
                         text: str,
                         settings: Settings = audio_default_settings,
                         verbose: bool = False,
                         tqdm_desc: str = 'Enc '):
    x = torch.tensor(text_to_id(text, cmudict), dtype=torch.long, device=settings.device).unsqueeze(0)
    mel, _ = tacotron.generate(x)
    mel = mel.transpose(1, 2)

    single_encode_step_output, sr = vocoder.random_sample_speech(mel, message_bits, tqdm_desc=tqdm_desc)

    return single_encode_step_output, sr