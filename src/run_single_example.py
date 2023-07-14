import os
from typing import Optional
import torch
from scipy.io.wavfile import read, write
from PIL import Image

from config import Settings, text_default_settings, image_default_settings, audio_default_settings
from model import get_model, get_feature_extractor, get_tokenizer
from utils import SingleExampleOutput, check_dir

# Load message
message_file_path = os.path.join('temp', 'message.txt')
with open(message_file_path, 'r', encoding='utf-8') as f:
    message = f.read()
# message *= 10


def test_text(settings: Settings = text_default_settings, context: Optional[str] = None):
    if settings.algo == 'sample':
        from random_sample_cy import encode_text
    elif settings.algo in ['Discop', 'Discop_baseline']:
        from stega_cy import encode_text, decode_text
    else:
        raise NotImplementedError("`Settings.algo` must belong to {'Discop', 'Discop_baseline', 'sample'}!")

    if context is None:
        context = 'We were both young when I first saw you, I close my eyes and the flashback starts.'

    model = get_model(settings)
    tokenizer = get_tokenizer(settings)

    single_example_output: SingleExampleOutput = encode_text(model, tokenizer, message, context, settings)
    print(single_example_output)
    if settings.algo != 'sample':
        message_encoded = message[:single_example_output.n_bits]
        message_decoded = decode_text(model, tokenizer, single_example_output.generated_ids, context, settings)
        print(message_encoded)
        print(message_decoded)
        print(message_encoded == message_decoded)


def test_image(settings: Settings = image_default_settings,
               context_ratio: float = 0.5,
               original_img: Image = Image.open(os.path.join('temp', 'small.png'))):
    if settings.algo == 'sample':
        from random_sample_cy import encode_image
    elif settings.algo in ['Discop', 'Discop_baseline']:
        from stega_cy import encode_image, decode_image
    else:
        raise NotImplementedError("`Settings.algo` must belong to {'Discop', 'Discop_baseline', 'sample'}!")

    model = get_model(settings)
    feature_extractor = get_feature_extractor(settings)

    single_example_output: SingleExampleOutput = encode_image(model,
                                                              feature_extractor,
                                                              message,
                                                              context_ratio=context_ratio,
                                                              original_img=original_img)
    print(single_example_output)

    stego_img = single_example_output.stego_object
    # stego_img.save('stego.png')
    if settings.algo != 'sample':
        message_encoded = message[:single_example_output.n_bits]
        message_decoded = decode_image(model, feature_extractor, stego_img, context_ratio=context_ratio)
        print(message_encoded == message_decoded)


def test_tts(settings: Settings = audio_default_settings,
             text: str = "We are both young.",
             save_audio_dir: Optional[str] = 'temp'):
    from stega_tts import get_tts_model, encode_speech, decode_speech, random_sample_speech

    vocoder, tacotron, cmudict = get_tts_model(settings)

    # Encode
    if settings.algo == 'sample':
        single_example_output, sr = random_sample_speech(vocoder, tacotron, cmudict, message, text, settings)
    elif settings.algo in ['Discop', 'Discop_baseline']:
        single_example_output, sr = encode_speech(vocoder, tacotron, cmudict, message, text, settings)
    else:
        raise NotImplementedError("`Settings.algo` must belong to {'Discop', 'Discop_baseline', 'sample'}!")

    print(single_example_output)

    wav = single_example_output.stego_object
    if save_audio_dir is not None:
        check_dir(save_audio_dir)
        write(os.path.join(save_audio_dir, 'test.flac'), sr, wav)
    message_encoded = message[:single_example_output.n_bits]

    # Decode
    if settings.algo != 'sample':
        if save_audio_dir is not None:
            sr, wav = read(os.path.join(save_audio_dir, 'test.flac'))
        message_decoded = decode_speech(vocoder, tacotron, cmudict, wav, text, settings)
        # print(message_decoded)
        print(message_encoded == message_decoded)


if __name__ == '__main__':
    # Text Generation
    settings = text_default_settings
    settings.device = torch.device('cuda:0')
    # settings.algo = 'Discop_baseline'
    # settings.algo = 'sample'
    context = """I remember this film, it was the first film I had watched at the cinema."""
    test_text(settings, context)

    # Image Completion
    settings = image_default_settings
    settings.device = torch.device('cuda:0')
    # settings.algo = 'Discop_baseline'
    # settings.algo = 'sample'
    test_image(settings)

    # Text-to-Speech
    settings: Settings = audio_default_settings
    settings.device = torch.device('cuda:0')
    # settings.algo = 'Discop_baseline'
    # settings.algo = 'sample'
    settings.seed = 1  # debug
    settings.top_p = 0.98
    test_tts(settings)
