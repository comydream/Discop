# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
from cython.operator cimport dereference as deref
from libc.math cimport log2
from libc.time cimport time, time_t, difftime
from libcpp cimport nullptr
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.memory cimport shared_ptr, make_shared

import random
import numpy as np
from PIL import Image
import torch
from scipy.stats import entropy
from tqdm import tqdm
# import time as py_time

from config import text_default_settings, image_default_settings, audio_default_settings
from model import get_model, get_tokenizer, get_feature_extractor
from utils import get_probs_indices_past, set_seed, SingleExampleOutput

## Classes & Structures
# Sampling (Encoding) results and statistics for single time step
cdef struct CySingleEncodeStepOutput:
    int sampled_index
    int n_bits
    double entropy_t
    double kld
    double minimum_entropy_t

cdef class SingleEncodeStepOutput:
    cdef public:
        int sampled_index, n_bits
        double entropy_t, kld, minimum_entropy_t
    def __init__(self,
                 int sampled_index,
                 int n_bits,
                 double entropy_t,
                 double kld,
                 double minimum_entropy_t):
        self.sampled_index = sampled_index
        self.n_bits = n_bits
        self.entropy_t = entropy_t
        self.kld = kld
        self.minimum_entropy_t = minimum_entropy_t

    def __call__(self):
        return self.sampled_index, self.n_bits, self.entropy_t, self.kld, self.minimum_entropy_t

    def __str__(self):
        d = {
            'sampled_index': self.sampled_index,
            'n_bits': self.n_bits,
            'entropy_t': self.entropy_t,
            'kld': self.kld,
            'minimum_entropy_t': self.minimum_entropy_t
        }
        return '\n'.join('{} = {}'.format(key, value) for (key, value) in d.items())


## Random sampling - single time step
cdef CySingleEncodeStepOutput cy_encode_step(list indices, list probs, string message_bits):
    # Encode step
    cdef:
        int index_idx, sampled_index
        double ptr = random.random(), minimum_entropy_t = -log2(probs[0]), entropy_t = entropy(probs, base=2)

    probs_cumsum = torch.tensor(probs).cumsum(dim=0)
    interval_begin = torch.cat((torch.tensor([0], device=probs_cumsum.device), probs_cumsum[:-1]), dim=0)
    index_idx = (ptr >= interval_begin).nonzero()[-1].item()
    sampled_index = indices[index_idx]

    return CySingleEncodeStepOutput(sampled_index, 0, entropy_t, 0, minimum_entropy_t)

## Random sampling - main loop
def encode(model, context, message_bits, settings, bint verbose = False, string tqdm_desc = b'Enc '):
    # Steganography Encoding (message_bits -> English text)
    cdef:
        int t = 0, length = settings.length, indices_idx
        double time_cost
        time_t start, end
        string stego_object
        list generated_ids = []

        # CySingleEncodeStepOutput
        CySingleEncodeStepOutput single_encode_step_output
        int sampled_index
        int capacity_t
        double entropy_t
        double kld_step
        double minimum_entropy_t

        # statistics
        int total_capacity = 0
        double total_entropy = 0.0
        double total_minimum_entropy = 0.0
        double total_log_probs = 0.0  # for perplexity
        double total_kld = 0.0
        double max_kld = 0.0
        double perplexity, ave_kld

    set_seed(settings.seed)

    past = None  # pass into the `past_keys_values` for speed-up
    prev = context  # indices that were never passed to the model before

    start = time(NULL)
    for t in tqdm(range(length), desc=tqdm_desc, ncols=70):
        probs, indices, past = get_probs_indices_past(model, prev, past, settings)
        probs = probs.tolist()
        indices = indices.tolist()

        single_encode_step_output = cy_encode_step(indices, probs, message_bits)
        sampled_index = single_encode_step_output.sampled_index
        capacity_t = single_encode_step_output.n_bits
        entropy_t = single_encode_step_output.entropy_t
        kld_step = single_encode_step_output.kld
        minimum_entropy_t = single_encode_step_output.minimum_entropy_t

        indices_idx = indices.index(sampled_index)

        # update statistics
        total_entropy += entropy_t
        total_minimum_entropy += minimum_entropy_t
        total_log_probs += log2(probs[indices_idx])
        total_kld += kld_step
        if kld_step > max_kld:
            max_kld = kld_step

        # when `capacity_t == 0`, cannot embed message, but still needs to return a token_index
        if capacity_t > 0:
            total_capacity += capacity_t
            message_bits = message_bits[capacity_t:]  # remove the encoded part of `message_bits`
        generated_ids.append(sampled_index)
        if settings.task == 'text':
            prev = torch.tensor([sampled_index], device=settings.device).unsqueeze(0)
        elif settings.task == 'image':
            prev = torch.tensor([sampled_index], device=settings.device)
    end = time(NULL)
    time_cost = difftime(end, start)

    perplexity = 2 ** (-1 / length * total_log_probs)
    ave_kld = total_kld / length
    return SingleExampleOutput(generated_ids, None, total_capacity, total_entropy, ave_kld, max_kld, perplexity,
                               time_cost, settings,
                               total_minimum_entropy)

def encode_text(model, tokenizer, message_bits, context, settings = text_default_settings):
    # tokenizer = get_tokenizer(settings)
    # model = get_model(settings)
    context = tokenizer(context, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(settings.device)

    single_encode_step_output = encode(model, context, message_bits, settings)
    single_encode_step_output.stego_object = tokenizer.decode(single_encode_step_output.generated_ids)

    return single_encode_step_output

def encode_image(model, feature_extractor, message_bits, settings = image_default_settings, bint verbose = False,
                 string tqdm_desc = b'Enc ',
                 double context_ratio = 0.0, original_img = None):
    cdef:
        int n_pixels_to_gen = 1024, n_pixels_context, n_px, width, height, width_after, height_after, height_context
    # feature_extractor = get_feature_extractor(settings)
    # model = get_model(settings)
    clusters = feature_extractor.clusters  # with shape (512, 3)
    n_px = feature_extractor.size  # 32

    context = torch.tensor([model.config.vocab_size - 1], device=settings.device)  # initialize with SOS token
    if context_ratio != 0.0:
        if original_img is None:
            raise ValueError('If you set `context_ratio`, please make sure that `original_img` is not None!')
        elif type(original_img) == str:
            img = Image.open(original_img)
        else:
            img = original_img
        width, height = img.size

        # resize if needed
        if width != n_px:
            width_after = n_px
            height_after = round(width_after / width * height)
            img = img.resize((width_after, height_after))
            width, height = width_after, height_after

        pixel_indices_lst = feature_extractor(img, return_tensors='pt')['input_ids'][0].to(settings.device)
        height_context = round(n_px * context_ratio)
        n_pixels_context = n_px * height_context
        primers = pixel_indices_lst[:n_pixels_context]
        context = torch.cat((context, primers), dim=-1)
        n_pixels_to_gen -= len(primers)

        output_pixel_ids = pixel_indices_lst.tolist()[:n_pixels_context]
    else:
        output_pixel_ids = []
    settings.length = n_pixels_to_gen

    single_encode_step_output = encode(model, context, message_bits, settings)
    output_pixel_ids.extend(single_encode_step_output.generated_ids)

    def pixel_ids_lst_to_pil_image(pixel_indices_lst):
        pixel_indices_array = np.array(pixel_indices_lst)
        img = Image.fromarray(
            np.reshape(np.rint(127.5 * (clusters[pixel_indices_array] + 1.0)), [n_px, n_px, 3]).astype(np.uint8))
        return img
    single_encode_step_output.stego_object = pixel_ids_lst_to_pil_image(output_pixel_ids)
    return single_encode_step_output

## Python interface
def encode_step(list indices, list probs, string message_bits):
    cdef CySingleEncodeStepOutput single_encode_step_output = cy_encode_step(indices, probs, message_bits)
    sampled_index = single_encode_step_output.sampled_index
    n_bits = single_encode_step_output.n_bits
    entropy_t = single_encode_step_output.entropy_t
    kld = single_encode_step_output.kld
    minimum_entropy_t = single_encode_step_output.minimum_entropy_t
    return SingleEncodeStepOutput(sampled_index, n_bits, entropy_t, kld, minimum_entropy_t)
