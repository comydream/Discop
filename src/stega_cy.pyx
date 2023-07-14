# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
from cython.operator cimport dereference as deref
from libc.math cimport log2
from libcpp cimport nullptr
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.memory cimport shared_ptr, make_shared

import time
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

cdef bint msg_exhausted_flag = False

## Classes & Structures
# Nodes of Huffman tree 
cdef struct Node:
    double prob
    shared_ptr[Node] left
    shared_ptr[Node] right
    int index
    # >=0 - index
    # -1 - None
    int search_path
# 0  - this node
# -1 - in left subtree
# 1  - in right subtree
# 9  - unknown


cdef inline bint is_leaf(shared_ptr[Node] node_ptr):
    return deref(node_ptr).index != -1

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


# Decoding results for single time step
cdef struct CySingleDecodeStepOutput:
    string message_decoded_t

cdef class SingleDecodeStepOutput:
    cdef public:
        string message_decoded_t

    def __init__(self, string message_decoded_t) -> None:
        self.message_decoded_t = message_decoded_t

    def __call__(self):
        return self.message_decoded_t

## Utils
# Building a Huffman tree
cdef shared_ptr[Node] create_huffman_tree(list indices, list probs, int search_for):
    # Returns a pointer to the root node of the Huffman tree
    # if `search_for == -1`, we don't need to initialize the `search_path` of any Node object
    cdef:
        int sz = len(indices)
        int i, search_path
        double prob
        shared_ptr[Node] node_ptr, first, second, ans
        queue[shared_ptr[Node]] q1, q2

    for i in range(sz - 1, -1, -1):
        # search_path = 0 if search_for == indices[i] else 9
        if search_for == indices[i]:
            search_path = 0
        else:
            search_path = 9
        node_ptr = make_shared[Node](
            Node(probs[i], shared_ptr[Node](nullptr), shared_ptr[Node](nullptr), indices[i], search_path))
        q1.push(node_ptr)

    while q1.size() + q2.size() > 1:
        # first
        if not q1.empty() and not q2.empty() and deref(q1.front()).prob < deref(q2.front()).prob:
            first = q1.front()
            q1.pop()
        elif q1.empty():
            first = q2.front()
            q2.pop()
        elif q2.empty():
            first = q1.front()
            q1.pop()
        else:
            first = q2.front()
            q2.pop()

        # second
        if not q1.empty() and not q2.empty() and deref(q1.front()).prob < deref(q2.front()).prob:
            second = q1.front()
            q1.pop()
        elif q1.empty():
            second = q2.front()
            q2.pop()
        elif q2.empty():
            second = q1.front()
            q1.pop()
        else:
            second = q2.front()
            q2.pop()

        # merge
        prob = deref(first).prob + deref(second).prob
        search_path = 9
        if deref(first).search_path != 9:
            search_path = -1
        elif deref(second).search_path != 9:
            search_path = 1
        q2.push(make_shared[Node](Node(prob, first, second, -1, search_path)))

    if not q2.empty():
        ans = q2.front()
    else:
        ans = q1.front()
    return ans

## Steganography process - single time step
# Sampling (Encoding) - single time step
cdef CySingleEncodeStepOutput cy_encode_step(list indices, list probs, string message_bits):
    # Encode step
    global msg_exhausted_flag
    cdef:
        int sampled_index, n_bits = 0
        double entropy_t = 0.0, kld = 0.0, minimum_entropy_t = 0.0, prob_sum, ptr, ptr_0, ptr_1, partition
        shared_ptr[Node] node_ptr = create_huffman_tree(indices, probs, -1)
        vector[int] path_table = [-1, 1]
        int len_message_bits = len(message_bits)

    # if len_message_bits > 0:
    #     print('len(message_bits) = {}'.format(len_message_bits))
    while not is_leaf(node_ptr):  # non-leaf node
        prob_sum = deref(node_ptr).prob
        ptr = random.random()
        ptr_0 = ptr * prob_sum
        ptr_1 = (ptr + 0.5) * prob_sum
        if ptr_1 > prob_sum:
            ptr_1 -= prob_sum

        partition = deref(deref(node_ptr).left).prob

        # path_table[0] = -1 if (ptr_0 < partition) else 1
        if ptr_0 < partition:
            path_table[0] = -1
        else:
            path_table[0] = 1
        # path_table[1] = -1 if (ptr_1 < partition) else 1
        if ptr_1 < partition:
            path_table[1] = -1
        else:
            path_table[1] = 1

        # node_ptr = deref(node_ptr).right if path_table[message_bits[n_bits] - 48] == 1 else deref(node_ptr).left
        if not msg_exhausted_flag and (len_message_bits <= n_bits):
            print('[*] The message is exhausted and will be padded with all zeros!')
            msg_exhausted_flag = True
        # print(n_bits)
        if msg_exhausted_flag:
            if path_table[0] == 1:
                node_ptr = deref(node_ptr).right
            else:
                node_ptr = deref(node_ptr).left
        else:
            if path_table[message_bits[n_bits] - 48] == 1:
                node_ptr = deref(node_ptr).right
            else:
                node_ptr = deref(node_ptr).left

        if path_table[0] != path_table[1]:
            n_bits += 1
    # print(deref(node_ptr).index)
    sampled_index = deref(node_ptr).index
    minimum_entropy_t = -log2(probs[0])
    entropy_t = entropy(probs, base=2)
    return CySingleEncodeStepOutput(sampled_index, n_bits, entropy_t, kld, minimum_entropy_t)


# Discop Baseline Sampling (Encoding) - single time step
cdef CySingleEncodeStepOutput cy_baseline_encode_step(list indices, list probs, string message_bits):
    # Encode step
    global msg_exhausted_flag
    cdef:
        int sampled_index, n_bits = 0, capacity, capacity_upper_bound, i
        double entropy_t = 0.0, kld = 0.0, minimum_entropy_t = 0.0, ptr, ptr_i, rotate_step_size
        int len_message_bits = len(message_bits)
    
    probs_cumsum = torch.tensor(probs).cumsum(dim=0)
    interval_begin = torch.cat((torch.tensor([0], device=probs_cumsum.device), probs_cumsum[:-1]), dim=0)

    # Determine capacity
    capacity = int(log2(1 / probs[0]))
    capacity_upper_bound = capacity + 1

    tbl = {}  # message bits -> token_index
    ptr = random.random()

    while capacity <= capacity_upper_bound:
        if capacity == 0:
            capacity += 1
            continue
        rotate_step_size = 2.0**-capacity
        is_available = True
        tbl_new = {}
        for i in range(2**capacity):
            ptr_i = ptr + i * rotate_step_size
            if ptr_i >= 1.0:
                ptr_i -= 1
            index_idx = (ptr_i >= interval_begin).nonzero()[-1].item()
            index = indices[index_idx]
            if index in tbl_new.values():
                is_available = False
                break
            tbl_new[i] = index
        if not is_available:
            break
        tbl = tbl_new
        n_bits = capacity
        capacity += 1
    if n_bits < 1:
        sampled_index = indices[(ptr >= interval_begin).nonzero()[-1].item()]
    else:
        cur_message_bits_decimal = 0
        base = 1
        for d in range(n_bits - 1, -1, -1):
            if message_bits[d] == b'1':
                cur_message_bits_decimal += base
            base *= 2
        sampled_index = tbl[cur_message_bits_decimal]

    minimum_entropy_t = -log2(probs[0])
    entropy_t = entropy(probs, base=2)
    return CySingleEncodeStepOutput(sampled_index, n_bits, entropy_t, kld, minimum_entropy_t)


# Decoding - single time step
cdef CySingleDecodeStepOutput cy_decode_step(list indices, list probs, int stego_t):
    # Decode step
    cdef:
        string message_decoded_t
        double prob_sum, ptr, ptr_0, ptr_1, partition
        shared_ptr[Node] node_ptr = create_huffman_tree(indices, probs, stego_t)
        vector[int] path_table = vector[int](2)
        map[int, string] path_table_swap

    while not is_leaf(node_ptr):  # non-leaf node
        prob_sum = deref(node_ptr).prob
        ptr = random.random()
        ptr_0 = ptr * prob_sum
        ptr_1 = (ptr + 0.5) * prob_sum
        if ptr_1 > prob_sum:
            ptr_1 -= prob_sum

        partition = deref(deref(node_ptr).left).prob

        # path_table[0] = -1 if (ptr_0 < partition) else 1
        if ptr_0 < partition:
            path_table[0] = -1
        else:
            path_table[0] = 1
        # path_table[1] = -1 if (ptr_1 < partition) else 1
        if ptr_1 < partition:
            path_table[1] = -1
        else:
            path_table[1] = 1

        if path_table[0] != path_table[1]:  # can embed 1 bit
            if deref(node_ptr).search_path == 9:  # fail to decode
                message_decoded_t = b'x'
                break

            if path_table[0] == -1:
                path_table_swap[-1] = b'0'
                path_table_swap[1] = b'1'
            else:
                path_table_swap[-1] = b'1'
                path_table_swap[1] = b'0'
            message_decoded_t += path_table_swap[deref(node_ptr).search_path]

            # walk
            if deref(node_ptr).search_path == -1:
                node_ptr = deref(node_ptr).left
            else:
                node_ptr = deref(node_ptr).right
        else:
            if path_table[0] == -1:
                node_ptr = deref(node_ptr).left
            else:
                node_ptr = deref(node_ptr).right

    if deref(node_ptr).search_path != 0:  # cannot reach a leaf node
        message_decoded_t = b'x'
    return CySingleDecodeStepOutput(message_decoded_t)


# Discop Baseline Decoding - single time step
cdef CySingleDecodeStepOutput cy_baseline_decode_step(list indices, list probs, int stego_t):
    # Decode step
    cdef:
        int capacity, capacity_upper_bound, n_bits = 0
        string message_decoded_t
        double ptr
    probs_cumsum = torch.tensor(probs).cumsum(dim=0)
    interval_begin = torch.cat((torch.tensor([0], device=probs_cumsum.device), probs_cumsum[:-1]), dim=0)

    # Determine capacity
    capacity = int(log2(1 / probs[0]))
    capacity_upper_bound = capacity + 1

    tbl = {}  # message bits -> token_index
    ptr = random.random()

    while capacity <= capacity_upper_bound:
        if capacity == 0:
            capacity += 1
            continue
        rotate_step_size = 2.0**-capacity
        is_available = True
        tbl_new = {}
        for i in range(2**capacity):
            ptr_i = ptr + i * rotate_step_size
            if ptr_i >= 1.0:
                ptr_i -= 1
            index_idx = (ptr_i >= interval_begin).nonzero()[-1].item()
            index = indices[index_idx]
            if index in tbl_new.values():
                is_available = False
                break
            tbl_new[i] = index
        if not is_available:
            break
        tbl = tbl_new
        n_bits = capacity
        capacity += 1
    if n_bits < 1:
        message_decoded_t = b''
    else:
        if stego_t not in tbl.values():  # Error
            message_decoded_t = b'x'
        tbl_swapped = dict(zip(tbl.values(), tbl.keys()))  # token_index -> message bits
        message_decoded_t = bin(tbl_swapped[stego_t])[2:].zfill(n_bits)
    return CySingleDecodeStepOutput(message_decoded_t)


def encode(model, context, message_bits, settings, bint verbose = False, string tqdm_desc = b'Enc '):
    # Steganography Encoding (message_bits -> English text)
    cdef:
        int t = 0, length = settings.length, indices_idx
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

    start = time.time()
    for t in tqdm(range(length), desc=tqdm_desc, ncols=70):
        probs, indices, past = get_probs_indices_past(model, prev, past, settings)
        probs = probs.tolist()
        indices = indices.tolist()

        if settings.algo == 'Discop':
            single_encode_step_output = cy_encode_step(indices, probs, message_bits)
        elif settings.algo == 'Discop_baseline':
            single_encode_step_output = cy_baseline_encode_step(indices, probs, message_bits)
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
    end = time.time()

    perplexity = 2 ** (-1 / length * total_log_probs)
    ave_kld = total_kld / length
    return SingleExampleOutput(generated_ids, None, total_capacity, total_entropy, ave_kld, max_kld, perplexity,
                               end - start, settings,
                               total_minimum_entropy)

def decode(model, context, list stego, settings, bint verbose = False, string tqdm_desc = b'Dec '):
    # Steganography Decoding (English text -> message_bits)
    cdef:
        int t = 0, length = len(stego), indices_idx
        string message_decoded

        # CySingleEncodeStepOutput
        CySingleDecodeStepOutput single_decode_step_output
        int sampled_index
        int capacity_t
        double entropy_t
        double kld_step
        double minimum_entropy_t

    set_seed(settings.seed)
    past = None  # pass into the `past_keys_values` for speed-up
    prev = context  # indices that were never passed to the model before

    start = time.time()
    # while t < length:
    for t in tqdm(range(length), desc=tqdm_desc, ncols=70):
        probs, indices, past = get_probs_indices_past(model, prev, past, settings)
        probs = probs.tolist()
        indices = indices.tolist()

        if settings.algo == 'Discop':
            single_decode_step_output = cy_decode_step(indices, probs, stego[t])
        elif settings.algo == 'Discop_baseline':
            single_decode_step_output = cy_baseline_decode_step(indices, probs, stego[t])
        message_decoded_t = single_decode_step_output.message_decoded_t

        # print(message_decoded_t)

        if message_decoded_t == b'x':
            print('Fail to decode!')
            break
        message_decoded += message_decoded_t
        if settings.task == 'text':
            prev = torch.tensor([stego[t]], device=settings.device).unsqueeze(0)
        elif settings.task == 'image':
            prev = torch.tensor([stego[t]], device=settings.device)
    end = time.time()
    print('Decode time = {}s'.format(end - start))
    return message_decoded

def encode_text(model, tokenizer, message_bits, context, settings = text_default_settings):
    # tokenizer = get_tokenizer(settings)
    # model = get_model(settings)
    context = tokenizer(context, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(settings.device)

    single_encode_step_output = encode(model, context, message_bits, settings)
    single_encode_step_output.stego_object = tokenizer.decode(single_encode_step_output.generated_ids)

    return single_encode_step_output

def decode_text(model, tokenizer, list stego, context, settings = text_default_settings):
    # tokenizer = get_tokenizer(settings)
    # model = get_model(settings)
    context = tokenizer(context, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(settings.device)

    message_decoded = decode(model, context, stego, settings)

    return message_decoded

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

def decode_image(model, feature_extractor, img, settings = image_default_settings, bint verbose = False,
                 string tqdm_desc = b'Dec ',
                 double context_ratio = 0.0):
    cdef:
        int height_context, n_pixels_context, n_pixels_to_gen = 1024, n_px
        list stego
        string message_decoded
    if type(img) == str:
        img = Image.open(img)

    # feature_extractor = get_feature_extractor(settings)
    # model = get_model(settings)
    clusters = feature_extractor.clusters  # with shape (512, 3)
    n_px = feature_extractor.size  # 32

    stego = feature_extractor(img)['input_ids'][0].tolist()
    if len(stego) != 1024:
        raise ValueError('len(stego) must be 1024!')

    context = torch.tensor([model.config.vocab_size - 1], device=settings.device)  # initialize with SOS token
    if context_ratio != 0.0:
        pixel_indices_lst = feature_extractor(img, return_tensors='pt')['input_ids'][0].to(settings.device)
        height_context = round(n_px * context_ratio)
        n_pixels_context = n_px * height_context
        primers = pixel_indices_lst[:n_pixels_context]
        context = torch.cat((context, primers), dim=-1)
        stego = stego[n_pixels_context:]
        n_pixels_to_gen -= len(primers)

    settings.length = n_pixels_to_gen

    message_decoded = decode(model, context, stego, settings)
    return message_decoded

## Python interface
def encode_step(settings, list indices, list probs, string message_bits):
    cdef CySingleEncodeStepOutput single_encode_step_output
    if settings.algo == 'Discop':
        single_encode_step_output = cy_encode_step(indices, probs, message_bits)
    elif settings.algo == 'Discop_baseline':
        single_encode_step_output = cy_baseline_encode_step(indices, probs, message_bits)
    sampled_index = single_encode_step_output.sampled_index
    n_bits = single_encode_step_output.n_bits
    entropy_t = single_encode_step_output.entropy_t
    kld = single_encode_step_output.kld
    minimum_entropy_t = single_encode_step_output.minimum_entropy_t
    return SingleEncodeStepOutput(sampled_index, n_bits, entropy_t, kld, minimum_entropy_t)

def decode_step(settings, list indices, list probs, int stego_t):
    cdef CySingleDecodeStepOutput single_decode_step_output
    if settings.algo == 'Discop':
        single_decode_step_output = cy_decode_step(indices, probs, stego_t)
    elif settings.algo == 'Discop_baseline':
        single_decode_step_output = cy_baseline_decode_step(indices, probs, stego_t)
    if single_decode_step_output.message_decoded_t == b'x':
        raise ValueError('Fail to decode!')
    return single_decode_step_output.message_decoded_t
