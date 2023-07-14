import os
import random
import torch
import torch.nn.functional as F
from typing import List, Tuple
from transformers import PreTrainedTokenizer, PreTrainedModel

from config import Settings


# Sampling (Encoding) results and statistics for single example
class SingleExampleOutput:
    def __init__(self, generated_ids, stego_object, n_bits, total_entropy, ave_kld, max_kld, perplexity, time_cost, settings,
                 total_minimum_entropy):
        self.generated_ids = generated_ids
        self.stego_object = stego_object
        self.algo = settings.algo
        self.temp = settings.temp
        self.top_p = settings.top_p
        self.n_bits = n_bits
        if generated_ids is not None:
            self.n_tokens = len(generated_ids)
        else:
            self.n_tokens = len(stego_object)
        self.total_entropy = total_entropy
        self.ave_kld = ave_kld
        self.max_kld = max_kld
        self.embedding_rate = n_bits / self.n_tokens
        self.utilization_rate = n_bits / total_entropy if total_entropy != 0 else 0
        self.perplexity = perplexity
        self.time_cost = time_cost
        self.total_minimum_entropy = total_minimum_entropy

    def __str__(self) -> str:
        d = self.__dict__
        excluded_attr = ['generated_ids']
        selected_attr = list(d.keys())
        for x in excluded_attr:
            selected_attr.remove(x)
        return '\n'.join('{} = {}'.format(key, d[key]) for key in selected_attr)


def set_seed(sd):
    random.seed(sd)


# The token indices should be filtered out and their corresponding reasons
# https://huggingface.co/gpt2/raw/main/vocab.json
# filter_out_indices_gpt = {
#     -1: "endoftext can't happen",
#     198: "1 newline can't happen",
#     628: "2 newlines can't happen",
#     220: "just one space can't happen",
#     302: "`\u0120re` can't happen",
#     797: "`\u0120Re` can't happen",
#     15860: "`\u0120Enh` can't happen",
#     2943: "`EC` can't happen",
#     764: "`\u0120.` (764) may cause failed decoding to `.` (13)",
#     837: "`\u0120,` (837) may cause failer decoding to `,` (11)"
# }
filter_out_indices_gpt = {
    -1: "endoftext can't happen",
    198: "1 newline can't happen",
    628: "2 newlines can't happen",
    764: "`\u0120.` (764) may cause failed decoding to `.` (13)",
    837: "`\u0120,` (837) may cause failer decoding to `,` (11)"
}
contain_dollar_lst = [
    3, 720, 7198, 13702, 16763, 17971, 22799, 25597, 29568, 29953, 32047, 32382, 32624, 34206, 35307, 36737, 38892, 39280, 40111,
    43641, 45491, 47113, 48082
]
contain_bad_ellipsis_lst = [19424, 20004, 39864, 44713, 44912, 47082]


def gen_random_message(seed=None, length: int = 1000, save_path: str = os.path.join('temp', 'message.txt')) -> None:
    # Generating binary message (str) randomly via build-in `random` lib
    import random
    random.seed(seed)

    message = ''
    for _ in range(length):
        message += str(random.randint(0, 1))
    print(message)

    if save_path is None:
        return message
    with open(save_path, 'w', encoding='utf-8') as fout:
        fout.write(message)


def limit_past(past):
    if past is None:
        return None
    past = list(past)
    for i in range(len(past)):
        past[i] = list(past[i])
        for j in range(len(past[i])):
            past[i][j] = past[i][j][:, :, -1022:]
    return past


@torch.no_grad()
def get_probs_indices_past(model: PreTrainedModel,
                           prev=None,
                           past=None,
                           settings: Settings = Settings(),
                           gpt_filter: bool = True) -> Tuple:
    # first, get logits from the model
    if settings.task == 'text':
        if 'gpt2' in settings.model_name:
            past = limit_past(past)
            model_output = model(prev, past_key_values=past)
            past = model_output.past_key_values
            logits = model_output.logits[0, -1, :].to(settings.device)
            if gpt_filter:
                for ele in filter_out_indices_gpt.keys():
                    logits[ele] = -1e10
        elif settings.model_name == 'transfo-xl-wt103':
            model_output = model(prev, mems=past)
            past = model_output.mems
            logits = model_output.logits[0, -1, :].to(settings.device)
            logits[0] = -1e10  # <eos>
            logits[24] = -1e10  # <unk>
    elif settings.task == 'image':
        model_output = model(prev, past_key_values=past)
        past = model_output.past_key_values
        logits = model_output.logits[0, :].to(settings.device)

    logits, indices = logits.sort(descending=True)
    logits = logits.double()
    indices = indices.int()

    if settings.temp is None:
        settings.temp = 1.0
    logits_temp = logits / settings.temp
    probs = F.softmax(logits_temp, dim=-1)

    # Getting the top-p `probs` and `indices` from the last layer of `logits`
    if not (settings.top_p is None or settings.top_p == 1.0):
        assert settings.top_p > 0 and settings.top_p < 1.0, '`top_p` must be >0 and <=1!'
        cum_probs = probs.cumsum(0)
        k = (cum_probs > settings.top_p).nonzero()[0].item() + 1
        probs = probs[:k]
        indices = indices[:k]
        probs = 1 / cum_probs[k - 1] * probs  # Normalizing
    return probs, indices, past


def is_alpha(s: str) -> bool:
    # A-Za-z
    for i in range(len(s)):
        c = s[i].lower()
        if ord(c) < ord('a') or ord(c) > ord('z'):
            return False
    return True


def check_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
        print('A folder called "{}" is created.'.format(dir))


if __name__ == '__main__':
    gen_random_message(length=1000000)
