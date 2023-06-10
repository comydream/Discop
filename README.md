# Discop

Discop: Provably Secure Steganography in Practice Based on “Distribution Copies”

[Jinyang Ding](https://dingjinyang.github.io/), [Kejiang Chen](http://home.ustc.edu.cn/~chenkj/), [Yaofei Wang](http://faculty.hfut.edu.cn/yaofeiwang/en/index.htm), Na Zhao, [Weiming Zhang](http://staff.ustc.edu.cn/~zhangwm/), and [Nenghai Yu](http://staff.ustc.edu.cn/~ynh/)
In [IEEE Symposium on Security and Privacy (IEEE S&P) 2023](https://sp2023.ieee-security.org/)

[[PDF]](https://dingjinyang.github.io/uploads/Discop_sp23_paper.pdf) [[Cite]](#citation) [[Slides]](https://dingjinyang.github.io/uploads/Discop_sp23_slides.pdf) [[IEEE Computer Society]](https://doi.ieeecomputersociety.org/10.1109/SP46215.2023.00155) [[Chinese Blog Post]](https://comydream.github.io/2023/06/07/discop-sp23/)

## Brief Overview

Given a probability distribution to sample from, if we want to embed $n$ bits, we construct $2^{n}$ copies of the distribution by rotation and use the copy index to express information.

![](rotate.png)

To improve the embedding rate, we decompose the multi-variate distribution into multiple bi-variate distributions through a Huffman tree.

![](recursion.png)

The embedding rate can reach about 0.95 of its theoretical limit.

## Usage

> The code is still being collated.

### Preparation

First, please ensure that you have installed all the required libraries for this repository.

We recommend using [Anaconda](https://anaconda.org/anaconda/conda) and execute the following commands.

```shell
conda create -n discop python=3.8.12
conda activate discop

# Visit the PyTorch website (https://pytorch.org/get-started/locally/) for installation commands tailored to your environment
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
# Install other dependencies
python -m pip install -r requirements.txt

# Build the Cython files
python src/setup.py build_ext --build-lib=src/
```

### Run Single Example

Please note that the program may automatically download the required pretrained models and datasets during the first run.

```shell
python src/run_single_example.py
```

### Get Statistics

```shell
python src/get_statistics.py
```

## Acknowledgment

In the text generation and image completion tasks, we directly employ the pre-trained models provided by [Hugging Face](https://huggingface.co/models).

In the text-to-speech (TTS) task, we utilize publicly available pre-trained models from [bshall/Tacotron](https://github.com/bshall/Tacotron/tree/main/tacotron) and [bshall/UniversalVocoding](https://github.com/bshall/UniversalVocoding).

## Citation

If you find this work useful, please consider citing:

```
@inproceedings{dingDiscopProvablySecure2023,
  title      = {Discop: {{Provably Secure Steganography}} in {{Practice Based}} on ``{{Distribution Copies}}''},
  shorttitle = {Discop},
  booktitle  = {2023 {{IEEE Symposium}} on {{Security}} and {{Privacy}} ({{SP}})},
  author     = {Ding, Jinyang and Chen, Kejiang and Wang, Yaofei and Zhao, Na and Zhang, Weiming and Yu, Nenghai},
  year       = {2023},
  month      = may,
  pages      = {2238--2255},
  publisher  = {{IEEE Computer Society}},
  address    = {Los Alamitos, CA, USA},
  doi        = {10.1109/SP46215.2023.00155},
  url        = {https://doi.ieeecomputersociety.org/10.1109/SP46215.2023.00155},
  isbn       = {978-1-66549-336-9},
  langid     = {english}
}
```
