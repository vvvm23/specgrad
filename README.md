# (Unofficial) PyTorch Implemention of "SpecGrad: Diffusion Probabilistic Model based Neural Vocoder with Adaptive Noise Spectral Shaping"

> Thanks to Microsoft's implementation of
> [PriorGrad](https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder),
> which I use as a base of this implementation.

> **This repository is a work-in-progress and does not produce good outputs yet. Stay tuned!**

## Setup
To begin, create a python environment using your method of choice.

Then, run the following to install requirements
```shell
pip install -r requirements.txt
```

I use `accelerate` for data parallel training. Even if you only wish to train
on a single-device, run the following command:
```shell
accelerate config
```

If you want to run with PyTorch 2.0 support, run the following:
```shell
pip install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu{cuda version}
```
replacing `{cuda version}` with your installed CUDA version (eg: `118`)

Download [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) and extract to `dataset/ljspeech/`.

Create the `train`, `valid`, and `test` splits by creating text files in
`dataset/ljspeech/` with names `train.txt`, `valid.txt`, `test.txt`, containing
the line separated audio file paths.

## Usage
Run training using the following command:
```shell
accelerate launch train.py --config-path {optional-path-to-config-yaml}
```
Checkpoints will be stored in `exp/{date}_{time}`.

Once training is complete, run the following to run the inference loop on
a file of your choice:
```shell
accelerate launch inference.py {input-wav-file} {output-wav-file} --resume-dir {checkpoint-dir}
```
This script simply takes an input file, computes a mel-spectrogram, and
attempts to reconstruct the waveform using the mel-spectrogram alone.

## Configuration
`TODO: explain how to configure using yaml and command line`

### TODO:
- [X] Refactor model code copied from PriorGrad
- [X] Dataset preprocessing to support extra precomputed components of SpecGrad (TF-filter per input)
- [X] Configuration system
- [X] Basic training script
- [X] Basic inference script
- [X] Data parallel training
- [ ] **Debug bad results** <- We are here
- [ ] Optimisations

### Potential Future Features
- Fast inference schedulers
- Progressive distillation for fast sampling?
- Conditioning methods? (speaker embeddings)
- Larger dataset test?
- Explore other filter designs?

### Reference
**SpecGrad: Diffusion probabilistic model based neural vocoder with adaptive noise spectral shaping**
> Yuma Koizumi, Heiga Zen, Kohei Yatabe, Nanxin Chen, Michiel Bacchiani
```
@article{koizumi2022specgrad,
  title={SpecGrad: Diffusion probabilistic model based neural vocoder with adaptive noise spectral shaping},
  author={Koizumi, Yuma and Zen, Heiga and Yatabe, Kohei and Chen, Nanxin and Bacchiani, Michiel},
  journal={arXiv preprint arXiv:2203.16749},
  year={2022}
}
```
