# (Unofficial) PyTorch Implemention of "SpecGrad: Diffusion Probabilistic Model based Neural Vocoder with Adaptive Noise Spectral Shaping"

> Thanks to Microsoft's implementation of
> [PriorGrad](https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder),
> which I use as a base of this implementation.

## Setup
`TODO`

## Usage
`TODO`

## Configuration
`TODO`

### TODO:
- [X] Refactor model code copied from PriorGrad
- [X] Dataset preprocessing to support extra precomputed components of SpecGrad (TF-filter per input)
- [X] Configuration system
- [X] Basic training script
- [ ] Basic inference script (w/ HF schedulers)
- [X] Data parallel training (w/ HF accelerate)
- [ ] Optimisations

### Potential Future Features
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
