# kvquant-lab

**Research in progress: when does KV-cache quantization for LLMs actually work?**

This project studies the TurboQuant quantizers ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026) against plain MSE quantization under equal bit budgets. The evaluation is being rebuilt around one bit-accounting function, config-driven logged runs and preregistered predictions, so that every number can be reproduced.

## Status

- This repository holds the first-generation implementation.
- The rebuilt evaluation is in progress and supersedes the first-generation results; its results will be published here when it is complete.

## Layout

| Folder | Contents |
|---|---|
| `turboquant/` | Quantizer implementations |
| `benchmarks/` | Evaluation scripts |
| `training/` | Memory-caching training script |
| `paper/`, `results/` | First-generation drafts and outputs (superseded) |

## Setup

```bash
pip install torch transformers datasets rouge-score scipy
python turboquant/turbo_quant_demo.py
```

Hardware used so far: NVIDIA RTX 4060 Laptop GPU.

## References

```bibtex
@article{zandieh2026turboquant,
  title={Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Ali and Mirrokni, Vahab},
  journal={arXiv:2504.19874},
  year={2026}
}

@article{behrouz2026mc,
  title={Memory Caching: RNNs with Growing Memory},
  author={Behrouz, Ali and Li, Zeman and Deng, Yuan and Zhong, Peilin and Razaviyayn, Meisam and Mirrokni, Vahab},
  journal={arXiv:2602.24281},
  year={2026}
}
```
