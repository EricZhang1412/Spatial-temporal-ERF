# Unveiling the Spatial-temporal Effective Receptive Fields of Spiking Neural Networks
![Endpoint Badge](https://img.shields.io/endpoint)


This repository provides the official implementation of the paper **"Unveiling the Spatial-temporal Effective Receptive Fields of Spiking Neural Networks"** (NeurIPS 2025).

## 📁 File Structure

```
 
```

---

## ⚙️ Prerequisites

The following setup has been tested and verified to work:

- **Python** ≥ 3.5
- **PyTorch** ≥ 1.9.0
- **CUDA** ≥ 10.2

## 🚀 Quick Start

### Run Main Training
```
python main_training_parallel.py
```

### Implementation Notes
- **Synaptic weight regulation**: Located in `./models/quant_function.py` (lines 22-39, `QCon2d` class)
- **Spike activity regulation**: Implemented in `functions.py` (lines 39-50, `proposedLoss` function)

---

## 🙏 Acknowledgments

This code is built upon the [Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting](https://github.com/brain-intelligence-lab/temporal_efficient_training) code. We thank the original authors for their valuable work.

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wei2024q,
  title={Q-SNNs: Quantized Spiking Neural Networks},
  author={Wei, Wenjie and Liang, Yu and Belatreche, Ammar and Xiao, Yichen and Cao, Honglin and Ren, Zhenbang and Wang, Guoqing and Zhang, Malu and Yang, Yang},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={8441--8450},
  year={2024}
}
```

---

## 📧 Contact

For questions regarding this implementation, please contact: **Wenjie Wei** 📧 wjwei@std.uestc.edu.cn
