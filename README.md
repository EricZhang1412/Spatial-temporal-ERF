# Unveiling the Spatial-temporal Effective Receptive Fields of Spiking Neural Networks
![Static Badge](https://img.shields.io/badge/NeurIPS-2025-blue)



This repository provides the official implementation of the paper **"Unveiling the Spatial-temporal Effective Receptive Fields of Spiking Neural Networks"** (NeurIPS 2025).

## 📁 File Structure

```
.
├── erf_compute # Main folder for ST-ERF
├── det # Codebase for Detection Experiments (Originated from mmsegmentation by OpenMMLab)
├── seg # 
└── README.md
```

---

## 🚀 Quick Start

### Environment Setup

Set your python environment with pytorch (lts or 2.6.0+)...

### Play with Spatial-temporal Effective Receptive Field

Go to the directory `erf_compute/`. The structure of this folder is as follows:
```
.
├── izhikevich.py
├── LICENSE
├── spatial_Erf # Spatial ERF Code
│   ├── erf_scnn # S-ERF for Spiking-CNNs
│   └── erf_sdt  # S-ERF for Spiking-Transformers
└── temporal_erf_compute.py # Temporal ERF Code
```

## 🙏 Acknowledgments

This whole project is influenced by [*![Static Badge](https://img.shields.io/badge/NeurIPS-2016-yellow)
Understanding the Effective Receptive Field in Deep Convolutional Neural Networks*](https://papers.nips.cc/paper_files/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html) .


## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
[TODO]
```

---

## 📧 Contact

For questions regarding this implementation, please contact: **Jieyuan/Eric** 📧 ericzh_uestc@std.uestc.edu.cn
