# SPRA-LoRA: Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA

## 📌 TL;DR
SPRA-LoRA is a parameter-efficient federated fine-tuning framework for Large Language Models. It freezes and periodically resamples random projection matrices while dynamically scaling learning rates based on client adaptation ranks, significantly reducing communication overhead while maintaining model performance.

---

## 🎨 Overview & Architecture

<p align="center">
  <img src="Figures/spralora.png" alt="SPRA-LoRA Overview" width="800"/>
</p>

> *Figure 1: High-level overview of the SPRA-LoRA architecture, depicting stochastic projection resampling and rank-aware learning rate adaptation across federated clients.*

---

## 📁 Repository Structure

```text
.
├── GLUE/          # Implementation, scripts, and evaluation for GLUE benchmark datasets
├── Common Sense/  # Implementation, scripts, and evaluation for Common Sense reasoning datasets
├── Figures/        # Figures and visual assets for documentation
├── Figures/        # Figures and visual assets for documentation
├── Commonsense_requirements.text        # Required python pacakges for the Commonsense implementation codes
├── GLUE_requirements.text        # Required python pacakges for the GLUE implementation codes
└── README.md      # Repository documentation

--- 
## 📁 Citation
If you find this work or repository useful in your research, please cite our paper:

```bibtex
@inproceedings{
spralora,
title={Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA},
author={Saber Malekmohammadi and Kevin Kuo and Virginia Smith and Golnoosh Farnadi},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=uPdmJxDBM2}
}
```

