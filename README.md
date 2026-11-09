# SPRA-LoRA: Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA

## 📌 TL;DR
SPRA-LoRA is a parameter-efficient federated fine-tuning framework for Large Language Models. It freezes and periodically resamples random projection matrices while dynamically scaling learning rates based on client adaptation ranks, significantly reducing communication overhead while maintaining model performance.

---

## 🎨 Overview & Architecture

<p align="center">
  <img src="Figures/spralora.png" alt="SPRA-LoRA Overview" width="800"/>
</p>

> *Figure 1: High-level overview of the SPRA-LoRA architecture, depicting stochastic projection resampling requiring rank-aware learning rates across clients.*

---

## 📁 Repository Structure

```text
.
├── GLUE/                              # Implementation codes and files for the GLUE benchmark
├── Common Sense/                      # Implementation codes and files for the Common Sense dataset
├── Figures/                           # Figures and visual assets for documentation
├── Commonsense_requirements.text      # Required Python packages for the Common Sense implementation
├── GLUE_requirements.text             # Required Python packages for the GLUE implementation
└── README.md                          # Repository documentation
