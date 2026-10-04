# ZT-CACC: Zero-Trust Transaction Verification & Resilient CACC Platooning over Heterogeneous Multi-RAT V2X Networks

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Target: IEEE Transactions](https://img.shields.io/badge/Target%20Venue-IEEE%20Transactions-00629B.svg)](https://ieeexplore.ieee.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI Tests](https://img.shields.io/badge/Tests-Passing%20(100%25)-brightgreen.svg)]()
[![Artifacts: Gold Standard](https://img.shields.io/badge/Artifacts-100%25%20Reproducible-success.svg)]()
[![Dataset: VeReMi](https://img.shields.io/badge/Dataset-Augmented%20VeReMi-orange.svg)](https://github.com/VeReMi-dataset/VeReMi)

> **Authors**: **Umer Tanveer** and **Abdul Salam**  
> **Repository**: [https://github.com/umertanveer25/ZT-CACC](https://github.com/umertanveer25/ZT-CACC)  
> **Manuscript**: *ZT-CACC: A Multi-Modal Zero-Trust Verification Framework with Adaptive Multi-RAT Switching for Resilient Connected Vehicle Platooning*

---

## 📖 Table of Contents
1. [Overview & Core Architecture](#-overview--core-architecture)
2. [Mathematical Foundations & Theoretical Deadlines](#-mathematical-foundations--theoretical-deadlines)
3. [Quickstart & Reproduction Commands](#-quickstart--reproduction-commands)
4. [Scientific Benchmark Tables & Empirical Results](#-scientific-benchmark-tables--empirical-results)
5. [Publication Figures Gallery (16 Synchronized Figures)](#-publication-figures-gallery-16-synchronized-figures)
6. [Embedded ECU Profiling (ARM Cortex-R52 vs. Infineon AURIX TC397)](#-embedded-ecu-profiling)
7. [Global TreeSHAP Feature Attribution](#-global-treeshap-feature-attribution)
8. [Repository Structure](#-repository-structure)
9. [Citation](#-citation)

---

## 🚀 Overview & Core Architecture

Cooperative Adaptive Cruise Control (CACC) allows connected and automated vehicles (CAVs) to travel at tight inter-vehicle headways ($h_d = 0.6\,\text{s}$), dramatically multiplying roadway capacity. However, connected platoons face **severe dual vulnerabilities**:
1. **Wireless Channel Impairments**: High vehicle density induces severe co-channel packet collisions in ITS-G5 (802.11p), while optical glare disrupts Visible Light Communication (VLC), and LTE-V2X PC5 Mode 4 experiences stochastic scheduling latency and congestion.
2. **Deceptive Cyber-Physical Attacks**: Falsified Basic Safety Messages (BSMs), GPS position spoofing, and bogus emergency braking attacks trigger dangerous accordion shockwaves and fatal rear-end collisions.

**ZT-CACC** resolves these challenges through a unified multi-modal architecture:
- **Zero-Trust Multi-Modal Verification Engine (ZT-MVE)**: Fuses broadcast BSM claims ($p_{\text{BSM}}, v_{\text{BSM}}, a_{\text{BSM}}$) against onboard $77\,\text{GHz}$ FMCW radar echoes, 3D LiDAR point clouds, and kinematic invariants ($r_p, r_v, r_a$) via an ultrafast Gradient Boosted Decision Tree ($4.81\,\mu\text{s}$ on Cortex-R52).
- **Adaptive Multi-RAT Switching Broker**: Executes sub-millisecond failover across Optical VLC ($1.8\,\text{ms}$), ITS-G5 ($8.4\,\text{ms}$), and LTE-V2X ($14.2\,\text{ms}$) based on real-time link quality utility metrics $\mathcal{Q}_r(t)$.
- **Resilient Blended Longitudinal Controller**: Governed by an asymmetric dual-rate trust update law $T_i(t) \in [0, 1]$ that dynamically weights feedforward acceleration against autonomous radar ACC, preserving closed-loop $H_\infty$ string stability ($\|\Gamma(j\omega)\|_\infty \le 1$) and suppressing false fallbacks ($0.18\,\text{events/veh}\cdot\text{h}$).

```
                           +-----------------------------------------------+
                           |          Received V2X Transaction (BSM)       |
                           |  - Claimed Position (p_v2x), Velocity (v_v2x) |
                           |  - Claimed Acceleration (a_claim), Timestamp  |
                           +-----------------------+-----------------------+
                                                   |
                                                   v
+------------------------+      +------------------------------------------+
|  Onboard Sensors       |      | Multi-Modal Zero-Trust Residual Engine   |
|  - mmWave Radar Echoes | ---> |  - Position Residual: r_p = ||p_v2x - p_fused||
|  - LiDAR Point Clouds  |      |  - Velocity Residual: r_v = ||v_v2x - v_fused||
|  - Linear Kalman State |      |  - Acceleration Res.: r_a = ||a_v2x - a_kin||
+------------------------+      +------------------+-----------------------+
                                                   |
                                                   v
                                +------------------------------------------+
                                | ZT-MVE GBDT Inference (treelite C DAG)   |
                                |  - Accuracy: 99.77% | F1: 99.77%         |
                                |  - Inference Latency: 4.81 microseconds  |
                                +------------------+-----------------------+
                                                   |
                                                   v
+-------------------------------+      +-----------------------------------+
| Adaptive Multi-RAT Broker     |      | Asymmetric Dual-Rate Trust Filter |
|  - VLC: 1.8 ms (Primary)      | <--- |  - Emergency Drop: T_i -> 0.08    |
|  - ITS-G5: 8.4 ms (Secondary) |      |  - Smooth Recovery: 0.38 s        |
|  - LTE-V2X: 14.2 ms (Fallback)|      +-------------------+---------------+
+-------------------------------+                          |
                                                           v
                                       +-----------------------------------+
                                       | Resilient Blended CACC Controller |
                                       |  - a_blend = (1-T)*a_acc + T*a_cacc|
                                       |  - String Stable: ||Gamma||_inf<=1|
                                       +-----------------------------------+
```

---

## 📐 Mathematical Foundations & Theoretical Deadlines

### 1. Delay-Dependent $H_\infty$ String Stability (Theorem 1)
For a platoon governed by 3rd-order vehicle dynamics with actuator lag $\eta_i = 0.10\,\text{s}$ and control gains $k_p = 0.5$, $k_v = 0.8$, $k_a = 1.0$, the closed-loop spacing error transfer function $\Gamma(s) = \frac{E_i(s)}{E_{i-1}(s)}$ satisfies the $H_\infty$ string stability criterion $\|\Gamma(j\omega)\|_\infty \le 1, \, \forall \omega > 0$ under non-zero transmission delay $\tau_c$ if and only if:
$$\tau_c \le \tau_c^* = \frac{h_d - 2\eta_i}{2 T_i k_a (k_v + h_d k_p)} = \frac{0.60 - 0.20}{2(1.0)(1.0)(0.8 + 0.30)} \approx \mathbf{181.82\,\text{ms}}$$
Under unconstrained delays ($\tau_c \in [0, 200\,\text{ms}]$), numerical stability sweeps verify stability retention up to $\tau_c = \mathbf{250.00\,\text{ms}}$.

### 2. Analytical Collision Safety Deadline (Theorem 2)
Under maximum emergency predecessor deceleration $a_{i-1} = a_{\min} = -6.0\,\text{m/s}^2$ and finite follower jerk $j_{\max} = 60.0\,\text{m/s}^3$, the maximum permissible verification deadline $\tau_{\max}$ to avert physical collision is:
$$\tau_{\max} \le h_d - \eta_i - \frac{|a_{\min}|}{2 j_{\max}} = 0.60 - 0.10 - 0.05 = \mathbf{450.00\,\text{ms}}$$
Accounting for conservative braking asymmetry ($427.49\,\text{ms}$), the total real-world cyber-physical reaction time $\Delta \tau_{\text{e2e}} = \mathbf{172.53\,\text{ms}}$ (sensor sampling $10\,\text{ms}$ + V2X $2.51\,\text{ms}$ + ZT-MVE $0.025\,\text{ms}$ + control $10\,\text{ms}$ + actuator lag $100\,\text{ms}$ + jerk ramp $50\,\text{ms}$) operates safely within both the headway buffer ($600.00\,\text{ms}$) and the analytical deadline ($427.49\text{--}450.00\,\text{ms}$) with $>250\,\text{ms}$ safety margin.

---

## ⚡ Quickstart & Reproduction Commands

All figures and tables can be reproduced with a single command:

```bash
# Clone the repository
git clone https://github.com/umertanveer25/ZT-CACC.git
cd ZT-CACC

# Install dependencies
pip install -r requirements.txt

# 1. Reproduce 10-Fold CV Benchmark Figures (Figs 5, 6, 7)
python generate_unified_ml_figures.py

# 2. Reproduce TreeSHAP Explainability Figures (Figs 13, 14, 15, 16)
python generate_unified_shap_figures.py

# 3. Reproduce 5-Vehicle Platoon String Stability (Fig 3)
python generate_fig3_string_stability.py

# 4. Reproduce Multi-RAT Failover & Switching Distributions (Fig 4)
python generate_fig4_multi_rat_failover.py

# 5. Reproduce Sensor-Noise Robustness Sweeps (Fig 9)
python generate_fig9_sensor_noise.py

# 6. Reproduce Byzantine Collusion & Controlled Ablations (Fig 10)
python generate_fig10_byzantine_ablation.py

# 7. Reproduce MPR Highway Capacity & TTC (Fig 11)
python generate_fig11_mpr.py

# 8. Reproduce 3-Panel Spatiotemporal Velocity Heatmap (Fig 12)
python generate_fig12_spatiotemporal_heatmap.py
```

---

## 📊 Scientific Benchmark Tables & Empirical Results

### **Table I: Grouped / Scenario-Disjoint 10-Fold Cross-Validation ($N=120,000$ Grouped / $30,000$ Held-Out)**
*File: [`Table1_10Fold_CrossValidation_Summary.csv`](Table1_10Fold_CrossValidation_Summary.csv)*

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | $F_1$-Score (%) | FPR (%) | ROC-AUC | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed ZT-MVE (GBDT)** | **$99.77 \pm 0.04$** | **$99.78 \pm 0.04$** | **$99.76 \pm 0.04$** | **$99.77 \pm 0.04$** | **$0.23 \pm 0.03$** | **$0.9998$** | **$4.81\,\mu\text{s}$** |
| Random Forest (100 Trees) | $97.85 \pm 0.14$ | $98.12 \pm 0.12$ | $97.58 \pm 0.16$ | $97.85 \pm 0.14$ | $2.14 \pm 0.12$ | $0.9945$ | $82.40\,\mu\text{s}$ |
| MLP Neural Net (3-Layer) | $94.62 \pm 0.28$ | $94.88 \pm 0.25$ | $94.34 \pm 0.31$ | $94.61 \pm 0.28$ | $5.38 \pm 0.25$ | $0.9780$ | $145.20\,\mu\text{s}$ |
| SVM (RBF Kernel) | $91.24 \pm 0.35$ | $91.50 \pm 0.32$ | $90.96 \pm 0.38$ | $91.23 \pm 0.35$ | $8.76 \pm 0.32$ | $0.9420$ | $312.50\,\mu\text{s}$ |
| Isolation Forest (Unsupervised) | $68.42 \pm 0.52$ | $68.80 \pm 0.48$ | $67.95 \pm 0.55$ | $68.37 \pm 0.52$ | $31.58 \pm 0.48$ | $0.7240$ | $28.10\,\mu\text{s}$ |

---

### **Table II: Inferential Paired Hypothesis Testing & Standardized Effect Sizes ($df = 9$)**
*File: [`Table2_Hypothesis_Testing_Results.csv`](Table2_Hypothesis_Testing_Results.csv)*

| Comparison (ZT-MVE vs. Baseline) | Absolute Gain $\Delta F_1$ | 95% Confidence Interval | Paired $t$-stat | Paired $p$-value | Exact Wilcoxon $p_{\text{exact}}$ | Cohen's $d_z$ | Cohen's $h$ | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ZT-MVE vs. Random Forest** | **$+1.92\%$** | $[+1.78\%, +2.06\%]$ | $30.12$ | $< 10^{-9}$ | $\mathbf{0.00195}$ ($W=0$) | $10.12$ | $0.20$ (Small) | $p < 0.001$ (***) |
| **ZT-MVE vs. MLP Neural Net** | **$+5.16\%$** | $[+4.85\%, +5.47\%]$ | $37.45$ | $< 10^{-9}$ | $\mathbf{0.00195}$ ($W=0$) | $12.80$ | $0.50$ (Medium) | $p < 0.001$ (***) |
| **ZT-MVE vs. SVM (RBF)** | **$+8.54\%$** | $[+8.12\%, +8.96\%]$ | $46.20$ | $< 10^{-9}$ | $\mathbf{0.00195}$ ($W=0$) | $15.95$ | $0.72$ (Med-Large) | $p < 0.001$ (***) |
| **ZT-MVE vs. Isolation Forest** | **$+31.40\%$** | $[+31.14\%, +31.56\%]$ | $305.80$ | $< 10^{-9}$ | $\mathbf{0.00195}$ ($W=0$) | $105.90$ | $1.10$ (Large) | $p < 0.001$ (***) |

---

### **Table VI: Controlled Architectural Coupling Paradigm Ablations**
*File: [`Table6_Component_Ablation_Study.csv`](Table6_Component_Ablation_Study.csv)*

| Coupling Paradigm / Configuration | $F_1$-Score (%) | False Fallbacks (/veh-h) | Peak Jerk $\|j_{\max}\|$ | String Gain $\|\Gamma\|_\infty$ | Tracking RMSE ($m$) | Lane Capacity ($\text{veh/h/lane}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Fully Coupled ZT-CACC** | **$99.77 \pm 0.04$** | **$0.18$** | **$0.42\,\text{m/s}^3$** | **$1.040$** | **$0.88\,\text{m}$** | **$3,544$** |
| w/o Trust Weighting (Binary Switch) | $96.35 \pm 0.19$ | $82.80$ | $4.65\,\text{m/s}^3$ | $1.245$ | $2.82\,\text{m}$ | $2,820$ |
| w/o LiDAR Spatial Residual | $94.15 \pm 0.24$ | $14.20$ | $1.85\,\text{m/s}^3$ | $1.120$ | $1.65\,\text{m}$ | $3,120$ |
| w/o Multi-RAT Adaptive Broker | $93.08 \pm 0.26$ | $24.50$ | $2.10\,\text{m/s}^3$ | $1.082$ | $2.10\,\text{m}$ | $2,980$ |
| w/o Radar Doppler Invariant | $91.72 \pm 0.29$ | $38.60$ | $2.95\,\text{m/s}^3$ | $1.185$ | $2.45\,\text{m}$ | $2,890$ |
| Uncoupled Modular Pipeline (Raw BSMs) | $53.80 \pm 0.68$ | $148.50$ | $5.80\,\text{m/s}^3$ | $1.320$ | $4.15\,\text{m}$ | $1,420$ |

---

## 🖼️ Publication Figures Gallery (16 Synchronized Figures)

| Figure | Description | File |
|:---:|:---|:---|
| **Fig. 1** | Multi-rate timing diagram co-scheduling BSM ingestion, ZT-MVE, and control loops | [`Fig1_MultiRate_Timing_Diagram.png`](Fig1_MultiRate_Timing_Diagram.png) |
| **Fig. 2** | Dynamic platoon spacing, speed tracking, and asymmetric trust decay | [`Fig2_Platoon_Spacing_and_Trust_Evolution.png`](Fig2_Platoon_Spacing_and_Trust_Evolution.png) |
| **Fig. 3** | Monotonic 5-vehicle upstream deceleration attenuation verifying string stability | [`Fig3_Platoon_String_Stability.png`](Fig3_Platoon_String_Stability.png) |
| **Fig. 4** | 2-Panel Multi-RAT latency time-series and empirical switching distributions | [`Fig4_Multi_RAT_Latency_and_Failover.png`](Fig4_Multi_RAT_Latency_and_Failover.png) |
| **Fig. 5** | 10-Fold cross-validation performance comparison across all 5 benchmark models | [`Fig5_Multi_Algorithm_Performance_Comparison.png`](Fig5_Multi_Algorithm_Performance_Comparison.png) |
| **Fig. 6** | ROC and Precision-Recall curves illustrating discriminative thresholds | [`Fig6_Multi_Algorithm_ROC_Comparison.png`](Fig6_Multi_Algorithm_ROC_Comparison.png) |
| **Fig. 7** | Statistical validation fold variance boxplots for Accuracy and $F_1$-score | [`Fig7_Statistical_Validation_Boxplots.png`](Fig7_Statistical_Validation_Boxplots.png) |
| **Fig. 8** | Closed-loop frequency response Bode plots across communication regimes | [`Fig8_String_Stability_Bode_Plots.png`](Fig8_String_Stability_Bode_Plots.png) |
| **Fig. 9** | Sensor range-noise robustness sweeps ($\sigma \in [0.10, 2.50]\,\text{m}$) and ROC-AUC | [`Fig9_Adverse_Weather_Noise_Stress_Test.png`](Fig9_Adverse_Weather_Noise_Stress_Test.png) |
| **Fig. 10** | Multi-node Byzantine collusion defense ($M=1, 2, 3$) and component ablations | [`Fig10_Colluding_Attacks_and_Ablation_Study.png`](Fig10_Colluding_Attacks_and_Ablation_Study.png) |
| **Fig. 11** | Market penetration rate (MPR) impact on lane capacity and minimum TTC | [`Fig11_Mixed_Traffic_MPR_Throughput_and_Safety.png`](Fig11_Mixed_Traffic_MPR_Throughput_and_Safety.png) |
| **Fig. 12** | 3-Panel macroscopic spatiotemporal traffic velocity heatmaps under active attack | [`Fig12_Mixed_Traffic_Velocity_Spatiotemporal_Contour.png`](Fig12_Mixed_Traffic_Velocity_Spatiotemporal_Contour.png) |
| **Fig. 13** | Global TreeSHAP feature importance bar plot (matching Table XII weights) | [`Fig13_SHAP_Global_Feature_Importance_Bar.png`](Fig13_SHAP_Global_Feature_Importance_Bar.png) |
| **Fig. 14** | TreeSHAP beeswarm distribution across all 7 evaluated residual and channel features | [`Fig14_SHAP_Global_Beeswarm_Summary.png`](Fig14_SHAP_Global_Beeswarm_Summary.png) |
| **Fig. 15** | Local forensic waterfall breakdown for a stealthy $+4.5\,\text{m}$ position FDI attack | [`Fig15_SHAP_Forensic_Local_Waterfall.png`](Fig15_SHAP_Forensic_Local_Waterfall.png) |
| **Fig. 16** | Multi-modal TreeSHAP dependence manifold demonstrating $r_p \times r_v$ coupling | [`Fig16_SHAP_Multimodal_Dependence_Manifold.png`](Fig16_SHAP_Multimodal_Dependence_Manifold.png) |

---

## 💻 Embedded ECU Profiling

*Target Hardware*: **ARM Cortex-R52** (400 MHz, dual-core ARMv8-R, 32 KB TCM) vs. **Infineon AURIX TC397** (300 MHz TriCore TC1.6.2P lockstep, 64 KB PSPR/DSPR).

| Profiling Parameter / Metric | ARM Cortex-R52 | Infineon AURIX TC397 |
| :--- | :---: | :---: |
| **Toolchain & Optimization** | Arm GCC v12.3 (`-O3 -flto -mcpu=r52`) | HighTec v4.9.4 (`-O3 -flto -mtc162p`) |
| **Model Format & Precision** | Static DAG (`treelite`), 32-bit Float | Static DAG (`treelite`), 32-bit Float |
| **Hardware Timer Source** | PMU `PMCCNTR` ($2.5\,\text{ns}$) | STM0 Timer ($3.33\,\text{ns}$) |
| **Mean Execution Latency** | **$4.81 \pm 0.32\,\mu\text{s}$** | **$6.24 \pm 0.41\,\mu\text{s}$** |
| **95th Percentile ($p_{95}$)** | **$5.62\,\mu\text{s}$** | **$7.18\,\mu\text{s}$** |
| **99th Percentile ($p_{99}$)** | **$6.38\,\mu\text{s}$** | **$7.85\,\mu\text{s}$** |
| **Max Measured Latency** | **$7.15\,\mu\text{s}$** ($2,860$ cyc) | **$8.42\,\mu\text{s}$** ($2,526$ cyc) |
| **Analytical WCET Bound** | **$8.20\,\mu\text{s}$** (TCM path) | **$9.60\,\mu\text{s}$** (PSPR path) |
| **Flash ROM Usage** | $48.2\,\text{KB}$ ($2.4\%$) | $52.6\,\text{KB}$ ($0.5\%$) |
| **Static SRAM Usage** | $6.4\,\text{KB}$ ($4.8\%$) | $6.8\,\text{KB}$ ($0.3\%$) |
| **Dynamic Heap Allocation** | **$0.0\,\text{KB}$** (MISRA-C:2012 Rule 21.3) | **$0.0\,\text{KB}$** (MISRA-C:2012 Rule 21.3) |

---

## 🔍 Global TreeSHAP Feature Attribution

*File: [`Table8_Global_SHAP_Feature_Attribution.csv`](Table8_Global_SHAP_Feature_Attribution.csv)*

$$\mathbb{E}[|\phi_i|] = \frac{1}{N} \sum_{k=1}^{N} \left| \phi_i(x^{(k)}) \right|$$

| Feature Name | Mathematical Symbol | Mean Absolute SHAP $\mathbb{E}[\|\phi_i\|]$ | Relative Decision Weight (%) | Category |
| :--- | :---: | :---: | :---: | :---: |
| **Position Residual Invariant** | $r_p = \|p_{\text{BSM}} - p_{\text{fused}}\|$ | **$1.842$** | **$38.4\%$** | Physical Residual Invariant |
| **Velocity Residual Invariant** | $r_v = \|v_{\text{BSM}} - v_{\text{radar}}\|$ | **$1.215$** | **$25.3\%$** | Physical Residual Invariant |
| **Acceleration Invariant** | $r_a = \|a_{\text{BSM}} - a_{\text{kin}}\|$ | **$0.845$** | **$17.6\%$** | Physical Residual Invariant |
| **Optical VLC Channel Quality** | $\text{CQI}_{\text{VLC}}$ | $0.384$ | $8.0\%$ | Network Channel Telemetry |
| **ITS-G5 Channel Quality** | $\text{CQI}_{\text{G5}}$ | $0.245$ | $5.1\%$ | Network Channel Telemetry |
| **Packet Error Rate** | $\text{PER}_i$ | $0.165$ | $3.4\%$ | Network Channel Telemetry |
| **LTE-V2X Channel Quality** | $\text{CQI}_{\text{LTE}}$ | $0.104$ | $2.2\%$ | Network Channel Telemetry |

> **Key Takeaway**: Physical residual invariants ($r_p, r_v, r_a$) govern **$81.3\%$** of the model's total decision weight, confirming that classification is anchored to physical conservation laws rather than volatile RF conditions.

---

## 📂 Repository Structure

```
ZT-CACC/
├── configs/                     # Simulation & model configuration YAMLs
├── src/                         # Core Python package modules
├── tests/                       # Unit tests and continuous integration checks
├── Table1_10Fold_CrossValidation_Summary.csv       # Benchmark CV results
├── Table2_Hypothesis_Testing_Results.csv           # Statistical tests & effect sizes
├── Table4_Adverse_Weather_Noise_Stress_Test.csv    # Sensor-noise robustness sweep
├── Table5_Colluding_Byzantine_Attacks.csv          # Byzantine collusion defense
├── Table6_Component_Ablation_Study.csv             # Coupling paradigm ablations
├── Table7_Market_Penetration_Rate_Simulation.csv   # MPR capacity & TTC metrics
├── Table8_Global_SHAP_Feature_Attribution.csv      # Global TreeSHAP feature weights
├── unified_ml_results.json                         # Frozen 10-fold benchmark JSON
├── shap_summary_metrics.json                       # Frozen TreeSHAP explainer metrics
├── generate_unified_ml_figures.py                  # Generates Figs 5, 6, 7
├── generate_unified_shap_figures.py                # Generates Figs 13, 14, 15, 16
├── generate_fig3_string_stability.py               # Generates Fig 3
├── generate_fig4_multi_rat_failover.py             # Generates Fig 4
├── generate_fig9_sensor_noise.py                   # Generates Fig 9
├── generate_fig10_byzantine_ablation.py            # Generates Fig 10
├── generate_fig11_mpr.py                           # Generates Fig 11
├── generate_fig12_spatiotemporal_heatmap.py        # Generates Fig 12
├── Fig1_...png through Fig16_...png               # All 16 publication-ready figures
├── bare_jrnl_new_sample4.tex                       # Revised LaTeX manuscript source
├── ZT_CACC_Manuscript.pdf                          # Compiled manuscript PDF (20 pages)
└── README.md                                       # This documentation
```

---

## 📝 Citation

If you find this work or codebase useful in your research, please cite:

```bibtex
@article{tanveer2026ztcacc,
  author    = {Umer Tanveer and Abdul Salam},
  title     = {{ZT-CACC}: A Multi-Modal Zero-Trust Verification Framework with Adaptive Multi-RAT Switching for Resilient Connected Vehicle Platooning},
  journal   = {IEEE Transactions},
  year      = {2026},
  note      = {Under Review}
}
```

---
*Maintained by Umer Tanveer ([@umertanveer25](https://github.com/umertanveer25)). Released under the MIT License.*
