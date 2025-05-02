# End-to-End Optimization of Optical Communication Links with Deep Learning

## 🚀 Overview
This project demonstrates how **deep learning can optimize optical fiber communications** by learning **adaptive symbol constellations** that outperform traditional modulation schemes in nonlinear fiber channels. We developed an autoencoder system that jointly learns transmitter and receiver configurations specifically adapted to fiber channel impairments.

## 🔑 Key Contributions
- Successfully built an **autoencoder (AE) system** that learns **optimal constellations** for fiber optic channels
- Demonstrated that **learned constellations achieve better separation** than traditional schemes (PAM, QAM, QPSK) after channel distortion
- Developed **84 differentiable neural network models** of fiber channels using OptiSystem simulations
- Trained **70 AE configurations** across different channel conditions and symbol set sizes
- Achieved **100% symbol recovery accuracy** for 4-symbol constellations

## 🧠 Methodology

### Channel Modeling
1. Created datasets using **OptiSystem simulations** for 14 modulation schemes
2. Developed **6 neural network architectures** to model channel behavior
3. Evaluated models using **R² metric** to assess generalization capability

### Autoencoder Design
1. Jointly optimized **transmitter (constellation mapper)** and **receiver (demodulator)**
2. Integrated pre-trained channel models as **non-trainable components**
3. Used **cross-entropy loss** to maximize symbol recovery accuracy

## 📊 Results: Learned Constellations Outperform Traditional Schemes

### Key Findings
✅ **Learned constellations maintain better separation** after channel distortion compared to traditional schemes  
✅ **Noise-resilient channel models** produced the most robust constellations  
✅ **100% accuracy achieved** for 4-symbol constellations (vs. ~85% for QPSK)  
✅ Performance remains strong up to **16-symbol constellations** (85.6% accuracy)

### Constellation Comparison
| Scheme | Input Constellation | Output Constellation | Accuracy |
|--------|---------------------|----------------------|----------|
| Learned (AE) | ![Learned Input](figures/learned_input.png) | ![Learned Output](figures/learned_output.png) | **100%** |
| QPSK | ![QPSK Input](figures/qpsk_input.png) | ![QPSK Output](figures/qpsk_output.png) | ~85% |

### Top Performers by Symbol Set Size
| Symbol Count | Best Model | Accuracy |
|--------------|------------|----------|
| 4 | 256PSK Noise Resilient | **100%** |
| 8 | 8QAM Deeper | 94.8% |
| 16 | 16QAM Basic | 85.6% |

## 👥 Team
- **Thomas Haene** (Electrical Engineering) - Channel modeling & system integration
- **Alexandre Sleiman** (Electrical Engineering) - NN architectures & AE development  
- **Charlie Gil** (Software Engineering) - Autoencoder design & testing

**Supervisor:** Dr. Ioannis Psaromiligkos, McGill University

## 📜 License
Academic use permitted. Contact authors for commercial applications.
