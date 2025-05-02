End-to-End Optimization of Optical Communication Links with Deep Learning
Overview
This project explores the application of deep learning to optimize symbol constellations in optical fiber communication systems. Traditional modulation schemes (PAM, QAM, QPSK) may not be optimal for nonlinear fiber channels. We developed a deep learning-based autoencoder system that jointly learns transmitter and receiver configurations to maximize symbol recovery accuracy.

Key Contributions
Developed 84 differentiable neural network models of fiber optic channels using OptiSystem simulation data across 14 modulation schemes

Designed and trained 70 autoencoder systems to learn optimal symbol constellations adapted to channel impairments

Demonstrated that learned constellations outperform traditional schemes on simulated channels

Achieved 100% accuracy for 4-symbol constellations using noise-resilient channel models

Methodology
Channel Modeling:

Created datasets using OptiSystem simulations for 14 modulation schemes

Developed 6 neural network architectures to model channel behavior

Evaluated models using R² metric to assess generalization capability

Autoencoder Design:

Jointly optimized transmitter (constellation mapper) and receiver (demodulator)

Integrated pre-trained channel models as non-trainable components

Used cross-entropy loss to maximize symbol recovery accuracy

Results
Noise-resilient channel models showed best performance for autoencoder training

Learned constellations achieved better separation than traditional schemes

Performance decreased with increasing symbol set size, as expected

Top configurations:

4 symbols: 100% accuracy (256PSK Noise Resilient model)

8 symbols: 94.8% accuracy (8QAM Deeper model)

16 symbols: 85.6% accuracy (16QAM Basic model)

Future Work
Develop more comprehensive channel models for arbitrary constellations

Investigate techniques to improve generalization across modulation schemes

Explore hardware implementation and real-world validation

Team Members
Thomas Haene (Electrical Engineering)

Alexandre Sleiman (Electrical Engineering)

Charlie Gil (Software Engineering)

Supervised by Dr. Ioannis Psaromiligkos, McGill University

License
This project is available for academic and research purposes. Please contact the authors for commercial use inquiries.
