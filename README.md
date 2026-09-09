# WM-811k Wafer Map Defect Pattern Recognition

An industrial-grade Deep Learning system for Semiconductor Wafer Defect Pattern Classification using the WM-811K dataset.

## Project Highlights
- **Exploratory Data Analysis (EDA):** Deep dive into wafer spatial distributions and defect characteristics.
- **Handling Extreme Class Imbalance:** Geometric data augmentation and specialized loss functions (Focal Loss).
- **Explainable AI (XAI):** Visualizing defect regions using Grad-CAM to assist manufacturing engineers.
- **Production-Ready Inference:** Model optimization and benchmarking with ONNX Runtime.

## Project Structure
- \data/\: Raw and processed wafer maps (ignored in git)
- \
otebooks/\: Step-by-step Jupyter notebooks for EDA, experiments, and model training
- \src/\: Modularized Python source code for data loading, training, and inference
- \models/\: Trained model weights and exported ONNX models
