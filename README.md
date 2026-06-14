# CS7641 Optimization & Learning Unified Notebook

## Overview
This notebook implements and evaluates optimization techniques for training neural networks on tabular datasets. It covers three main parts: randomized optimization methods on the last layers, optimizer ablation studies on full models, and regularization techniques using Adam optimizers.

## Features
- Randomized Optimization (RHC, SA, GA) applied to last layers (≤ ~50k parameters)  
- Optimizer ablations comparing SGD, Momentum, Nesterov, Adam variants, AdamW on full models  
- Regularization experiments including L2, Early Stopping, Dropout, Label Smoothing, and Input Noise  
- Works with two tabular datasets: Hotel Bookings (classification) and US Accidents (regression)  
- Modular and reproducible setup using PyTorch, sklearn preprocessing pipelines

## Tech Stack
- Python 3  
- Jupyter Notebook / Google Colab  
- PyTorch (torch, torch.nn, torch.optim)  
- Scikit-learn (model_selection, preprocessing, pipeline)  
- pandas, numpy

## How to Use
1. Place the datasets `hotel_bookings.csv` and `US_Accidents_March23.csv` in the `./data` folder relative to the notebook.  
2. Open the notebook `cs7641-optimization-learning-unified.ipynb` in Jupyter or Colab.  
3. Run all cells sequentially to reproduce all experiments and results.  
4. Modify parameters or datasets as needed for further experimentation.

## Status
This notebook is organized and cleaned for professional GitHub presentation, ensuring reproducibility and clarity of all included workflows.