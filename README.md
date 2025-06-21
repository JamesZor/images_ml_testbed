# images_ml_testbed
# Data Type Performance Experiment

A comprehensive analysis of different tensor data types (float32, float16, int8) and their impact on model accuracy, performance, and memory efficiency in machine learning workflows.

## Overview

This experiment evaluates the trade-offs between numerical precision, memory usage, and computational performance across different tensor data types. The study compares float32 (baseline), float16 (half precision), and int8 (quantized) representations.

## Quick Start

Run the experiments with the following commands:

```bash
# Baseline experiment with float32
python train.py data.tensor_dtype="float32" experiment.name="accuracy_baseline"

# Half precision experiment with float16  
python train.py data.tensor_dtype="float16" experiment.name="accuracy_half"

# Quantized experiment with int8
python train.py data.tensor_dtype="int8" experiment.name="accuracy_quantized"
```

## Results Summary

### Performance Metrics Comparison

| Data Type | Test Accuracy | Test F1 Score | Test Loss | Runtime (s) |
|-----------|---------------|---------------|-----------|-------------|
| **float32** | 99.36% ± 0.00 | 99.29 ± 0.00 | 0.0216 ± 0.0004 | 93.5 ± 52.5 |
| **float16** | 99.30% | 99.24 | 0.0215 | 56.7 |
| **int8** | 98.80% | 98.69 | 0.0382 | 58.1 |

### Individual Experiment Results

| Experiment Name | Data Type | Accuracy | F1 Score | Loss | Runtime |
|----------------|-----------|----------|----------|------|---------|
| image_class_prediction | float32 | 99.36% | 99.29 | 0.021319 | 130.59s |
| accuracy_baseline | float32 | 99.36% | 99.30 | 0.021850 | 56.38s |
| accuracy_half | float16 | 99.30% | 99.24 | 0.021527 | 56.69s |
| accuracy_quantized | int8 | 98.80% | 98.69 | 0.038176 | 58.10s |

## Analysis

### 🎯 Accuracy Impact
- **Float16 vs Float32**: -0.06% accuracy loss (minimal impact)
- **Int8 vs Float32**: -0.56% accuracy loss (moderate impact)
- **Float32 Baseline**: 99.36% accuracy

### 💾 Memory Efficiency
- **Float32**: 100% memory usage (baseline)
- **Float16**: 50% memory usage (2x compression)
- **Int8**: 25% memory usage (4x compression)

### ⚡ Performance Analysis
- **Float16**: ~40% faster than float32 baseline
- **Int8**: ~38% faster than float32 baseline  
- **Float32**: Baseline performance

## Visualizations
