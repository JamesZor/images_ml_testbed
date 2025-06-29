# images_ml_testbed

## Simple models experiment

A simple CNN, MLR model and Deeper CNN.

configs: 
    - experiments: mnist_simple_models ,
    - models : mnist_cnn, mnist_mlr, simple_cnn 
    - data : mnist_float32 
    - training: mnist_basic 


    

### Simple CNN 

```bash
# run simple CNN model 
python train.py experiments=mnist_simple_models experiment.name='test_simple_cnn'
```

#### Model Architecture Summary

| Metric | Value |
|--------|-------|
| Total params | 235,146 |
| Trainable params | 235,146 |
| Non-trainable params | 0 |
| Total mult-adds (MB) | 0.24 |
| Input size (MB) | 0.00 |
| Forward/backward pass size (MB) | 0.00 |
| Params size (MB) | 0.94 |
| Estimated Total Size (MB) | 0.95 |

#### Layer Details

| Layer (type:depth-idx) | Input Shape | Output Shape | Param # |
|------------------------|-------------|--------------|---------|
| MNIST_CNN | [1, 28, 28] | [1, 10] | -- |
| ├─Sequential: 1-1 | [1, 28, 28] | [1, 10] | -- |
| │    └─Sequential: 2-1 | [1, 28, 28] | [1, 256] | -- |
| │    │    └─Flatten: 3-1 | [1, 28, 28] | [1, 784] | -- |
| │    │    └─Linear: 3-2 | [1, 784] | [1, 256] | 200,960 |
| │    │    └─ReLU: 3-3 | [1, 256] | [1, 256] | -- |
| │    │    └─Dropout: 3-4 | [1, 256] | [1, 256] | -- |
| │    └─Sequential: 2-2 | [1, 256] | [1, 128] | -- |
| │    │    └─Linear: 3-5 | [1, 256] | [1, 128] | 32,896 |
| │    │    └─ReLU: 3-6 | [1, 128] | [1, 128] | -- |
| │    │    └─Dropout: 3-7 | [1, 128] | [1, 128] | -- |
| │    └─Linear: 2-3 | [1, 128] | [1, 10] | 1,290 |

#### Test Results

| Test Metric | Value |
|-------------|-------|
| test_acc | 0.9721 |
| test_f1 | 0.9713 |
| test_loss | 0.0949 |

![CNN Prediction Visualization](https://github.com/JamesZor/images_ml_testbed/blob/main/results/mnist_cnn/2025-06-28_11-58/prediction_visualization.png)

### MLR (Multi Logistic Regression)

```bash
# Run simple Multi Logistic regression 
python train.py experiments=mnist_simple_models model=mnist_mlr experiment.name='test_mlr'
```

#### Model Architecture Summary

| Metric | Value |
|--------|-------|
| Total params | 7,850 |
| Trainable params | 7,850 |
| Non-trainable params | 0 |
| Total mult-adds (MB) | 0.01 |
| Input size (MB) | 0.00 |
| Forward/backward pass size (MB) | 0.00 |
| Params size (MB) | 0.03 |
| Estimated Total Size (MB) | 0.03 |

#### Layer Details

| Layer (type:depth-idx) | Input Shape | Output Shape | Param # |
|------------------------|-------------|--------------|---------|
| MNIST_MLR | [1, 28, 28] | [1, 10] | -- |
| ├─Sequential: 1-1 | [1, 28, 28] | [1, 10] | -- |
| │    └─Flatten: 2-1 | [1, 28, 28] | [1, 784] | -- |
| │    └─Linear: 2-2 | [1, 784] | [1, 10] | 7,850 |

#### Test Results

| Test Metric | Value |
|-------------|-------|
| test_acc | 0.9207 |
| test_f1 | 0.9179 |
| test_loss | 0.2789 |

![MLR Prediction Visualization](https://github.com/JamesZor/images_ml_testbed/blob/main/results/mnist_mlr/2025-06-28_12-04/prediction_visualization.png)

### Deeper CNN 

```bash
# Run SimpleCNN model 
python train.py experiments=mnist_simple_models model=cnn_simple experiment.name='test_cnn_model_v2' data.input_size=[1,1,28,28]
```

#### Model Architecture Summary

| Metric | Value |
|--------|-------|
| Total params | 421,642 |
| Trainable params | 421,642 |
| Non-trainable params | 0 |
| Total mult-adds (MB) | 4.28 |
| Input size (MB) | 0.00 |
| Forward/backward pass size (MB) | 0.30 |
| Params size (MB) | 1.69 |
| Estimated Total Size (MB) | 1.99 |

#### Layer Details

| Layer (type:depth-idx) | Input Shape | Output Shape | Param # |
|------------------------|-------------|--------------|---------|
| SimpleCNN | [1, 1, 28, 28] | [1, 10] | -- |
| ├─Sequential: 1-1 | [1, 1, 28, 28] | [1, 64, 7, 7] | -- |
| │    └─Conv2d: 2-1 | [1, 1, 28, 28] | [1, 32, 28, 28] | 320 |
| │    └─ReLU: 2-2 | [1, 32, 28, 28] | [1, 32, 28, 28] | -- |
| │    └─MaxPool2d: 2-3 | [1, 32, 28, 28] | [1, 32, 14, 14] | -- |
| │    └─Conv2d: 2-4 | [1, 32, 14, 14] | [1, 64, 14, 14] | 18,496 |
| │    └─ReLU: 2-5 | [1, 64, 14, 14] | [1, 64, 14, 14] | -- |
| │    └─MaxPool2d: 2-6 | [1, 64, 14, 14] | [1, 64, 7, 7] | -- |
| ├─Sequential: 1-2 | [1, 64, 7, 7] | [1, 10] | -- |
| │    └─Flatten: 2-7 | [1, 64, 7, 7] | [1, 3136] | -- |
| │    └─Linear: 2-8 | [1, 3136] | [1, 128] | 401,536 |
| │    └─ReLU: 2-9 | [1, 128] | [1, 128] | -- |
| │    └─Dropout: 2-10 | [1, 128] | [1, 128] | -- |
| │    └─Linear: 2-11 | [1, 128] | [1, 10] | 1,290 |

#### Test Results

| Test Metric | Value |
|-------------|-------|
| test_acc | 0.9915 |
| test_f1 | 0.9913 |
| test_loss | 0.0245 |

![MLR Prediction Visualization](https://github.com/JamesZor/images_ml_testbed/blob/main/results/simple_cnn/2025-06-28_13-04/prediction_visualization.png)

### Combined Model Architecture Summary

| Metric | SimpleCNN | CNN (FC Network) | MLR |
|--------|-----------|------------------|-----|
| Total mult-adds (MB) | 4.28 | 0.24 | 0.01 |
| Forward/backward pass size (MB) | 0.30 | 0.00 | 0.00 |
| Params size (MB) | 1.69 | 0.94 | 0.03 |
| Estimated Total Size (MB) | 1.99 | 0.95 | 0.03 |

### Combined Test Results

| Test Metric | SimpleCNN | CNN (FC Network) | MLR |
|-------------|-----------|------------------|-----|
| test_acc | 0.9915 | 0.9721 | 0.9207 |
| test_f1 | 0.9913 | 0.9713 | 0.9179 |
| test_loss | 0.0245 | 0.0949 | 0.2789 |


## Running MNIST Evaluation 

```bash 
python mnist_evaluate.py --experiment_path experiments_mlr/2025-06-27_17-33/ --model_type mnist_mlr --num_samples 30 --layout 3x10
```


## CIFAR - 100 



Running the simple_cnn model ( Deeper CNN) model on CIFAR-100 ( 100 classes), we get the following results:

```bash 
 python train.py experiments=cifar100_test
```


| Test Metric | Value |
|-------------|-------|
| test_acc | 0.3126 |
| test_f1 | 0.2314 |
| test_loss | 2.7993 |


![SimpleCNN_cifar100](https://github.com/JamesZor/images_ml_testbed/blob/main/results/simple_cnn/2025-06-28_13-04/prediction_visualization.png)


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
