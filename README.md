# d23it158_External_Practical_Exam

# Robustness & Adversarial Defenses on an Automated Cheque-Amount Digit Reader

This repository contains the official implementation, evaluation, and adversarial defense strategy for an AI-based automated cheque-clearing system as part of the **Responsible and Safe AI (AIML4001)** Practical Examination.

## 📌 Scenario & Objectives

A cooperative bank is introducing automated cheque clearing, using a computer-vision model (CNN) to read handwritten digits of cheque amounts and account numbers. However, the system is vulnerable to adversarial evasion attacks (such as **FGSM** and **PGD**), where tiny, almost invisible changes in pixels can force a digit `1` to be read as a `7` or `3` as `8`. 

This project implements a robust verification framework to evaluate the vulnerability of the baseline CNN, benchmark it against standard adversarial attacks, and harden the system using **adversarial fine-tuning**.

---

## 🛠️ Project Structure & Setup

### Requirements
- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Matplotlib

All tasks run deterministically on the **CPU** (using `torch.manual_seed(42)` and `pixel values inside [0, 1]`).

### Model Architecture
* **Convolutional Layers:** 2 layers (with MaxPool2d and ReLU activation)
* **Fully Connected Layers:** 2 layers (hidden size 128, final projection 10)
* **Optimizer:** Adam (Learning Rate = 0.001)

---

## 📊 Experimental Results

### Part A: Baseline Model under Attack

The baseline model achieves an excellent clean test accuracy of **98.5%**. However, it is highly sensitive to adversarial perturbations as the perturbation radius $\epsilon$ increases:

| Epsilon ($\epsilon$) | FGSM Accuracy (%) | PGD Accuracy (%) |
|---------------------|-------------------|------------------|
| **0.00**            | 98.5%             | 98.5%            |
| **0.05**            | 94.0%             | 92.7%            |
| **0.10**            | 81.9%             | 69.8%            |
| **0.20**            | 28.0%             | 0.8%             |
| **0.30**            | 2.2%              | 0.0%             |

* **Attack Success Rate ($\epsilon = 0.20$):** **72.0%** for FGSM and **99.2%** for PGD.

---

### Part B: Adversarial Training (Defense)

To secure the system, we fine-tuned the model for **2 epochs** on a **50:50 mixed batch** of clean and FGSM-perturbed images ($\epsilon = 0.20$).

| Model Model | Clean Accuracy | FGSM ($\epsilon=0.20$) | PGD ($\epsilon=0.20$) |
| :--- | :--- | :--- | :--- |
| **Baseline Model** | 98.5% | 28.0% | 0.8% |
| **Adversarially Trained** | **98.8%** | **89.8%** | **20.2%** |

* **Clean Accuracy Change:** `+0.4%` (robust model is highly generalized)
* **FGSM Robustness Gain:** `+61.8%` 
* **PGD Robustness Gain:** `+19.4%` 

---

## 📈 Visualizations

### 1. Clean vs. Adversarial Digits ($\epsilon = 0.20$)
Below are 5 clean digits alongside their perturbed adversarial counterparts demonstrating misread classifications:

![Clean vs Adversarial](clean_vs_adversarial.png)

### 2. Accuracy vs. Epsilon
Evaluating performance degradation under attack:

![Accuracy vs Epsilon](accuracy_vs_epsilon.png)

---

## 📝 Technical Answers

* **Q1. Sensitivity to Small Perturbations:** Under $\epsilon = 0.20$, accuracy falls below 50%. A tiny change can shift the image representation across highly non-linear, high-dimensional neural network decision boundaries.
* **Q2. FGSM vs. PGD:** PGD is a significantly stronger attack because it iteratively takes small gradient-directed steps and projects the result back into the $\epsilon$-ball constraint, avoiding the linear approximations that make FGSM weaker.
* **Q3. Mis-read Digit Behavior:** Digits like `2` were misclassified as `1`, and `4` as `9`. Adversarial training dramatically boosted defense accuracy (FGSM robustness climbed to **89.8%**).
* **Q4. Epsilon Definition & Recommendations:** $\epsilon$ represents the max $L_\infty$-norm perturbation allowed per pixel to maintain visual imperceptibility. Recommended defenses include **Input Transformation** (JPEG compression or smoothing) and **Randomized Smoothing** to build guaranteed security margins.
