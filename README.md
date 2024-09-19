
# DDRL: A Diffusion-Driven Reinforcement Learning Approach for Enhanced TSP Solutions

**A novel approach integrating diffusion models and reinforcement learning to solve the Traveling Salesman Problem (TSP) with improved scalability and stability.**

## Abstract

The **Traveling Salesman Problem (TSP)** is a fundamental challenge in combinatorial optimization, known for its NP-hard complexity. While **Reinforcement Learning (RL)** has shown effectiveness in managing larger and more complex TSP instances, it faces challenges such as training instability and the need for substantial training resources.

On the other hand, **Diffusion models**—which iteratively refine noisy inputs to generate high-quality solutions—offer scalability and exploration capabilities for TSP, but they may struggle with optimality in complex cases and require large, resource-intensive training datasets.

To address these limitations, we propose **DDRL (Diffusion-Driven Reinforcement Learning)**, which integrates diffusion models with Reinforcement Learning. DDRL leverages a latent variable to generate an adjacency matrix, effectively merging image and graph learning within a unified RL framework. By utilizing a pre-trained diffusion model as a prior, DDRL demonstrates strong scalability and improved convergence stability.

Our approach aligns the training of DDRL with the **diffusion policy gradient**, providing theoretical evidence of its effectiveness in solving TSP. Moreover, we introduce novel **constraint datasets**, including obstacle and path constraints, to assess the generalization capabilities of DDRL. Through extensive validation, DDRL is shown to outperform existing methods in both basic and constrained TSP problems.
