# Assembly Planning with the Burr Puzzle  
*A Mini-Research Project for CSE 691*

> **Authors:** Max Gao · Anjali Kaushik  
> **Institution:** Arizona State University

---

## ✨ Project Overview
This repository contains the simulation code, heuristic search algorithms, and evaluation scripts used in our mini-research project **“Assembly Planning with the Burr Puzzle.”**  
The goal is to solve a 6-piece Burr puzzle under realistic physics constraints using an **offline reinforcement-learning–inspired look-ahead search** that combines:

The framework naturally supports **single-arm, dual-arm, or unlimited-arm** virtual manipulators and records assembly plans that can be replayed or exported.

---

## 🎯 Key Features
| Feature | Description |
|---------|-------------|
| **Physics-aware Motion Feasibility** | Collision detection + gravity support checks using *trimesh* |
| **Configurable Look-ahead** | `top_k`, `rollout_depth` parameters with parallel execution |
| **Multi-Arm Constraints** | Limit #pieces without bottom support to model 1-arm / 2-arm robots |
| **Assembly-by-Disassembly Mode** | Automatically searches from goal → start, then reverses plan |
| **Headless or GUI Simulation** | Off-screen rendering for fast evaluation, optional OpenGL viewer |
| **CSV / GIF Logging** | Saves per-step states, tracks move cost, and generates animated assembly sequence |
