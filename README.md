# 46750 - Optimization in Modern Power Systems - Assignment 2
Group 8: Peter Bach Morup, Louis Singh, Hubert Przeor


## Overview
Optimization-based tool for Network Expansion planning prepared for Group 7.

This repository contains MILP models for planning radial distribution networks. The core solver functions are in solver.py and support:

Model 1 – Single-Period

Deterministic expansion model that selects substation activations, node assignments, line usage, reinforcements, and flows under a yearly budget. Objective: minimize discounted system cost.

Model 2 – Multi-Year

Intertemporal model with monotone investments, yearly budgets, changing demands, connection/disconnection costs, and discounted costs over time.

Model 3 – Stochastic (Scenario-Based)

Multi-year model run across demand scenarios. Investments are shared across scenarios, while operations vary per scenario. Objective minimizes expected discounted cost including load shedding penalties.

# Notebooks

Model_1.ipynb – run single-period model

Model_2.ipynb – run multi-year model

Model_3.ipynb – scenario-based extension

Monte_carlo.ipynb – runs a Monte Carlo simulation of future demand probabilities

# Data Structures

classes.py defines the DistributionNetwork class used by all models.