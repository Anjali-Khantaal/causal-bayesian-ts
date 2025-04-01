# Causal-Bayesian-TS

## Bayesian Time Series Modeling with Causal Interventions

This repository contains code and resources for a project exploring the integration of Bayesian time series forecasting with causal inference. Specifically, we investigate the effects of interventions (e.g., monetary policy changes) on key economic variables using Bayesian models and causal graphs.

### Project Overview

This project aims to advance time series analysis by integrating **Bayesian modeling** with **causal inference**. The primary focus is to forecast economic indicators and assess the effect of policy interventions using causal reasoning.

We explore two modeling paradigms:
- **Bayesian VAR Model**: Captures dynamic interdependencies between economic variables.
- **Bayesian RNN**: Suitable for modeling complex, nonlinear temporal relationships.

Both are combined with a **Directed Acyclic Graph (DAG)** to model causal effects (e.g., `interest_rate → asset_price`).

Interventions are simulated using **Pearl’s do-operator**, and uncertainty is quantified using posterior predictive intervals.

---

### Methodology

_Still under review!_

---

### Project Structure

_Under Development_

---

### High Level To-Do List

- [ ] Set up and preprocess real-world financial datasets (e.g., FRED)
- [ ] Implement baseline Bayesian VAR model
- [ ] Integrate DAG with causal assumptions
- [ ] Implement Bayesian RNN with temporal attention
- [ ] Apply Pearl’s `do-operator` for intervention analysis
- [ ] Evaluate model with posterior predictive checks
- [ ] Visualize causal effects and forecast intervals
- [ ] Compare performance across model types

---

### Requirements

To install dependencies:

```bash
pip install -r requirements.txt
```
---

### Team
- Abdelrahman Elmay
- Anish Ambreth
- Anjali Khantaal

