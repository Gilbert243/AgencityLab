# AgencityLab

**AgencityLab** is a scientific Python framework dedicated to the study and quantification of **Agencity** — an observable designed to measure a system’s ability to actively structure its own evolution.

Unlike classical metrics (variance, entropy, energy), Agencity captures **dynamic organization and decision-like behavior within a system**, independent of domain-specific assumptions.

Agencity can be analyzed:
- at a **single scale**
- across **multiple scales** via the **Agencity spectrum** \( b(\tau) \)

---

## 👤 Author

**Gilbert BEMWIZ**  
Conceptor of the Agencity theory  
Engineer and developer of the AgencityLab framework  

---

## 🧠 Vision

AgencityLab aims to provide:

- a **rigorous computational implementation** of Agencity theory  
- a **domain-independent analysis tool** applicable to:
  - physical systems  
  - biological signals  
  - economic dynamics  
  - computational processes  
- an experimental platform to explore links with:
  - **Shannon information theory**
  - **Landauer’s principle**
  - **informational physics hypotheses** (e.g. Vopson)

---

## 🔬 Core Concept

Agencity is not a measure of randomness or order.

It quantifies:

> **the capacity of a system to select and sustain a trajectory among those allowed by its underlying dynamics**

This leads to a hierarchy of observables:

```

u → u* → X* → A* → M → O → β → b

````

Where:

- **u** : raw signal  
- **u*** : normalized signal  
- **X*** : activation  
- **A*** : activity  
- **M** : memory (causal correlation)  
- **O** : organization  
- **β** : structured agency  
- **b** : Agencity (final observable)

---

## ⚙️ Installation

```bash
pip install agencitylab
````

Optional features:

```bash
pip install agencitylab[visualization]
pip install agencitylab[dashboard]
pip install agencitylab[all]
```

---

## 🚀 Quick Start

```python
import numpy as np
from agencitylab.api import compute_agencity

xi = np.linspace(0, 10, 200)
u = np.sin(xi)

result = compute_agencity(xi, u)

print(result.summary())
```

---

## 📊 Visualization

```python
from agencitylab.api import visualize_agencity

visualize_agencity(result, kind="timeseries")
```

Available visualizations:

* `timeseries` → full pipeline signals
* `spectrum` → frequency structure of b
* `phase` → dynamic trajectories
* `components` → internal decomposition

---

## 📈 Multi-Scale Analysis

Agencity can be analyzed across scales:

```python
from agencitylab.analysis.multi_scale import agencity_spectrum

taus = [0.5, 1, 2, 5, 10]
spec = agencity_spectrum(xi, u, taus)
```

This produces a **multi-scale signature** of the system.

---

## 🧬 Agencity Signature

Each system exhibits a characteristic **Agencity signature**:

* constant → inactive
* sinus → structured dynamics
* noise → high variability, low organization
* multi-scale → emergent organization

This enables:

* system classification
* clustering
* regime detection

---

## 🧪 Example Systems

AgencityLab can analyze:

* periodic signals
* stochastic processes
* hybrid signals (signal + noise)
* multi-scale dynamics

---

## 📊 Comparison with Classical Metrics

Agencity complements traditional measures:

| Metric       | Captures                    | Limitation                     |
| ------------ | --------------------------- | ------------------------------ |
| Variance     | dispersion                  | ignores structure              |
| Entropy      | randomness                  | ignores dynamics               |
| Energy       | amplitude                   | ignores organization           |
| Autocorr     | temporal dependence         | local only                     |
| **Agencity** | dynamic structuring ability | multi-scale, causal, intrinsic |

---

## 🧠 Interpretation

* A **zigzag trajectory is not noise** if imposed by constraints
* A system is not “ordered” or “disordered” absolutely
* Agencity is **contextual and intrinsic to system dynamics**

---

## 📁 Project Structure

```
agencitylab/
├── core/            # mathematical operators
├── analysis/        # spectrum, signatures, clustering
├── api/             # user-facing interface
├── visualization/   # plotting tools
├── examples/        # usage examples
```

---

## 🔭 Roadmap

* [ ] Advanced multi-scale theory
* [ ] Real-world datasets integration
* [ ] GPU acceleration
* [ ] Interactive dashboards
* [ ] Scientific publication support

---

## 📜 License

MIT License

---

## 🤝 Contributions

Contributions are welcome.

You can:

* open issues
* propose improvements
* extend analysis modules
* test on new domains

---

## ⚠️ Disclaimer

Agencity is an emerging theoretical framework.
This library provides a computational implementation intended for:

* research
* experimentation
* conceptual exploration

---

## 🌍 Perspective

AgencityLab is part of a broader effort to build:

> **a unified science of dynamic systems combining physics, information, and structure**
