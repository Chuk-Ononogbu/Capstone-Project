# The Mispriced Reliability Option: Demand Flexibility and Untapped Alpha in Power Markets

**MASE Capstone Research Paper — Johns Hopkins SAIS, Sustainable Energy Program**

---

## Overview

This repository contains the full research paper and supporting analysis code for a graduate capstone examining whether demand-side flexibility in competitive electricity markets is systematically mispriced.

The core argument: demand response earns almost nothing most of the time and a great deal during rare grid emergencies. That payoff structure is identical to a financial option, and standard investment analysis is not designed to value options correctly. By treating ERCOT's Operating Reserve Demand Curve (ORDC) adder as the underlying asset and calibrating a two-state mean-reverting jump-diffusion model to seven years of historical data, this paper estimates the true option value of demand flexibility and shows that it rises in direct proportion to market volatility.

---

## Key Findings

- The ORDC adder is near zero in **96–98% of all 15-minute market intervals**, confirming the option-like payoff structure of demand response
- A single 80-hour period during **Winter Storm Uri (February 2021) contributed 48.6% of the entire Regime 2 historical benchmark** — less than 1% of the hours generated nearly half the value
- Forward-looking simulated option values range from **$2,070 to $12,231 per MW-year** depending on the market regime
- Simulated option value rises with ORDC volatility consistently across all three study regimes (**Vega confirmed positive across all regimes**)
- Per-MW option value is **independent of portfolio size**, meaning the barrier to capturing scarcity value is activation cost, not scale

---

## Repository Structure

```
capstone_analysis/
├── data/
│   ├── raw/              # Raw ERCOT ORDC settlement data (15-min intervals)
│   └── processed/        # Cleaned and deseasonalized series
├── outputs/
│   ├── figures/          # All 10 publication-quality figures (300 dpi)
│   └── tables/           # Calibration parameters, results, sensitivity CSVs
└── src/
    └── analysis.py       # Full analysis pipeline (calibration + simulation)
```

---

## Data

Data source: [ERCOT Real-Time Market settlement records](https://www.ercot.com/mp/data-products/data-product-details?id=NP6-793-ER)

- **Series used:** Real-Time On-Line Reserve Price (RTOLCAP adder)
- **Study period:** January 2018 – December 2024
- **Observations:** ~209,000 usable 15-minute intervals after cleaning
- **Three regimes:** Pre-shift baseline (2018–Feb 2019), high-volatility period including Uri (Mar 2019–2021), post-Uri reformed regime (2022–2024)

---

## Methodology

The ORDC adder is modeled as a **two-state mean-reverting jump-diffusion process**:

- **Quiet state** (96–98% of intervals): adder set to zero
- **Active state**: adder follows an Ornstein-Uhlenbeck process with Poisson jumps

```
dS(t) = α(μ − S(t))dt + σdW(t) + J·dN(t)
```

Parameters are calibrated separately for each regime using a stepwise moment-matching procedure. A **Monte Carlo simulation** (10,000 paths × 35,040 intervals) estimates the expected annual option payoff under the physical probability measure.

---

## Running the Analysis

```bash
# Install dependencies
pip install numpy pandas scipy matplotlib seaborn statsmodels

# Run full pipeline
python src/analysis.py
```

Outputs are saved to `outputs/figures/` and `outputs/tables/`. Runtime is approximately 3–4 minutes at 10,000 paths.

---

## Requirements

```
numpy
pandas
scipy
matplotlib
seaborn
statsmodels
```

Python 3.8 or higher recommended.

---

## Paper

The full capstone paper is available in this repository as `Capstone_Project.docx`.

**Key references:**
- Cartea & Figueroa (2005) — mean-reverting jump-diffusion model
- Hogan (2005) — ORDC theoretical foundation
- Huisman & Mahieu (2003) — two-state regime structure
- Dixit & Pindyck (1994) — real options theory

---

*Johns Hopkins SAIS · MASE · May 2026*
