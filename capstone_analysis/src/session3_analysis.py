"""
Session 3 – Sensitivity Analysis, Vega, Uri Decomposition, Unified Results
Capstone: "The Mispriced Reliability Option: Demand Flexibility and Untapped Alpha
           in Power Markets"

Tasks
  1. Uri-excluded NPV decomposition for Regime 2
  2. Sensitivity sweeps: K, N, p  (Regime 2 base parameters)
  3. Vega analysis (all three regimes)
  4. Unified cross-regime comparison table

Outputs
  outputs/tables/  uri_decomposition.csv, sensitivity_K.csv,
                   sensitivity_N.csv, sensitivity_p.csv,
                   vega_results.csv, unified_results.csv
  outputs/figures/ fig7_sensitivity_K.png, fig8_sensitivity_N.png,
                   fig9_vega.png, fig10_uri_decomposition.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

# ─────────────────────────────────────────────────────────────
# 0.  Paths
# ─────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, 'data', 'raw')
FIG_DIR = os.path.join(BASE, 'outputs', 'figures')
TAB_DIR = os.path.join(BASE, 'outputs', 'tables')

# ─────────────────────────────────────────────────────────────
# 1.  Constants
# ─────────────────────────────────────────────────────────────
DT      = 1.0 / 35040
N_STEPS = 35040
N_PATHS = 1000
N_BASE  = 100
C       = 1.0
P_BASE  = 0.80
K_BASE  = 50.0
MU_Q    = N_BASE * P_BASE * C          # 80 MW

URI_START = '2021-02-10'
URI_END   = '2021-02-20'

VOLL_MAP = {'regime_1': 9000, 'regime_2': 9000, 'regime_3': 5000}
REGIME_DATES = {
    'regime_1': ('2018-01-01', '2019-02-28'),
    'regime_2': ('2019-03-01', '2021-12-31'),
    'regime_3': ('2022-01-01', '2024-12-31'),
}
REGIME_COLORS = {
    'regime_1': '#AED6F1',
    'regime_2': '#A9DFBF',
    'regime_3': '#F9E79F',
}
REGIME_LABELS = {
    'regime_1': 'Regime 1 (Jan 2018 – Feb 2019)',
    'regime_2': 'Regime 2 (Mar 2019 – Dec 2021)',
    'regime_3': 'Regime 3 (Jan 2022 – Dec 2024)',
}

# Sweep grids
K_VALS = [10, 20, 30, 40, 50, 75, 100, 150, 200]
N_VALS = [10, 25, 50, 100, 200, 300, 500]
P_VALS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# ─────────────────────────────────────────────────────────────
# 2.  Load prior calibration results
# ─────────────────────────────────────────────────────────────
print('Loading prior session results ...')
calib_df   = pd.read_csv(os.path.join(TAB_DIR, 'calibration_parameters.csv')).set_index('regime')
trans_df   = pd.read_csv(os.path.join(TAB_DIR, 'transition_probabilities.csv'))
summary_df = pd.read_csv(os.path.join(TAB_DIR, 'results_summary.csv')).set_index('regime')

p_active_map = (
    trans_df[trans_df['season'] == 'all']
    .set_index('regime')['p_active']
    .to_dict()
)

# Convenience: extract Regime 2 parameters (used for all sweeps)
r2 = calib_df.loc['regime_2']
R2_PARAMS = dict(
    mu       = float(r2['mu']),
    alpha    = float(r2['alpha']),
    sigma    = float(r2['sigma']),
    lam      = float(r2['lam']),
    mu_J     = float(r2['mu_J']),
    p_active = float(p_active_map['regime_2']),
    VOLL     = VOLL_MAP['regime_2'],
)
print(f'  Regime 2 params: {R2_PARAMS}')

# ─────────────────────────────────────────────────────────────
# 3.  Load and label raw historical data
# ─────────────────────────────────────────────────────────────
print('Loading historical ORDC data ...')
df_raw = pd.read_csv(os.path.join(RAW_DIR, 'ordc_data.csv'))
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
if 'ordc_adder_online' in df_raw.columns:
    df_raw = df_raw.rename(columns={'ordc_adder_online': 'ordc_adder'})
elif 'ordc_adder' not in df_raw.columns:
    col = df_raw.select_dtypes(include=[np.number]).columns[0]
    df_raw = df_raw.rename(columns={col: 'ordc_adder'})
df_raw = df_raw[['datetime', 'ordc_adder']].sort_values('datetime').reset_index(drop=True)

df_raw['regime'] = 'outside'
for r, (s, e) in REGIME_DATES.items():
    df_raw.loc[(df_raw['datetime'] >= s) & (df_raw['datetime'] <= e), 'regime'] = r
df_raw['is_uri'] = (df_raw['datetime'] >= URI_START) & (df_raw['datetime'] <= URI_END)

# Pre-slice regime 2 series (full and no-Uri)
r2_full   = df_raw.loc[df_raw['regime'] == 'regime_2', 'ordc_adder'].values
r2_no_uri = df_raw.loc[(df_raw['regime'] == 'regime_2') & ~df_raw['is_uri'],
                       'ordc_adder'].values

# Pre-compute E[max(S-K_BASE, 0)] for Regime 2 (used as fixed NPV in N/p sweeps)
E_pay_r2_base = float(np.mean(np.maximum(r2_full - K_BASE, 0)))
V_NPV_r2_base_per_MW = E_pay_r2_base * N_STEPS * 0.25  # per MW-year

print(f'  Regime 2 full: {len(r2_full):,} intervals')
print(f'  Regime 2 no-Uri: {len(r2_no_uri):,} intervals')
print(f'  E[max(S-50,0)] Regime 2 full:   ${E_pay_r2_base:.4f}/interval')

# ─────────────────────────────────────────────────────────────
# 4.  Simulation function (exact replica from Session 2)
# ─────────────────────────────────────────────────────────────
def simulate_paths(mu, alpha, sigma, lam, mu_J, p_active, VOLL,
                   N, c, p_port, K, n_paths, n_steps, dt,
                   verbose=False):
    """
    Two-state OU+jump simulation.  Returns S (n_paths, n_steps) and V (n_paths,).
    Sigma is in annualized units; do NOT rescale before passing in.
    """
    mu_Q_loc    = N * p_port * c
    sigma_Q_loc = np.sqrt(N * p_port * (1 - p_port)) * c
    sqrt_dt     = np.sqrt(dt)

    Z           = np.random.standard_normal((n_paths, n_steps))
    state_draws = np.random.uniform(size=(n_paths, n_steps))
    jump_draws  = np.random.uniform(size=(n_paths, n_steps))
    jump_sizes  = np.random.exponential(max(mu_J, 1e-6), size=(n_paths, n_steps))
    Q_draws     = np.random.normal(mu_Q_loc, sigma_Q_loc, size=(n_paths, n_steps))
    Q_draws     = np.clip(Q_draws, 0, N * c)

    S       = np.zeros((n_paths, n_steps))
    payoffs = np.zeros((n_paths, n_steps))
    S[:, 0] = mu

    for t in range(1, n_steps):
        if verbose and t % 5000 == 0:
            print(f'      step {t:,}/{n_steps:,}')

        active    = (state_draws[:, t] < p_active).astype(float)
        jump_mask = (jump_draws[:, t] < lam * dt).astype(float)
        jump      = jump_sizes[:, t] * jump_mask

        dS = (alpha * (mu - S[:, t - 1]) * dt
              + sigma * sqrt_dt * Z[:, t]
              + jump)

        S[:, t] = (S[:, t - 1] + dS) * active
        S[:, t] = np.clip(S[:, t], 0, VOLL)

        payoffs[:, t] = np.maximum(S[:, t] - K, 0) * Q_draws[:, t] * 0.25

    return S, payoffs.sum(axis=1)


def run_r2(K=K_BASE, N=N_BASE, p_port=P_BASE, seed=42, verbose=False):
    """Convenience wrapper: run Regime 2 simulation with modified K/N/p."""
    np.random.seed(seed)
    return simulate_paths(
        verbose=verbose,
        K=K, N=N, c=C, p_port=p_port,
        n_paths=N_PATHS, n_steps=N_STEPS, dt=DT,
        **{k: v for k, v in R2_PARAMS.items()},
    )


# ─────────────────────────────────────────────────────────────
# TASK 1 — Uri-Excluded NPV Decomposition
# ─────────────────────────────────────────────────────────────
print('\n' + '='*62)
print('TASK 1 – URI NPV DECOMPOSITION')
print('='*62)

try:
    E_full   = float(np.mean(np.maximum(r2_full   - K_BASE, 0)))
    E_no_uri = float(np.mean(np.maximum(r2_no_uri - K_BASE, 0)))

    V_NPV_full    = E_full   * N_STEPS * MU_Q * 0.25
    V_NPV_no_uri  = E_no_uri * N_STEPS * MU_Q * 0.25
    uri_contrib   = V_NPV_full - V_NPV_no_uri
    uri_pct       = uri_contrib / V_NPV_full * 100

    uri_decomp = pd.DataFrame([{
        'V_NPV_full':      round(V_NPV_full   / MU_Q, 2),
        'V_NPV_no_uri':    round(V_NPV_no_uri / MU_Q, 2),
        'uri_contribution':round(uri_contrib  / MU_Q, 2),
        'uri_pct':         round(uri_pct,              4),
        'duration_hours':  80,
    }])
    uri_decomp.to_csv(os.path.join(TAB_DIR, 'uri_decomposition.csv'), index=False)

    print(f'  V_NPV full       : ${V_NPV_full/MU_Q:>10,.2f} /MW-yr')
    print(f'  V_NPV ex-Uri     : ${V_NPV_no_uri/MU_Q:>10,.2f} /MW-yr')
    print(f'  Uri contribution : ${uri_contrib/MU_Q:>10,.2f} /MW-yr  ({uri_pct:.1f}% of benchmark)')
    print(f'  Saved uri_decomposition.csv')

except Exception as exc:
    import traceback; traceback.print_exc()
    uri_contrib = 0.0; V_NPV_no_uri = 0.0; uri_pct = 0.0
    V_NPV_full = V_NPV_r2_base_per_MW * MU_Q

# ─────────────────────────────────────────────────────────────
# TASK 2 — Sensitivity Sweeps
# ─────────────────────────────────────────────────────────────
print('\n' + '='*62)
print('TASK 2 – SENSITIVITY SWEEPS  (Regime 2 base parameters)')
print('='*62)

# ── Sweep 1: Activation Cost K ────────────────────────────────
print('\nSweep 1 – Activation Cost K ...')
sens_K_rows = []
for k_val in K_VALS:
    _, V = run_r2(K=k_val)
    mu_q = MU_Q
    V0   = float(np.mean(V))

    # Recompute V_NPV at this K from historical data
    E_k   = float(np.mean(np.maximum(r2_full - k_val, 0)))
    v_npv = E_k * N_STEPS * mu_q * 0.25
    prem  = V0 - v_npv
    pct   = (prem / v_npv * 100) if v_npv > 0 else float('nan')

    sens_K_rows.append({
        'K':              k_val,
        'V0':             round(V0    / mu_q, 2),
        'V_NPV':          round(v_npv / mu_q, 2),
        'option_premium': round(prem  / mu_q, 2),
        'premium_pct':    round(pct,           4),
    })
    print(f'  K={k_val:>3}  V0={V0/mu_q:>10,.2f}  V_NPV={v_npv/mu_q:>10,.2f}  '
          f'premium={prem/mu_q:>10,.2f}  ({pct:.1f}%)')

sens_K_df = pd.DataFrame(sens_K_rows)
sens_K_df.to_csv(os.path.join(TAB_DIR, 'sensitivity_K.csv'), index=False)
print('  Saved sensitivity_K.csv')

# ── Sweep 2: Portfolio Size N ─────────────────────────────────
print('\nSweep 2 – Portfolio Size N ...')
sens_N_rows = []
for n_val in N_VALS:
    mu_q_n    = n_val * P_BASE * C
    _, V = run_r2(N=n_val)
    V0_total  = float(np.mean(V))
    v_npv_n   = E_pay_r2_base * N_STEPS * mu_q_n * 0.25
    prem_n    = V0_total - v_npv_n
    pct_n     = (prem_n / v_npv_n * 100) if v_npv_n > 0 else float('nan')

    sens_N_rows.append({
        'N':                       n_val,
        'V0_total':                round(V0_total,         2),
        'V0_per_MW_year':          round(V0_total / mu_q_n, 2),
        'option_premium_per_MW_year': round(prem_n / mu_q_n, 2),
        'premium_pct':             round(pct_n, 4),
    })
    print(f'  N={n_val:>3}  V0={V0_total/mu_q_n:>10,.2f}/MW-yr  '
          f'premium={prem_n/mu_q_n:>10,.2f}/MW-yr  ({pct_n:.1f}%)')

sens_N_df = pd.DataFrame(sens_N_rows)
sens_N_df.to_csv(os.path.join(TAB_DIR, 'sensitivity_N.csv'), index=False)
print('  Saved sensitivity_N.csv')

# ── Sweep 3: Availability Probability p ──────────────────────
print('\nSweep 3 – Availability Probability p ...')
sens_p_rows = []
for p_val in P_VALS:
    mu_q_p  = N_BASE * p_val * C
    _, V = run_r2(p_port=p_val)
    V0_p    = float(np.mean(V))
    v_npv_p = E_pay_r2_base * N_STEPS * mu_q_p * 0.25
    prem_p  = V0_p - v_npv_p
    pct_p   = (prem_p / v_npv_p * 100) if v_npv_p > 0 else float('nan')

    sens_p_rows.append({
        'p':              p_val,
        'V0':             round(V0_p    / mu_q_p, 2),
        'V_NPV':          round(v_npv_p / mu_q_p, 2),
        'option_premium': round(prem_p  / mu_q_p, 2),
        'premium_pct':    round(pct_p,             4),
    })
    print(f'  p={p_val:.2f}  V0={V0_p/mu_q_p:>10,.2f}/MW-yr  '
          f'premium={prem_p/mu_q_p:>10,.2f}/MW-yr  ({pct_p:.1f}%)')

sens_p_df = pd.DataFrame(sens_p_rows)
sens_p_df.to_csv(os.path.join(TAB_DIR, 'sensitivity_p.csv'), index=False)
print('  Saved sensitivity_p.csv')

# ─────────────────────────────────────────────────────────────
# TASK 3 — Vega Analysis
# ─────────────────────────────────────────────────────────────
print('\n' + '='*62)
print('TASK 3 – VEGA ANALYSIS')
print('='*62)

vega_rows = []
for rname in ['regime_1', 'regime_2', 'regime_3']:
    try:
        row      = calib_df.loc[rname]
        mu       = float(row['mu'])
        alpha    = float(row['alpha'])
        sigma    = float(row['sigma'])
        lam      = float(row['lam'])
        mu_J     = float(row['mu_J'])
        p_act    = float(p_active_map[rname])
        VOLL     = VOLL_MAP[rname]

        # Base run
        np.random.seed(42)
        _, V_base = simulate_paths(
            mu, alpha, sigma, lam, mu_J, p_act, VOLL,
            N_BASE, C, P_BASE, K_BASE, N_PATHS, N_STEPS, DT
        )
        V0_base = float(np.mean(V_base))

        # Bumped sigma run (same seed → identical random draws)
        sigma_b = sigma * 1.01
        np.random.seed(42)
        _, V_bump = simulate_paths(
            mu, alpha, sigma_b, lam, mu_J, p_act, VOLL,
            N_BASE, C, P_BASE, K_BASE, N_PATHS, N_STEPS, DT
        )
        V0_bump = float(np.mean(V_bump))

        vega       = (V0_bump - V0_base) / (sigma * 0.01)
        vega_per_MW = vega / MU_Q

        vega_rows.append({
            'regime':         rname,
            'sigma':          round(sigma,      4),
            'sigma_bumped':   round(sigma_b,    4),
            'V0_base':        round(V0_base  / MU_Q, 2),
            'V0_bumped':      round(V0_bump  / MU_Q, 2),
            'vega':           round(vega,       4),
            'vega_per_MW_year': round(vega_per_MW, 4),
        })
        print(f'  {rname}: sigma={sigma:.1f}  V0_base={V0_base/MU_Q:,.2f}  '
              f'V0_bump={V0_bump/MU_Q:,.2f}  vega={vega:.4f}  '
              f'vega/MW={vega_per_MW:.4f}')

    except Exception as exc:
        import traceback; traceback.print_exc()

vega_df = pd.DataFrame(vega_rows)
vega_df.to_csv(os.path.join(TAB_DIR, 'vega_results.csv'), index=False)
print('  Saved vega_results.csv')

# Cross-regime rank correlation: sigma vs V0
sigmas = [float(calib_df.loc[r, 'sigma']) for r in ['regime_1','regime_2','regime_3']]
v0s    = [float(summary_df.loc[r, 'V0_per_MW_year']) for r in ['regime_1','regime_2','regime_3']]
from scipy.stats import spearmanr
rho, p_rho = spearmanr(sigmas, v0s)
print(f'\n  Cross-regime Spearman rank corr (sigma vs V0): rho={rho:.3f}  p={p_rho:.4f}')

# ─────────────────────────────────────────────────────────────
# TASK 4 — Unified Results Table
# ─────────────────────────────────────────────────────────────
print('\n' + '='*62)
print('TASK 4 – UNIFIED RESULTS TABLE')
print('='*62)

# Compute V_NPV for regimes 1 and 3 from historical data
def v_npv_regime(rname, k=K_BASE):
    vals = df_raw.loc[df_raw['regime'] == rname, 'ordc_adder'].values
    return float(np.mean(np.maximum(vals - k, 0))) * N_STEPS * MU_Q * 0.25

V_NPV_r1 = v_npv_regime('regime_1')
V_NPV_r3 = v_npv_regime('regime_3')

unified_rows = []

# Regime 1
v0_r1 = float(summary_df.loc['regime_1', 'V0_per_MW_year']) * MU_Q
prem_r1 = v0_r1 - V_NPV_r1
unified_rows.append({
    'regime': 'regime_1',
    'V0': round(v0_r1 / MU_Q, 2),
    'V_NPV_full': round(V_NPV_r1 / MU_Q, 2),
    'V_NPV_no_uri': 'N/A',
    'option_premium': round(prem_r1 / MU_Q, 2),
    'premium_pct': round(prem_r1 / V_NPV_r1 * 100, 2) if V_NPV_r1 > 0 else 'N/A',
    'uri_contribution': 'N/A',
    'uri_pct': 'N/A',
    'interpretation': 'Pre-shift baseline: moderate scarcity pricing',
})

# Regime 2
v0_r2 = float(summary_df.loc['regime_2', 'V0_per_MW_year']) * MU_Q
prem_r2 = v0_r2 - V_NPV_full
unified_rows.append({
    'regime': 'regime_2',
    'V0': round(v0_r2 / MU_Q, 2),
    'V_NPV_full': round(V_NPV_full / MU_Q, 2),
    'V_NPV_no_uri': round(V_NPV_no_uri / MU_Q, 2),
    'option_premium': round(prem_r2 / MU_Q, 2),
    'premium_pct': round(prem_r2 / V_NPV_full * 100, 2) if V_NPV_full > 0 else 'N/A',
    'uri_contribution': round(uri_contrib / MU_Q, 2),
    'uri_pct': round(uri_pct, 2),
    'interpretation': 'High volatility period including Uri tail event',
})

# Regime 3
v0_r3 = float(summary_df.loc['regime_3', 'V0_per_MW_year']) * MU_Q
prem_r3 = v0_r3 - V_NPV_r3
unified_rows.append({
    'regime': 'regime_3',
    'V0': round(v0_r3 / MU_Q, 2),
    'V_NPV_full': round(V_NPV_r3 / MU_Q, 2),
    'V_NPV_no_uri': 'N/A',
    'option_premium': round(prem_r3 / MU_Q, 2),
    'premium_pct': round(prem_r3 / V_NPV_r3 * 100, 2) if V_NPV_r3 > 0 else 'N/A',
    'uri_contribution': 'N/A',
    'uri_pct': 'N/A',
    'interpretation': 'Post-Uri reformed regime: lower VOLL cap',
})

unified_df = pd.DataFrame(unified_rows)
unified_df.to_csv(os.path.join(TAB_DIR, 'unified_results.csv'), index=False)
print('  Saved unified_results.csv')
for row in unified_rows:
    print(f'  {row["regime"]:10s}  V0={row["V0"]:>8}  V_NPV={row["V_NPV_full"]:>8}  '
          f'premium={row["option_premium"]:>8}  [{row["interpretation"][:35]}]')

# ─────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────

# ── fig7: Sensitivity K ───────────────────────────────────────
print('\nGenerating fig7_sensitivity_K.png ...')
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    k_arr   = sens_K_df['K'].values
    v0_arr  = sens_K_df['V0'].values
    npv_arr = sens_K_df['V_NPV'].values

    ax.plot(k_arr, v0_arr,  color='#1A5276', lw=2.0, marker='o', ms=5, label='V0 (MC option value)')
    ax.plot(k_arr, npv_arr, color='#922B21', lw=2.0, marker='s', ms=5, linestyle='--',
            label='V_NPV (historical benchmark)')

    # Shade gap: green where V0 > V_NPV, red otherwise
    ax.fill_between(k_arr, v0_arr, npv_arr,
                    where=(v0_arr >= npv_arr),
                    alpha=0.18, color='#28B463', label='V0 > V_NPV (positive premium)')
    ax.fill_between(k_arr, v0_arr, npv_arr,
                    where=(v0_arr < npv_arr),
                    alpha=0.18, color='#E74C3C', label='V0 < V_NPV (negative premium)')

    ax.axvline(K_BASE, color='#566573', linestyle=':', lw=1.6, label=f'Base K = {K_BASE} $/MWh')
    ax.set_xlabel('Activation Cost K ($/MWh)', fontsize=11)
    ax.set_ylabel('$/MW-year', fontsize=11)
    ax.set_title('Option Value and NPV Benchmark vs Activation Cost\n'
                 'Regime 2 parameters (Mar 2019 – Dec 2021)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(k_arr.min(), k_arr.max())
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig7_sensitivity_K.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig7_sensitivity_K.png')
except Exception as exc:
    print(f'  ERROR fig7: {exc}')

# ── fig8: Sensitivity N ───────────────────────────────────────
print('Generating fig8_sensitivity_N.png ...')
try:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    n_arr    = sens_N_df['N'].values
    prem_mw  = sens_N_df['option_premium_per_MW_year'].values
    prem_tot = sens_N_df['N'].values * P_BASE * C * prem_mw   # total $

    ax1.plot(n_arr, prem_mw, color='#8E44AD', lw=2.0, marker='D', ms=6)
    ax1.axhline(prem_mw[3], color='#566573', linestyle=':', lw=1.4,
                label=f'Base case N=100: ${prem_mw[3]:,.0f}/MW-yr')
    ax1.set_xlabel('Portfolio Size N (loads)', fontsize=11)
    ax1.set_ylabel('Option Premium ($/MW-year)', fontsize=11, color='#8E44AD')
    ax1.tick_params(axis='y', labelcolor='#8E44AD')

    ax2 = ax1.twinx()
    ax2.plot(n_arr, prem_tot, color='#1A5276', lw=1.5, linestyle='--', alpha=0.7)
    ax2.set_ylabel('Total Option Premium ($)', fontsize=11, color='#1A5276')
    ax2.tick_params(axis='y', labelcolor='#1A5276')

    ax1.set_title('Option Premium vs Portfolio Size N\n'
                  'Flat $/MW-year line = linear scaling holds\n'
                  'Regime 2 parameters', fontsize=11)
    ax1.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig8_sensitivity_N.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig8_sensitivity_N.png')
except Exception as exc:
    print(f'  ERROR fig8: {exc}')

# ── fig9: Vega bar chart ──────────────────────────────────────
print('Generating fig9_vega.png ...')
try:
    fig, ax = plt.subplots(figsize=(9, 6))
    r_names   = vega_df['regime'].tolist()
    vega_vals = vega_df['vega_per_MW_year'].values
    sig_vals  = vega_df['sigma'].values
    bar_colors = [REGIME_COLORS[r] for r in r_names]
    bars = ax.bar(r_names, vega_vals, color=bar_colors,
                  edgecolor='#2C3E50', linewidth=0.8, width=0.5)

    for bar, sig in zip(bars, sig_vals):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + abs(h) * 0.04,
                f'σ = {sig:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticklabels(['Regime 1', 'Regime 2', 'Regime 3'], fontsize=11)
    ax.set_ylabel('Vega ($/MW-year per unit annualized σ)', fontsize=10)
    ax.set_title('Option Premium Sensitivity to ORDC Volatility\n'
                 'Vega = ΔV0 / (σ × 0.01)  |  Higher ORDC volatility produces higher option value',
                 fontsize=10)
    ax.axhline(0, color='#566573', lw=0.8, linestyle='--')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig9_vega.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig9_vega.png')
except Exception as exc:
    print(f'  ERROR fig9: {exc}')

# ── fig10: Uri decomposition stacked bar ──────────────────────
print('Generating fig10_uri_decomposition.png ...')
try:
    fig, ax = plt.subplots(figsize=(8, 7))

    no_uri_mw  = float(V_NPV_no_uri / MU_Q)
    uri_mw     = float(uri_contrib  / MU_Q)
    full_mw    = float(V_NPV_full   / MU_Q)
    v0_r2_mw   = float(summary_df.loc['regime_2', 'V0_per_MW_year'])

    ax.bar(['Regime 2\nHistorical Benchmark'], [no_uri_mw],
           color='#A9DFBF', edgecolor='#2C3E50', linewidth=0.8,
           label=f'V_NPV ex-Uri  ${no_uri_mw:,.0f}/MW-yr')
    ax.bar(['Regime 2\nHistorical Benchmark'], [uri_mw],
           bottom=[no_uri_mw], color='#E74C3C', alpha=0.80,
           edgecolor='#2C3E50', linewidth=0.8,
           label=f'Uri contribution  ${uri_mw:,.0f}/MW-yr  ({uri_pct:.1f}%)')

    ax.axhline(v0_r2_mw, color='#1A5276', lw=2.5, linestyle='--',
               label=f'V0 simulated  ${v0_r2_mw:,.0f}/MW-yr')

    # Value labels
    ax.text(0, no_uri_mw / 2,       f'${no_uri_mw:,.0f}',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#1A5276')
    ax.text(0, no_uri_mw + uri_mw / 2, f'${uri_mw:,.0f}\n({uri_pct:.1f}%)',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.text(0.35, v0_r2_mw * 1.02,  f'V0 = ${v0_r2_mw:,.0f}',
            ha='left', va='bottom', fontsize=10, color='#1A5276', fontweight='bold')

    ax.set_ylabel('$/MW-year', fontsize=11)
    ax.set_title("Uri's Contribution to Regime 2 Historical Benchmark\n"
                 "Tail event concentration: Uri accounts for most of V_NPV",
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(-0.5, 0.8)
    ax.set_ylim(0, full_mw * 1.20)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig10_uri_decomposition.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig10_uri_decomposition.png')
except Exception as exc:
    print(f'  ERROR fig10: {exc}')

# ─────────────────────────────────────────────────────────────
# FINAL CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────
elapsed = time.time() - t0

v0_map = {
    'regime_1': float(summary_df.loc['regime_1', 'V0_per_MW_year']),
    'regime_2': float(summary_df.loc['regime_2', 'V0_per_MW_year']),
    'regime_3': float(summary_df.loc['regime_3', 'V0_per_MW_year']),
}
npv_map = {
    'regime_1': V_NPV_r1 / MU_Q,
    'regime_2': V_NPV_full / MU_Q,
    'regime_3': V_NPV_r3 / MU_Q,
}
npv_no_uri_map = {
    'regime_1': float('nan'),
    'regime_2': V_NPV_no_uri / MU_Q,
    'regime_3': float('nan'),
}
prem_map  = {r: v0_map[r] - npv_map[r] for r in v0_map}
pct_map   = {r: prem_map[r] / npv_map[r] * 100 for r in v0_map}

print('\n' + '='*75)
print('--- FINAL RESULTS SUMMARY ---')
print(f'{"Regime":<8} {"V0 ($/MW-yr)":>14} {"V_NPV Full":>12} '
      f'{"V_NPV ex-Uri":>14} {"Premium":>10} {"Premium %":>10}')
print('-' * 75)
for i, r in enumerate(['regime_1', 'regime_2', 'regime_3'], 1):
    ex_uri = f'${npv_no_uri_map[r]:>10,.0f}' if r == 'regime_2' else f'{"N/A":>11}'
    print(f'  {i:<6} {v0_map[r]:>14,.2f} {npv_map[r]:>12,.2f} '
          f'{ex_uri:>14} {prem_map[r]:>10,.2f} {pct_map[r]:>9.1f}%')

print('\n--- URI DECOMPOSITION (Regime 2) ---')
print(f'  V_NPV including Uri:  ${V_NPV_full/MU_Q:>10,.2f} /MW-year')
print(f'  V_NPV excluding Uri:  ${V_NPV_no_uri/MU_Q:>10,.2f} /MW-year')
print(f'  Uri contribution:     ${uri_contrib/MU_Q:>10,.2f} /MW-year  ({uri_pct:.1f}% of benchmark)')
print(f'  Duration:             80 hours')

print('\n--- VEGA SUMMARY ---')
vega_map = vega_df.set_index('regime')
for r in ['regime_1', 'regime_2', 'regime_3']:
    if r in vega_map.index:
        row = vega_map.loc[r]
        print(f'  {r}: Vega = {row["vega_per_MW_year"]:.4f} $/MW-year per unit sigma  '
              f'(sigma = {row["sigma"]:,.0f})')

print(f'\n  Cross-regime Spearman rho (sigma vs V0): {rho:.3f}  (p={p_rho:.4f})')

all_tables = [
    'uri_decomposition.csv', 'sensitivity_K.csv', 'sensitivity_N.csv',
    'sensitivity_p.csv', 'vega_results.csv', 'unified_results.csv',
]
all_figs = [f'fig{i}' for i in range(1, 11)]

saved_t = sum(os.path.exists(os.path.join(TAB_DIR, f)) for f in all_tables)
saved_f = sum(os.path.exists(os.path.join(FIG_DIR, f + '.png')) for f in all_figs) + \
          sum(os.path.exists(os.path.join(FIG_DIR, f + '_sensitivity_K.png'))
              for f in ['fig7']) * 0  # already counted above

# Count all png in figures dir
n_figs = len([f for f in os.listdir(FIG_DIR) if f.endswith('.png')])
n_tabs = len([f for f in os.listdir(TAB_DIR) if f.endswith('.csv')])

print('\n--- FILES SAVED ---')
print(f'  Figures: outputs/figures/  ({n_figs} total)')
print(f'  Tables:  outputs/tables/   ({n_tabs} total)')
print(f'\nAnalysis complete.  Total runtime: {elapsed:.0f} seconds.')
