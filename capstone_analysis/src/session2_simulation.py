"""
Session 2 – Monte Carlo Simulation and Option Valuation
Capstone: "The Mispriced Reliability Option: Demand Flexibility and Untapped Alpha
           in Power Markets"

Loads Session 1 calibration results and runs vectorised two-state simulations
for each regime, then produces results summary, Uri stress test, and figures 4–6.

Outputs
  outputs/tables/results_summary.csv
  outputs/tables/uri_stress_test.csv
  outputs/figures/fig4_simulated_paths.png
  outputs/figures/fig5_payoff_distribution.png
  outputs/figures/fig6_option_vs_npv.png
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
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE, 'data', 'raw')
FIG_DIR  = os.path.join(BASE, 'outputs', 'figures')
TAB_DIR  = os.path.join(BASE, 'outputs', 'tables')

# ─────────────────────────────────────────────────────────────
# 1.  Constants
# ─────────────────────────────────────────────────────────────
DT      = 1.0 / 35040
N_STEPS = 35040
N_PATHS = 1000
N       = 100
C       = 1.0        # MW per load
P_PORT  = 0.80       # availability probability
K       = 50.0       # $/MWh activation cost
MU_Q    = N * P_PORT * C                         # 80 MW
SIGMA_Q = np.sqrt(N * P_PORT * (1 - P_PORT)) * C # ~4 MW

VOLL_MAP = {'regime_1': 9000, 'regime_2': 9000, 'regime_3': 5000}

REGIME_DATES = {
    'regime_1': ('2018-01-01', '2019-02-28'),
    'regime_2': ('2019-03-01', '2021-12-31'),
    'regime_3': ('2022-01-01', '2024-12-31'),
}
URI_START = '2021-02-10'
URI_END   = '2021-02-20'

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

# ─────────────────────────────────────────────────────────────
# 2.  Load prior calibration results
# ─────────────────────────────────────────────────────────────
print('Loading Session 1 calibration results ...')

calib_df = pd.read_csv(os.path.join(TAB_DIR, 'calibration_parameters.csv'))
calib_df = calib_df.set_index('regime')

trans_df = pd.read_csv(os.path.join(TAB_DIR, 'transition_probabilities.csv'))
# Keep overall (season == 'all') p_active per regime
p_active_map = (
    trans_df[trans_df['season'] == 'all']
    .set_index('regime')['p_active']
    .to_dict()
)

print('  Calibration parameters:')
for r in ['regime_1', 'regime_2', 'regime_3']:
    row = calib_df.loc[r]
    print(f'    {r}: mu={row["mu"]:.2f}  alpha={row["alpha"]:.2f}  '
          f'sigma={row["sigma"]:.2f}  lam={row["lam"]:.2f}  '
          f'mu_J={row["mu_J"]:.2f}  p_active={p_active_map[r]:.4f}')

# ─────────────────────────────────────────────────────────────
# 3.  Load raw data for NPV benchmark
# ─────────────────────────────────────────────────────────────
print('\nLoading historical ORDC data for NPV benchmark ...')
raw_path = os.path.join(RAW_DIR, 'ordc_data.csv')
df_raw   = pd.read_csv(raw_path)
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])

if 'ordc_adder_online' in df_raw.columns:
    df_raw = df_raw.rename(columns={'ordc_adder_online': 'ordc_adder'})
elif 'ordc_adder' not in df_raw.columns:
    num = df_raw.select_dtypes(include=[np.number]).columns[0]
    df_raw = df_raw.rename(columns={num: 'ordc_adder'})

df_raw = df_raw[['datetime', 'ordc_adder']].sort_values('datetime').reset_index(drop=True)

df_raw['regime'] = 'outside'
for r, (s, e) in REGIME_DATES.items():
    mask = (df_raw['datetime'] >= s) & (df_raw['datetime'] <= e)
    df_raw.loc[mask, 'regime'] = r

print(f'  Loaded {len(df_raw):,} rows')

# ─────────────────────────────────────────────────────────────
# 4.  Vectorised two-state simulation
# ─────────────────────────────────────────────────────────────
def simulate_paths(mu, alpha, sigma, lam, mu_J, p_active, VOLL,
                   N, c, p_port, K, n_paths, n_steps, dt):
    """
    Two-state (quiet / active) mean-reversion jump-diffusion simulation.
    Returns
      S        (n_paths, n_steps) – simulated price paths
      V        (n_paths,)         – total annual payoff per path
    """
    mu_Q_loc    = N * p_port * c
    sigma_Q_loc = np.sqrt(N * p_port * (1 - p_port)) * c
    sqrt_dt     = np.sqrt(dt)

    # Pre-generate all random draws
    Z            = np.random.standard_normal((n_paths, n_steps))
    state_draws  = np.random.uniform(size=(n_paths, n_steps))
    jump_draws   = np.random.uniform(size=(n_paths, n_steps))
    jump_sizes   = np.random.exponential(max(mu_J, 1e-6), size=(n_paths, n_steps))
    Q_draws      = np.random.normal(mu_Q_loc, sigma_Q_loc, size=(n_paths, n_steps))
    Q_draws      = np.clip(Q_draws, 0, N * c)

    S       = np.zeros((n_paths, n_steps))
    payoffs = np.zeros((n_paths, n_steps))
    S[:, 0] = mu   # initialise at active conditional mean

    for t in range(1, n_steps):
        if t % 5000 == 0:
            print(f'      step {t:,}/{n_steps:,}')

        active    = (state_draws[:, t] < p_active).astype(float)
        jump_mask = (jump_draws[:, t] < lam * dt).astype(float)
        jump      = jump_sizes[:, t] * jump_mask

        dS = (alpha * (mu - S[:, t - 1]) * dt
              + sigma * sqrt_dt * Z[:, t]
              + jump)

        # Quiet paths → 0; active paths → evolve OU+jump
        S[:, t] = (S[:, t - 1] + dS) * active
        S[:, t] = np.clip(S[:, t], 0, VOLL)

        payoffs[:, t] = np.maximum(S[:, t] - K, 0) * Q_draws[:, t] * 0.25

    V = payoffs.sum(axis=1)
    return S, V


# ─────────────────────────────────────────────────────────────
# 5.  Run simulation for each regime
# ─────────────────────────────────────────────────────────────
print('\n' + '='*62)
print('MONTE CARLO SIMULATION  (n_paths=1,000  n_steps=35,040)')
print('='*62)

results = {}   # regime → metrics dict
stored_S = {}  # regime → S array (kept for figures)

for regime_name in ['regime_1', 'regime_2', 'regime_3']:
    print(f'\n--- {regime_name} ---')
    try:
        row      = calib_df.loc[regime_name]
        mu       = float(row['mu'])
        alpha    = float(row['alpha'])
        sigma    = float(row['sigma'])
        lam      = float(row['lam'])
        mu_J     = float(row['mu_J'])
        p_active = float(p_active_map[regime_name])
        VOLL     = VOLL_MAP[regime_name]

        print(f'  mu={mu:.2f}  alpha={alpha:.1f}  sigma={sigma:.1f}  '
              f'lam={lam:.2f}  mu_J={mu_J:.2f}  '
              f'p_active={p_active:.4f}  VOLL={VOLL}')

        np.random.seed(42)
        t_sim = time.time()
        S_paths, V = simulate_paths(
            mu, alpha, sigma, lam, mu_J, p_active, VOLL,
            N, C, P_PORT, K, N_PATHS, N_STEPS, DT
        )
        print(f'  Simulation done in {time.time() - t_sim:.1f}s')

        # ── NPV benchmark – use full regime historical series ────
        hist_vals    = df_raw.loc[df_raw['regime'] == regime_name, 'ordc_adder'].values
        E_pay_hist   = float(np.mean(np.maximum(hist_vals - K, 0)))
        V_NPV        = E_pay_hist * N_STEPS * MU_Q * 0.25

        # ── Option value statistics ──────────────────────────────
        V0       = float(np.mean(V))
        sigma_V  = float(np.std(V, ddof=1))
        ci_half  = 1.96 * sigma_V / np.sqrt(N_PATHS)
        ci_lower = V0 - ci_half
        ci_upper = V0 + ci_half
        opt_prem = V0 - V_NPV
        prem_pct = (opt_prem / V_NPV * 100) if V_NPV > 0 else 0.0

        print(f'  V0      = ${V0/MU_Q:>12,.2f} /MW-yr')
        print(f'  V_NPV   = ${V_NPV/MU_Q:>12,.2f} /MW-yr')
        print(f'  Premium = ${opt_prem/MU_Q:>12,.2f} /MW-yr  ({prem_pct:.1f}%)')
        print(f'  95% CI  = [${ci_lower/MU_Q:,.2f}, ${ci_upper/MU_Q:,.2f}] /MW-yr')

        results[regime_name] = {
            'regime':             regime_name,
            # $/MW-year (saved to CSV)
            'V0_per_MW_year':     round(V0       / MU_Q, 2),
            'V_NPV_per_MW_year':  round(V_NPV    / MU_Q, 2),
            'option_premium':     round(opt_prem  / MU_Q, 2),
            'premium_pct':        round(prem_pct,          4),
            'ci_lower':           round(ci_lower  / MU_Q, 2),
            'ci_upper':           round(ci_upper  / MU_Q, 2),
            # raw dollar totals (used internally by figures)
            '_V':       V,
            '_V0':      V0,
            '_V_NPV':   V_NPV,
            '_ci_half': ci_half,
        }
        stored_S[regime_name] = S_paths

    except Exception as exc:
        import traceback
        print(f'  ERROR in {regime_name}: {exc}')
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# 6.  Uri stress test
# ─────────────────────────────────────────────────────────────
print('\n--- Uri Stress Test ---')
try:
    np.random.seed(42)
    reg2      = calib_df.loc['regime_2']
    N_URI     = 320   # 80 hours × 4 intervals/hr

    S_uri, V_uri = simulate_paths(
        mu       = float(reg2['mu']),
        alpha    = float(reg2['alpha']),
        sigma    = float(reg2['sigma']),
        lam      = 50.0,
        mu_J     = 500.0,
        p_active = 1.0,          # system continuously active during Uri
        VOLL     = VOLL_MAP['regime_2'],
        N=N, c=C, p_port=P_PORT, K=K,
        n_paths  = N_PATHS,
        n_steps  = N_URI,
        dt       = DT,
    )
    mean_pay_80hr  = float(np.mean(V_uri))
    frac_year      = 80.0 / 8760.0
    annualized_uri = mean_pay_80hr / frac_year

    print(f'  Mean payoff 80-hr window : ${mean_pay_80hr:>12,.2f}')
    print(f'  Annualised estimate      : ${annualized_uri:>12,.2f}')

    uri_df = pd.DataFrame([{
        'scenario':            'Uri (regime_2 params, lam=50, mu_J=500, p_active=1.0)',
        'mean_payoff_80hr':    round(mean_pay_80hr,  2),
        'annualized_estimate': round(annualized_uri, 2),
    }])
    uri_df.to_csv(os.path.join(TAB_DIR, 'uri_stress_test.csv'), index=False)
    print('  Saved uri_stress_test.csv')

except Exception as exc:
    import traceback
    print(f'  ERROR Uri stress test: {exc}')
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# 7.  Save results_summary.csv
# ─────────────────────────────────────────────────────────────
summary_cols = ['regime', 'V0_per_MW_year', 'V_NPV_per_MW_year',
                'option_premium', 'premium_pct', 'ci_lower', 'ci_upper']
summary_rows = [
    {k: results[r][k] for k in summary_cols}
    for r in ['regime_1', 'regime_2', 'regime_3']
    if r in results
]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(TAB_DIR, 'results_summary.csv'), index=False)
print('\nSaved results_summary.csv')

# ─────────────────────────────────────────────────────────────
# 8.  Figures
# ─────────────────────────────────────────────────────────────

# ── fig4: Simulated paths ─────────────────────────────────────
print('\nGenerating fig4_simulated_paths.png ...')
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, r in zip(axes, ['regime_1', 'regime_2', 'regime_3']):
        S_r = stored_S[r]
        x   = np.arange(N_STEPS)

        for i in range(50):
            ax.plot(x, S_r[i], color='#808080', lw=0.25, alpha=0.35)

        mean_path = S_r.mean(axis=0)
        ax.plot(x, mean_path, color='#1A5276', lw=1.6, label='Mean path', zorder=3)

        ax.set_title(REGIME_LABELS[r], fontsize=9)
        ax.set_xlabel('Interval (15-min steps)', fontsize=9)
        ax.set_ylabel('ORDC Adder ($/MWh)', fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(0, N_STEPS)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        'Simulated ORDC Adder Paths – Two-State Model  '
        '(50 of 1,000 paths shown, bold = mean)',
        fontsize=11
    )
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_simulated_paths.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig4_simulated_paths.png')
except Exception as exc:
    print(f'  ERROR fig4: {exc}')

# ── fig5: Payoff distribution ─────────────────────────────────
print('Generating fig5_payoff_distribution.png ...')
try:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, r in zip(axes, ['regime_1', 'regime_2', 'regime_3']):
        if r not in results:
            ax.set_title(f'{r}: failed')
            continue
        res   = results[r]
        V     = res['_V']
        V0    = res['_V0']
        V_NPV = res['_V_NPV']
        prem  = V0 - V_NPV

        counts, _, _ = ax.hist(
            V, bins=60, color=REGIME_COLORS[r],
            edgecolor='#566573', linewidth=0.25, alpha=0.88
        )
        y_max = counts.max()

        ax.axvline(V0,    color='#1A5276', lw=2.0, linestyle='-',
                   label=f'V0 = ${V0/MU_Q:,.0f} /MW-yr')
        ax.axvline(V_NPV, color='#922B21', lw=2.0, linestyle='--',
                   label=f'V_NPV = ${V_NPV/MU_Q:,.0f} /MW-yr')

        # Shade gap between V_NPV and V0
        lo, hi = min(V0, V_NPV), max(V0, V_NPV)
        ax.axvspan(lo, hi, alpha=0.18, color='#8E44AD', zorder=0)

        mid_x = (lo + hi) / 2
        ax.text(
            mid_x, y_max * 0.60,
            f'Option\nPremium\n${prem/MU_Q:,.0f}\n$/MW-yr',
            ha='center', va='center', fontsize=7.5, color='#6C3483',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.80)
        )

        ax.set_title(REGIME_LABELS[r], fontsize=9)
        ax.set_xlabel('Annual Payoff ($)', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(
        'Distribution of Simulated Annual Payoffs  (n_paths = 1,000)\n'
        'Solid = V0 (MC mean)  |  Dashed = V_NPV (historical benchmark)  |  '
        'Shaded gap = option premium',
        fontsize=10
    )
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig5_payoff_distribution.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig5_payoff_distribution.png')
except Exception as exc:
    print(f'  ERROR fig5: {exc}')

# ── fig6: Option value vs NPV grouped bar chart ───────────────
print('Generating fig6_option_vs_npv.png ...')
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    regimes = ['regime_1', 'regime_2', 'regime_3']
    labels  = ['Regime 1', 'Regime 2', 'Regime 3']
    x       = np.arange(len(regimes))
    width   = 0.35

    v0_vals  = [results[r]['V0_per_MW_year']    for r in regimes]
    npv_vals = [results[r]['V_NPV_per_MW_year']  for r in regimes]
    prems    = [results[r]['option_premium']      for r in regimes]

    # Symmetric CI half-width in $/MW-yr
    ci_halves = [results[r]['_ci_half'] / MU_Q for r in regimes]

    bars_v0 = ax.bar(
        x - width / 2, v0_vals, width,
        label='V0 – MC option value',
        color=[REGIME_COLORS[r] for r in regimes],
        edgecolor='#2C3E50', linewidth=0.8,
        yerr=ci_halves, capsize=5,
        error_kw=dict(elinewidth=1.5, ecolor='#2C3E50'),
    )
    ax.bar(
        x + width / 2, npv_vals, width,
        label='V_NPV – historical benchmark',
        color='#AED6F1', edgecolor='#2C3E50', linewidth=0.8, alpha=0.65,
    )

    # Annotate option premium above each bar group
    for i, (v0, npv, prem) in enumerate(zip(v0_vals, npv_vals, prems)):
        top = max(v0, npv) + max(ci_halves[i], 0) + 5
        sign = '+' if prem >= 0 else ''
        ax.text(
            x[i], top,
            f'{sign}${prem:,.0f}\n$/MW-yr',
            ha='center', va='bottom', fontsize=9,
            color='#6C3483', fontweight='bold'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('$/MW-year', fontsize=11)
    ax.set_title(
        'Option Value vs NPV Benchmark by Regime\n'
        'Error bars: 95% CI on V0  |  Two-state zero-inflation model',
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig6_option_vs_npv.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig6_option_vs_npv.png')
except Exception as exc:
    print(f'  ERROR fig6: {exc}')

# ─────────────────────────────────────────────────────────────
# 9.  Console summary
# ─────────────────────────────────────────────────────────────
elapsed = time.time() - t0
pd.set_option('display.float_format', '{:,.2f}'.format)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 130)

print('\n' + '='*75)
print('RESULTS SUMMARY  (all values in $/MW-year)')
print('='*75)
print(summary_df.to_string(index=False))

print('\n' + '='*75)
print('FILES SAVED')
print('='*75)
all_files = [
    os.path.join(TAB_DIR, 'results_summary.csv'),
    os.path.join(TAB_DIR, 'uri_stress_test.csv'),
    os.path.join(FIG_DIR, 'fig4_simulated_paths.png'),
    os.path.join(FIG_DIR, 'fig5_payoff_distribution.png'),
    os.path.join(FIG_DIR, 'fig6_option_vs_npv.png'),
]
n_saved = 0
for f in all_files:
    ok = os.path.exists(f)
    n_saved += int(ok)
    print(f'  [{"OK     " if ok else "MISSING"}]  {os.path.relpath(f, BASE)}')

print(f'\n{n_saved} files saved to outputs/')
print(f'Total runtime: {elapsed:.1f}s')
print('Session 2 complete.  Stopping – do not start sensitivity analysis.')
