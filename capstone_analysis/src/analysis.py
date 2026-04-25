"""
Session 1 – ERCOT ORDC Adder Calibration
Capstone: "The Mispriced Reliability Option: Demand Flexibility and Untapped Alpha in Power Markets"

Outputs
  outputs/tables/calibration_parameters.csv
  outputs/tables/jump_sensitivity.csv
  outputs/figures/fig1_timeseries.png
  outputs/figures/fig2_distributions.png
  outputs/figures/fig3_qq_plots.png
  data/processed/seasonal_means.csv
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import warnings
import time

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

# ─────────────────────────────────────────────────────────────
# 0.  Paths
# ─────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE, 'data', 'raw')
PROC_DIR = os.path.join(BASE, 'data', 'processed')
FIG_DIR  = os.path.join(BASE, 'outputs', 'figures')
TAB_DIR  = os.path.join(BASE, 'outputs', 'tables')

for d in [RAW_DIR, PROC_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1.  Constants
# ─────────────────────────────────────────────────────────────
DT           = 1.0 / 35040      # 15-min interval as fraction of year
K_ACTIVATION = 50               # $/MWh
VOLL_DEFAULT = 9_000            # $/MWh – Regimes 1 & 2
VOLL_R3      = 5_000            # $/MWh – Regime 3
URI_START    = '2021-02-10'
URI_END      = '2021-02-20'

REGIMES = {
    'regime_1': {
        'start': '2018-01-01', 'end': '2019-02-28',
        'VOLL': VOLL_DEFAULT,
        'color': '#AED6F1',
        'label': 'Regime 1 (Jan 2018 – Feb 2019)',
    },
    'regime_2': {
        'start': '2019-03-01', 'end': '2021-12-31',
        'VOLL': VOLL_DEFAULT,
        'color': '#A9DFBF',
        'label': 'Regime 2 (Mar 2019 – Dec 2021)',
    },
    'regime_3': {
        'start': '2022-01-01', 'end': '2024-12-31',
        'VOLL': VOLL_R3,
        'color': '#F9E79F',
        'label': 'Regime 3 (Jan 2022 – Dec 2024)',
    },
}

JUMP_THRESHOLDS = [2.5, 3.0, 3.5]

# ─────────────────────────────────────────────────────────────
# 2.  Load data
# ─────────────────────────────────────────────────────────────
print('Loading ERCOT ORDC adder data ...')
IS_SYNTHETIC = False
raw_path       = os.path.join(RAW_DIR, 'ordc_data.csv')
synthetic_path = os.path.join(RAW_DIR, 'synthetic_ordc.csv')

def _load_raw(path):
    d = pd.read_csv(path)
    d['datetime'] = pd.to_datetime(d['datetime'])
    if 'ordc_adder_online' in d.columns:
        d = d.rename(columns={'ordc_adder_online': 'ordc_adder'})
    elif 'ordc_adder' not in d.columns:
        num = d.select_dtypes(include=[np.number]).columns[0]
        d = d.rename(columns={num: 'ordc_adder'})
    return d[['datetime', 'ordc_adder']]

if os.path.exists(raw_path):
    df = _load_raw(raw_path)
    print(f'  Loaded {len(df):,} rows from {raw_path}')
elif os.path.exists(synthetic_path):
    df = _load_raw(synthetic_path)
    IS_SYNTHETIC = True
    print(f'  Loaded synthetic data from {synthetic_path}')
else:
    print('  No data found – generating synthetic ERCOT-like dataset ...')
    IS_SYNTHETIC = True
    alpha_s, mu_s, sigma_s = 2.5, 35.0, 18.0
    lam_s, mu_J_s = 12, 180.0

    dates = pd.date_range('2018-01-01', '2024-12-31 23:45:00', freq='15min')
    n     = len(dates)
    S     = np.zeros(n)
    S[0]  = mu_s
    Z_s   = np.random.randn(n)
    B_s   = np.random.binomial(1, lam_s * DT, n)
    Jv    = np.random.exponential(mu_J_s, n)

    for i in range(1, n):
        voll  = VOLL_DEFAULT if dates[i].year <= 2021 else VOLL_R3
        S[i]  = S[i-1] + alpha_s*(mu_s - S[i-1])*DT + sigma_s*np.sqrt(DT)*Z_s[i] + Jv[i]*B_s[i]
        S[i]  = float(np.clip(S[i], 0, voll))

    df = pd.DataFrame({'datetime': dates, 'ordc_adder': S})
    df.to_csv(synthetic_path, index=False)
    print(f'  Synthetic data saved ({len(df):,} rows)')

df = df.sort_values('datetime').reset_index(drop=True)
DATA_NOTE = (
    'DATA: Real ERCOT ORDC Adder (ordc_adder_online)'
    if not IS_SYNTHETIC else
    'DATA: SYNTHETIC – mean-reversion jump-diffusion (Cartea & Figueroa 2005 params)'
)

# ─────────────────────────────────────────────────────────────
# 3.  Regime and Uri labels
# ─────────────────────────────────────────────────────────────
df['regime'] = 'outside'
for r, info in REGIMES.items():
    mask = (df['datetime'] >= info['start']) & (df['datetime'] <= info['end'])
    df.loc[mask, 'regime'] = r

df['is_uri'] = (df['datetime'] >= URI_START) & (df['datetime'] <= URI_END)
df['hour']   = df['datetime'].dt.hour
df['month']  = df['datetime'].dt.month

print('\nRegime row counts and mean adder:')
for r in REGIMES:
    sub = df[df['regime'] == r]
    print(f'  {r}: {len(sub):,} rows  mean={sub["ordc_adder"].mean():.3f} $/MWh')
print(f'  Uri period: {df["is_uri"].sum():,} rows  mean={df.loc[df["is_uri"], "ordc_adder"].mean():.3f} $/MWh')

# ─────────────────────────────────────────────────────────────
# 4.  Calibration
# ─────────────────────────────────────────────────────────────
print('\n' + '='*62)
print('CALIBRATION')
print('='*62)

seasonal_frames     = []
calibration_results = {}
jump_sens_records   = []

for regime_name, regime_info in REGIMES.items():
    print(f'\n--- {regime_name} ---')
    try:
        r_all   = df[df['regime'] == regime_name].copy()
        # Exclude Uri from calibration for regime 2
        r_calib = r_all[~r_all['is_uri']].copy() if regime_name == 'regime_2' else r_all.copy()
        if regime_name == 'regime_2':
            n_uri = r_all['is_uri'].sum()
            print(f'  Excluding Uri: {n_uri:,} rows removed (kept for stress test)')

        n_rows         = len(r_calib)
        years_in_regime = n_rows * DT
        print(f'  Calibration rows: {n_rows:,}  years: {years_in_regime:.4f}')

        # ── Step 1: Seasonal means ──────────────────────────────
        seasonal = (
            r_calib.groupby(['hour', 'month'])['ordc_adder']
            .mean()
            .reset_index()
            .rename(columns={'ordc_adder': 'seasonal_mean'})
        )
        seasonal['regime'] = regime_name
        seasonal_frames.append(seasonal)

        r_calib = r_calib.merge(seasonal[['hour', 'month', 'seasonal_mean']],
                                on=['hour', 'month'], how='left')
        r_calib['residual'] = r_calib['ordc_adder'] - r_calib['seasonal_mean']
        res = r_calib['residual'].values

        # ── Step 2: mu and alpha ────────────────────────────────
        mu_cal = float(r_calib['ordc_adder'].mean())

        y_ar = res[1:]
        x_ar = res[:-1]
        ar1  = sm.OLS(y_ar, sm.add_constant(x_ar)).fit()
        b_ar = float(ar1.params[1])
        b_safe = max(abs(b_ar), 1e-10)
        alpha_cal = max(-np.log(b_safe) / DT, 0.0)
        print(f'  mu={mu_cal:.4f}  b_AR1={b_ar:.6f}  alpha={alpha_cal:.4f}')

        # ── Step 3: Jump detection – all three thresholds ───────
        q75, q25  = np.percentile(res, 75), np.percentile(res, 25)
        robust_std = (q75 - q25) / 1.349
        print(f'  robust_std={robust_std:.4f}')

        jump_masks = {}
        for thr in JUMP_THRESHOLDS:
            jmask  = np.abs(res) > thr * robust_std
            n_j    = int(jmask.sum())
            lam_t  = n_j / years_in_regime
            mu_J_t = float(np.mean(np.abs(res[jmask]))) if n_j > 0 else 0.0
            jump_masks[thr] = jmask
            jump_sens_records.append({
                'regime': regime_name,
                'threshold_sigma': thr,
                'n_jumps': n_j,
                'lam': round(lam_t, 4),
                'mu_J': round(mu_J_t, 4),
            })
            print(f'    thr={thr}σ  n_jumps={n_j:,}  lam={lam_t:.2f}/yr  mu_J={mu_J_t:.2f}')

        # ── Step 4: sigma (using 3.0σ threshold) ────────────────
        jmask_30   = jump_masks[3.0]
        non_jump   = res[~jmask_30]
        sigma_cal  = float(np.std(non_jump)) / np.sqrt(DT)
        print(f'  sigma={sigma_cal:.4f}')

        # ── Step 5: lambda and mu_J (3.0σ threshold) ────────────
        n_j30    = int(jmask_30.sum())
        lam_cal  = n_j30 / years_in_regime
        mu_J_cal = float(np.mean(np.abs(res[jmask_30]))) if n_j30 > 0 else 0.0
        print(f'  lam={lam_cal:.4f}/yr  mu_J={mu_J_cal:.4f}  n_jumps={n_j30:,}')

        # ── Step 6: KS validation (quick: 1000 paths × 100 steps) ─
        VOLL = regime_info['VOLL']
        np.random.seed(42)
        n_ks, s_ks = 1000, 100
        S_ks  = np.full(n_ks, mu_cal)
        Z_ks  = np.random.randn(n_ks, s_ks)
        B_ks  = np.random.binomial(1, max(lam_cal, 0) * DT, (n_ks, s_ks))
        Jv_ks = np.random.exponential(max(mu_J_cal, 1.0), (n_ks, s_ks))
        sim_snap = []

        for step in range(s_ks):
            S_ks  = (S_ks
                     + alpha_cal * (mu_cal - S_ks) * DT
                     + sigma_cal * np.sqrt(DT) * Z_ks[:, step]
                     + Jv_ks[:, step] * B_ks[:, step])
            S_ks  = np.clip(S_ks, 0, VOLL)
            sim_snap.append(S_ks.copy())

        sim_vals_ks = np.concatenate(sim_snap)
        hist_vals   = r_calib['ordc_adder'].values
        ks_stat, ks_pval = stats.ks_2samp(hist_vals, sim_vals_ks)
        print(f'  KS: stat={ks_stat:.4f}  p={ks_pval:.4f}')

        calibration_results[regime_name] = {
            'regime':          regime_name,
            'mu':              round(mu_cal,       4),
            'alpha':           round(alpha_cal,    4),
            'sigma':           round(sigma_cal,    4),
            'lam':             round(lam_cal,      4),
            'mu_J':            round(mu_J_cal,     4),
            'n_jumps':         n_j30,
            'years_in_regime': round(years_in_regime, 4),
            'ks_stat':         round(ks_stat,      4),
            'ks_pvalue':       round(ks_pval,      4),
            # private – used only by figures below
            '_res':        res,
            '_jmask_30':   jmask_30,
            '_sim_vals':   sim_vals_ks,
            '_hist_vals':  hist_vals,
        }

    except Exception as exc:
        import traceback
        print(f'  ERROR in {regime_name}: {exc}')
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# 5.  Save processed seasonal means
# ─────────────────────────────────────────────────────────────
seasonal_df = pd.concat(seasonal_frames, ignore_index=True)
seasonal_df.to_csv(os.path.join(PROC_DIR, 'seasonal_means.csv'), index=False)
print(f'\nSeasonal means saved: {PROC_DIR}/seasonal_means.csv ({len(seasonal_df)} rows)')

# ─────────────────────────────────────────────────────────────
# 6.  Save tables
# ─────────────────────────────────────────────────────────────
table_cols = ['regime', 'mu', 'alpha', 'sigma', 'lam', 'mu_J',
              'n_jumps', 'years_in_regime', 'ks_stat', 'ks_pvalue']

calib_rows = []
for r in REGIMES:
    if r in calibration_results:
        calib_rows.append({k: calibration_results[r][k] for k in table_cols})

calib_df = pd.DataFrame(calib_rows)
calib_df.to_csv(os.path.join(TAB_DIR, 'calibration_parameters.csv'), index=False)
print(f'calibration_parameters.csv  ({len(calib_df)} regimes)')

jump_df = pd.DataFrame(jump_sens_records)
jump_df.to_csv(os.path.join(TAB_DIR, 'jump_sensitivity.csv'), index=False)
print(f'jump_sensitivity.csv  ({len(jump_df)} rows)')

# ─────────────────────────────────────────────────────────────
# 7.  Figures
# ─────────────────────────────────────────────────────────────

# ── fig1: Full time series ────────────────────────────────────
print('\nGenerating fig1_timeseries.png ...')
try:
    fig, ax = plt.subplots(figsize=(17, 5))
    ax.plot(df['datetime'], df['ordc_adder'],
            color='#1A252F', lw=0.25, alpha=0.65, zorder=2)

    regime_patches = []
    for r, info in REGIMES.items():
        ax.axvspan(pd.Timestamp(info['start']), pd.Timestamp(info['end']),
                   alpha=0.20, color=info['color'], zorder=1)
        regime_patches.append(
            plt.Rectangle((0, 0), 1, 1, fc=info['color'], alpha=0.5, label=info['label'])
        )

    ax.axvspan(pd.Timestamp(URI_START), pd.Timestamp(URI_END),
               alpha=0.60, color='#E74C3C', zorder=3,
               label='Winter Storm Uri (Feb 10–20, 2021)')

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('ORDC Adder ($/MWh)', fontsize=11)
    ax.set_title(
        f'ERCOT Real-Time ORDC Adder (Online) 2018–2024\n{DATA_NOTE}',
        fontsize=12
    )
    ax.set_xlim(df['datetime'].iloc[0], df['datetime'].iloc[-1])
    ax.set_ylim(bottom=0)

    handles = regime_patches + [
        plt.Rectangle((0, 0), 1, 1, fc='#E74C3C', alpha=0.6,
                       label='Winter Storm Uri (Feb 10–20, 2021)')
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_timeseries.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig1_timeseries.png')
except Exception as exc:
    print(f'  ERROR fig1: {exc}')

# ── fig2: Distribution histograms by regime ───────────────────
print('Generating fig2_distributions.png ...')
try:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (r, info) in zip(axes, REGIMES.items()):
        vals = df[df['regime'] == r]['ordc_adder'].values
        # Use 80 linear bins; log scale on y-axis shows tails
        ax.hist(vals, bins=80, color=info['color'],
                edgecolor='#566573', linewidth=0.25, alpha=0.90)
        ax.axvline(K_ACTIVATION, color='#1A5276', linestyle='--', lw=1.8,
                   label=f'K = {K_ACTIVATION} $/MWh')
        ax.set_yscale('log')
        ax.set_xlabel('ORDC Adder ($/MWh)', fontsize=10)
        ax.set_ylabel('Count (log scale)', fontsize=10)
        ax.set_title(f'{info["label"]}\nn = {len(vals):,}', fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        f'ORDC Adder Distributions by Regime (log Y-axis)\n{DATA_NOTE}',
        fontsize=11
    )
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_distributions.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig2_distributions.png')
except Exception as exc:
    print(f'  ERROR fig2: {exc}')

# ── fig3: QQ plots (simulated vs historical) ──────────────────
print('Generating fig3_qq_plots.png ...')
try:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, r in zip(axes, REGIMES.keys()):
        if r not in calibration_results:
            ax.set_title(f'{r}: calibration failed')
            continue
        cr       = calibration_results[r]
        h_vals   = cr['_hist_vals']
        s_vals   = cr['_sim_vals']
        ks_stat  = cr['ks_stat']
        ks_pval  = cr['ks_pvalue']

        percs  = np.linspace(1, 99, 199)
        q_hist = np.percentile(h_vals, percs)
        q_sim  = np.percentile(s_vals, percs)

        ax.scatter(q_hist, q_sim, s=16, alpha=0.75,
                   color=REGIMES[r]['color'], edgecolors='#2C3E50', linewidth=0.4)
        lo = min(q_hist.min(), q_sim.min())
        hi = max(q_hist.max(), q_sim.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.4, label='45° line')
        ax.set_xlabel('Historical Quantiles ($/MWh)', fontsize=9)
        ax.set_ylabel('Simulated Quantiles ($/MWh)', fontsize=9)
        ax.set_title(REGIMES[r]['label'], fontsize=9)
        ax.annotate(
            f'KS stat = {ks_stat:.3f}\np-value  = {ks_pval:.4f}',
            xy=(0.05, 0.88), xycoords='axes fraction', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85)
        )
        ax.legend(fontsize=7)

    fig.suptitle(
        f'QQ Plots – Simulated vs Historical ORDC Adder\n'
        f'{DATA_NOTE}\n'
        f'Note: Quick validation (1000 paths × 100 steps). '
        f'Full MLE calibration would improve fit at extremes.',
        fontsize=9
    )
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_qq_plots.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig3_qq_plots.png')
except Exception as exc:
    print(f'  ERROR fig3: {exc}')

# ─────────────────────────────────────────────────────────────
# 8.  Console summary
# ─────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print('\n' + '='*70)
print('CALIBRATION PARAMETERS')
print('='*70)
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)
print(calib_df.to_string(index=False))

print('\n' + '='*70)
print('JUMP SENSITIVITY (threshold comparison)')
print('='*70)
print(jump_df.to_string(index=False))

print('\n' + '='*70)
print('FILES SAVED')
print('='*70)
saved_tables  = [
    os.path.join(TAB_DIR, 'calibration_parameters.csv'),
    os.path.join(TAB_DIR, 'jump_sensitivity.csv'),
]
saved_figures = [
    os.path.join(FIG_DIR, 'fig1_timeseries.png'),
    os.path.join(FIG_DIR, 'fig2_distributions.png'),
    os.path.join(FIG_DIR, 'fig3_qq_plots.png'),
]
for f in saved_tables + saved_figures:
    status = 'OK     ' if os.path.exists(f) else 'MISSING'
    print(f'  [{status}]  {os.path.relpath(f, BASE)}')

print(f'\n{DATA_NOTE}')
print(f'Session 1 calibration complete.  Runtime: {elapsed:.1f}s')
print('Stopping here – do not start the simulation.')
