"""
Session 1 – ERCOT ORDC Adder Calibration (Two-State Zero-Inflation Model)
Capstone: "The Mispriced Reliability Option: Demand Flexibility and Untapped Alpha
           in Power Markets"

Methodology note
  The ORDC adder is zero-inflated (~90 % of intervals have adder < $1/MWh).
  IQR-based jump detection on the full series collapses robust_std to near zero
  and misclassifies virtually every non-zero residual as a jump.

  Fix: two-state model (Huisman & Mahieu 2003):
    State 0 (quiet):  adder < ACTIVE_THRESHOLD  → S(t) = 0
    State 1 (active): adder >= ACTIVE_THRESHOLD  → OU + jump-diffusion

  Calibration runs on active intervals only, using plain std (not IQR) as the
  dispersion measure, because the active sub-sample is no longer zero-inflated.
  Transition probability p_active is estimated empirically per regime and season.

Outputs
  outputs/tables/calibration_parameters.csv   (overwritten)
  outputs/tables/jump_sensitivity.csv          (overwritten)
  outputs/tables/zero_inflation_stats.csv      (new)
  outputs/tables/transition_probabilities.csv  (new)
  outputs/figures/fig1_timeseries.png          (unchanged)
  outputs/figures/fig2_distributions.png       (unchanged)
  outputs/figures/fig3_qq_plots.png            (overwritten – two-state KS)
  data/processed/seasonal_means.csv            (overwritten – active-only means)
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
DT               = 1.0 / 35040   # 15-min interval as fraction of year
K_ACTIVATION     = 50            # $/MWh portfolio activation cost
VOLL_DEFAULT     = 9_000         # $/MWh – Regimes 1 & 2
VOLL_R3          = 5_000         # $/MWh – Regime 3
ACTIVE_THRESHOLD = 1.0           # $/MWh – below this → State 0 (quiet)
URI_START        = '2021-02-10'
URI_END          = '2021-02-20'
SUMMER_MONTHS    = {6, 7, 8}

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
IS_SYNTHETIC   = False
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
    print(f'  Loaded {len(df):,} rows from {os.path.basename(raw_path)}')
elif os.path.exists(synthetic_path):
    df = _load_raw(synthetic_path)
    IS_SYNTHETIC = True
    print(f'  Loaded synthetic data ({len(df):,} rows)')
else:
    print('  No data found – generating synthetic dataset ...')
    IS_SYNTHETIC = True
    alpha_s, mu_s, sigma_s, lam_s, mu_J_s = 2.5, 35.0, 18.0, 12, 180.0
    dates = pd.date_range('2018-01-01', '2024-12-31 23:45:00', freq='15min')
    n = len(dates)
    S = np.zeros(n); S[0] = mu_s
    Z_s = np.random.randn(n)
    B_s = np.random.binomial(1, lam_s * DT, n)
    Jv  = np.random.exponential(mu_J_s, n)
    for i in range(1, n):
        voll = VOLL_DEFAULT if dates[i].year <= 2021 else VOLL_R3
        S[i] = np.clip(S[i-1] + alpha_s*(mu_s-S[i-1])*DT
                       + sigma_s*np.sqrt(DT)*Z_s[i] + Jv[i]*B_s[i], 0, voll)
    df = pd.DataFrame({'datetime': dates, 'ordc_adder': S})
    df.to_csv(synthetic_path, index=False)
    print(f'  Synthetic data saved ({n:,} rows)')

df = df.sort_values('datetime').reset_index(drop=True)

DATA_NOTE = (
    'DATA: Real ERCOT ORDC Adder (ordc_adder_online)'
    if not IS_SYNTHETIC else
    'DATA: SYNTHETIC – Cartea & Figueroa (2005) parameters'
)

# ─────────────────────────────────────────────────────────────
# 3.  Regime / Uri / state labels
# ─────────────────────────────────────────────────────────────
df['regime'] = 'outside'
for r, info in REGIMES.items():
    mask = (df['datetime'] >= info['start']) & (df['datetime'] <= info['end'])
    df.loc[mask, 'regime'] = r

df['is_uri']   = (df['datetime'] >= URI_START) & (df['datetime'] <= URI_END)
df['hour']     = df['datetime'].dt.hour
df['month']    = df['datetime'].dt.month
df['is_summer']= df['month'].isin(SUMMER_MONTHS)
df['is_active']= df['ordc_adder'] >= ACTIVE_THRESHOLD

print('\nRegime summary:')
for r in REGIMES:
    sub = df[df['regime'] == r]
    pct_a = sub['is_active'].mean() * 100
    print(f'  {r}: {len(sub):,} rows  '
          f'active={pct_a:.1f}%  mean={sub["ordc_adder"].mean():.3f} $/MWh')
print(f'  Uri: {df["is_uri"].sum():,} rows  '
      f'mean={df.loc[df["is_uri"],"ordc_adder"].mean():.3f} $/MWh')

# ─────────────────────────────────────────────────────────────
# 4.  Step 1 – Zero-inflation statistics
# ─────────────────────────────────────────────────────────────
print('\n' + '='*62)
print('STEP 1 – ZERO-INFLATION STATISTICS')
print('='*62)

zi_records = []
for r, info in REGIMES.items():
    sub = df[df['regime'] == r]
    n_tot    = len(sub)
    n_active = sub['is_active'].sum()
    n_quiet  = n_tot - n_active
    act_vals = sub.loc[sub['is_active'], 'ordc_adder']
    zi_records.append({
        'regime':       r,
        'n_total':      n_tot,
        'n_active':     int(n_active),
        'n_quiet':      int(n_quiet),
        'pct_zero':     round((n_quiet / n_tot) * 100, 3),
        'pct_active':   round((n_active / n_tot) * 100, 3),
        'mean_active':  round(float(act_vals.mean()), 4) if n_active > 0 else 0.0,
        'median_active':round(float(act_vals.median()), 4) if n_active > 0 else 0.0,
        'max_adder':    round(float(sub['ordc_adder'].max()), 4),
    })
    print(f'  {r}: quiet={n_quiet/n_tot*100:.1f}%  active={n_active/n_tot*100:.1f}%  '
          f'mean_active={act_vals.mean():.2f}  max={sub["ordc_adder"].max():.2f}')

zi_df = pd.DataFrame(zi_records)
zi_df.to_csv(os.path.join(TAB_DIR, 'zero_inflation_stats.csv'), index=False)
print(f'  Saved zero_inflation_stats.csv')

# ─────────────────────────────────────────────────────────────
# 5.  Step 3 – Transition probabilities
# ─────────────────────────────────────────────────────────────
print('\n' + '='*62)
print('STEP 3 – TRANSITION PROBABILITIES')
print('='*62)

trans_records = []
for r in REGIMES:
    sub = df[df['regime'] == r]
    for season_label, season_mask in [
        ('all',        pd.Series(True, index=sub.index)),
        ('summer',     sub['is_summer']),
        ('non_summer', ~sub['is_summer']),
    ]:
        grp = sub[season_mask]
        if len(grp) == 0:
            continue
        p = float(grp['is_active'].mean())
        trans_records.append({
            'regime': r, 'season': season_label,
            'n_intervals': len(grp),
            'n_active': int(grp['is_active'].sum()),
            'p_active': round(p, 6),
        })
        print(f'  {r} [{season_label:10s}]: p_active={p:.4f}')

trans_df = pd.DataFrame(trans_records)
trans_df.to_csv(os.path.join(TAB_DIR, 'transition_probabilities.csv'), index=False)
print(f'  Saved transition_probabilities.csv')

# ─────────────────────────────────────────────────────────────
# 6.  Step 2 – Calibrate on active intervals only
# ─────────────────────────────────────────────────────────────
print('\n' + '='*62)
print('STEP 2 – CALIBRATION ON ACTIVE INTERVALS (two-state model)')
print('='*62)
print(f'  Active threshold: {ACTIVE_THRESHOLD} $/MWh')
print(f'  Jump detection: plain std of active residuals (not IQR)')
print(f'  Ref: Huisman & Mahieu (2003) – regime-switching electricity prices\n')

seasonal_frames     = []
calibration_results = {}
jump_sens_records   = []

for regime_name, regime_info in REGIMES.items():
    print(f'--- {regime_name} ---')
    try:
        r_all = df[df['regime'] == regime_name].copy()

        # Exclude Uri from calibration in regime 2
        if regime_name == 'regime_2':
            n_uri = r_all['is_uri'].sum()
            r_calib_all = r_all[~r_all['is_uri']].copy()
            print(f'  Excluding Uri: {n_uri:,} rows removed (kept for stress test)')
        else:
            r_calib_all = r_all.copy()

        # ── Filter to active intervals only ──────────────────────
        r_act = r_calib_all[r_calib_all['is_active']].copy()
        n_act = len(r_act)
        n_all = len(r_calib_all)
        years_all    = n_all * DT
        years_active = n_act * DT
        p_active_regime = n_act / n_all
        print(f'  Total (excl Uri): {n_all:,} rows  active: {n_act:,} '
              f'({p_active_regime*100:.1f}%)  years_all: {years_all:.4f}')

        # ── Step 1: Seasonal means on active sub-sample ──────────
        seasonal = (
            r_act.groupby(['hour', 'month'])['ordc_adder']
            .mean()
            .reset_index()
            .rename(columns={'ordc_adder': 'seasonal_mean'})
        )
        seasonal['regime'] = regime_name
        seasonal_frames.append(seasonal)

        r_act = r_act.merge(seasonal[['hour', 'month', 'seasonal_mean']],
                            on=['hour', 'month'], how='left')
        # Fill any missing (hour,month) cells with global active mean
        global_act_mean = float(r_act['ordc_adder'].mean())
        r_act['seasonal_mean'] = r_act['seasonal_mean'].fillna(global_act_mean)
        r_act['residual'] = r_act['ordc_adder'] - r_act['seasonal_mean']
        res = r_act['residual'].values

        # ── Step 2: mu and alpha ─────────────────────────────────
        mu_cal = global_act_mean   # conditional mean of active process

        y_ar = res[1:]
        x_ar = res[:-1]
        ar1  = sm.OLS(y_ar, sm.add_constant(x_ar)).fit()
        b_ar  = float(ar1.params[1])
        b_safe = max(abs(b_ar), 1e-10)
        alpha_cal = max(-np.log(b_safe) / DT, 0.0)
        print(f'  mu={mu_cal:.4f}  b_AR1={b_ar:.6f}  alpha={alpha_cal:.4f}')

        # ── Step 3: Jump detection using plain std ───────────────
        base_std = float(np.std(res, ddof=1))
        print(f'  active residual std={base_std:.4f}  '
              f'(replacing IQR-based robust_std that collapsed to ~0)')

        jump_masks = {}
        for thr in JUMP_THRESHOLDS:
            jmask = np.abs(res) > thr * base_std
            n_j   = int(jmask.sum())
            # lambda expressed per calendar year (not per active-year)
            lam_t  = n_j / years_all
            mu_J_t = float(np.mean(np.abs(res[jmask]))) if n_j > 0 else 0.0
            jump_masks[thr] = jmask
            jump_sens_records.append({
                'regime': regime_name, 'threshold_sigma': thr,
                'n_jumps': n_j,
                'lam_per_year': round(lam_t, 4),
                'mu_J': round(mu_J_t, 4),
            })
            print(f'    thr={thr}σ  n_jumps={n_j:,}  '
                  f'lam={lam_t:.2f}/yr  mu_J={mu_J_t:.2f}')

        # ── Step 4: sigma (3.0σ threshold) ──────────────────────
        jmask_30  = jump_masks[3.0]
        non_jump  = res[~jmask_30]
        sigma_cal = float(np.std(non_jump, ddof=1)) / np.sqrt(DT)
        print(f'  sigma={sigma_cal:.4f}')

        # ── Step 5: lambda and mu_J (3.0σ) ──────────────────────
        n_j30    = int(jmask_30.sum())
        lam_cal  = n_j30 / years_all      # per calendar year
        mu_J_cal = float(np.mean(np.abs(res[jmask_30]))) if n_j30 > 0 else 0.0
        print(f'  lam={lam_cal:.4f}/yr  mu_J={mu_J_cal:.4f}  n_jumps={n_j30:,}')

        # ── Step 6: KS validation – two-state simulation ─────────
        # Simulate full distribution: draw state each step, then OU+jump if active
        VOLL = regime_info['VOLL']
        np.random.seed(42)
        n_ks, s_ks = 1000, 100

        S_ks  = np.full(n_ks, mu_cal)
        Z_ks  = np.random.randn(n_ks, s_ks)
        B_ks  = np.random.binomial(1, max(lam_cal, 0) * DT, (n_ks, s_ks))
        Jv_ks = np.random.exponential(max(mu_J_cal, 1.0), (n_ks, s_ks))
        U_ks  = np.random.uniform(size=(n_ks, s_ks))   # state draws

        sim_snap = []
        for step in range(s_ks):
            active_mask = U_ks[:, step] < p_active_regime
            # Evolve OU+jump only for active paths
            S_new = np.where(
                active_mask,
                np.clip(
                    S_ks
                    + alpha_cal * (mu_cal - S_ks) * DT
                    + sigma_cal * np.sqrt(DT) * Z_ks[:, step]
                    + Jv_ks[:, step] * B_ks[:, step],
                    0, VOLL
                ),
                0.0   # quiet state
            )
            S_ks = S_new
            sim_snap.append(S_ks.copy())

        sim_vals_ks = np.concatenate(sim_snap)
        # Compare against FULL historical regime (includes zeros)
        hist_vals_full = r_all['ordc_adder'].values
        ks_stat, ks_pval = stats.ks_2samp(hist_vals_full, sim_vals_ks)
        print(f'  KS (full dist, two-state): stat={ks_stat:.4f}  p={ks_pval:.4f}')

        calibration_results[regime_name] = {
            'regime':           regime_name,
            'mu':               round(mu_cal,         4),
            'alpha':            round(alpha_cal,       4),
            'sigma':            round(sigma_cal,       4),
            'lam':              round(lam_cal,         4),
            'mu_J':             round(mu_J_cal,        4),
            'n_jumps':          n_j30,
            'years_in_regime':  round(years_all,       4),
            'p_active':         round(p_active_regime, 6),
            'ks_stat':          round(ks_stat,         4),
            'ks_pvalue':        round(ks_pval,         4),
            # private – figures only
            '_sim_vals':        sim_vals_ks,
            '_hist_vals_full':  hist_vals_full,
        }
        print()

    except Exception as exc:
        import traceback
        print(f'  ERROR in {regime_name}: {exc}')
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# 7.  Save processed seasonal means
# ─────────────────────────────────────────────────────────────
seasonal_df = pd.concat(seasonal_frames, ignore_index=True)
seasonal_df.to_csv(os.path.join(PROC_DIR, 'seasonal_means.csv'), index=False)
print(f'Seasonal means saved ({len(seasonal_df)} rows, active intervals only)')

# ─────────────────────────────────────────────────────────────
# 8.  Save tables
# ─────────────────────────────────────────────────────────────
table_cols = ['regime', 'mu', 'alpha', 'sigma', 'lam', 'mu_J',
              'n_jumps', 'years_in_regime', 'p_active', 'ks_stat', 'ks_pvalue']

calib_rows = [
    {k: calibration_results[r][k] for k in table_cols}
    for r in REGIMES if r in calibration_results
]
calib_df = pd.DataFrame(calib_rows)
calib_df.to_csv(os.path.join(TAB_DIR, 'calibration_parameters.csv'), index=False)
print(f'calibration_parameters.csv saved ({len(calib_df)} regimes)')

jump_df = pd.DataFrame(jump_sens_records)
jump_df.to_csv(os.path.join(TAB_DIR, 'jump_sensitivity.csv'), index=False)
print(f'jump_sensitivity.csv saved ({len(jump_df)} rows)')

# ─────────────────────────────────────────────────────────────
# 9.  Figures
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
            plt.Rectangle((0, 0), 1, 1, fc=info['color'], alpha=0.5,
                          label=info['label'])
        )
    ax.axvspan(pd.Timestamp(URI_START), pd.Timestamp(URI_END),
               alpha=0.60, color='#E74C3C', zorder=3)

    uri_patch = plt.Rectangle((0, 0), 1, 1, fc='#E74C3C', alpha=0.6,
                               label='Winter Storm Uri (Feb 10–20 2021)')
    handles = regime_patches + [uri_patch]
    ax.legend(handles=handles, loc='upper left', fontsize=8)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('ORDC Adder ($/MWh)', fontsize=11)
    ax.set_title(f'ERCOT Real-Time ORDC Adder (Online) 2018–2024\n{DATA_NOTE}',
                 fontsize=12)
    ax.set_xlim(df['datetime'].iloc[0], df['datetime'].iloc[-1])
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_timeseries.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig1_timeseries.png')
except Exception as exc:
    print(f'  ERROR fig1: {exc}')

# ── fig2: Distribution histograms ────────────────────────────
print('Generating fig2_distributions.png ...')
try:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (r, info) in zip(axes, REGIMES.items()):
        vals = df[df['regime'] == r]['ordc_adder'].values
        ax.hist(vals, bins=80, color=info['color'],
                edgecolor='#566573', linewidth=0.25, alpha=0.90)
        ax.axvline(K_ACTIVATION, color='#1A5276', linestyle='--', lw=1.8,
                   label=f'K = {K_ACTIVATION} $/MWh')
        ax.set_yscale('log')
        ax.set_xlabel('ORDC Adder ($/MWh)', fontsize=10)
        ax.set_ylabel('Count (log scale)', fontsize=10)
        ax.set_title(f'{info["label"]}\nn = {len(vals):,}', fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(f'ORDC Adder Distributions by Regime\n{DATA_NOTE}', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_distributions.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig2_distributions.png')
except Exception as exc:
    print(f'  ERROR fig2: {exc}')

# ── fig3: QQ plots – two-state simulation vs full historical ──
print('Generating fig3_qq_plots.png ...')
try:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, r in zip(axes, REGIMES.keys()):
        if r not in calibration_results:
            ax.set_title(f'{r}: calibration failed')
            continue
        cr      = calibration_results[r]
        h_vals  = cr['_hist_vals_full']
        s_vals  = cr['_sim_vals']
        ks_stat = cr['ks_stat']
        ks_pval = cr['ks_pvalue']
        p_act   = cr['p_active']

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
            f'KS = {ks_stat:.3f}  p = {ks_pval:.4f}\np_active = {p_act:.3f}',
            xy=(0.05, 0.85), xycoords='axes fraction', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85)
        )
        ax.legend(fontsize=7)

    fig.suptitle(
        f'QQ Plots – Two-State Simulated vs Full Historical ORDC Adder\n'
        f'{DATA_NOTE}\n'
        f'Two-state model (Huisman & Mahieu 2003): State 0 = quiet (S=0), '
        f'State 1 = active OU+jump. Quick validation: 1000 paths × 100 steps.',
        fontsize=8.5
    )
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_qq_plots.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig3_qq_plots.png')
except Exception as exc:
    print(f'  ERROR fig3: {exc}')

# ─────────────────────────────────────────────────────────────
# 10.  Console summary
# ─────────────────────────────────────────────────────────────
elapsed = time.time() - t0
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 130)

print('\n' + '='*70)
print('CALIBRATION PARAMETERS (two-state model, active intervals only)')
print('='*70)
print(calib_df.to_string(index=False))

print('\n' + '='*70)
print('JUMP SENSITIVITY (plain std threshold, active sub-sample)')
print('='*70)
print(jump_df.to_string(index=False))

print('\n' + '='*70)
print('ZERO-INFLATION STATISTICS')
print('='*70)
print(zi_df[['regime','pct_zero','pct_active','mean_active','max_adder']].to_string(index=False))

print('\n' + '='*70)
print('TRANSITION PROBABILITIES')
print('='*70)
print(trans_df.to_string(index=False))

print('\n' + '='*70)
print('FILES SAVED')
print('='*70)
all_files = [
    os.path.join(TAB_DIR, 'calibration_parameters.csv'),
    os.path.join(TAB_DIR, 'jump_sensitivity.csv'),
    os.path.join(TAB_DIR, 'zero_inflation_stats.csv'),
    os.path.join(TAB_DIR, 'transition_probabilities.csv'),
    os.path.join(FIG_DIR, 'fig1_timeseries.png'),
    os.path.join(FIG_DIR, 'fig2_distributions.png'),
    os.path.join(FIG_DIR, 'fig3_qq_plots.png'),
]
for f in all_files:
    status = 'OK     ' if os.path.exists(f) else 'MISSING'
    print(f'  [{status}]  {os.path.relpath(f, BASE)}')

print(f'\n{DATA_NOTE}')
print(f'Session 1 (two-state recalibration) complete.  Runtime: {elapsed:.1f}s')
print('Stopping here – do not start the simulation.')
