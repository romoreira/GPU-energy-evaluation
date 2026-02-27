"""
plot_energy_lora.py
===================
Publication-quality figures for the 2² Partial Factorial Analysis of
LoRA fine-tuning energy consumption on three GPU platforms.

ACM SIGCOMM / NSDI style — font size 16, serif typography, 300 DPI png.

Directory structure expected
-----------------------------
Resultados/
├── Bruto/
│   ├── A100/
│   │   ├── results_accs_experimentos_muitos_passos_135m_complete.csv
│   │   ├── results_accs_experimentos_muitos_passos_360m_complete.csv
│   │   ├── results_accs_experimentos_poucos_passos_135m_complete.csv
│   │   ├── results_accs_experimentos_poucos_passos_360m_complete.csv
│   │   └── results_f1_* (same naming pattern)
│   ├── A40/   (same naming pattern)
│   └── RTX6000/ (same naming pattern)
├── A100/
│   └── Emissão/
│       ├── Muitos Passos 135/emissions/emissions.csv
│       ├── Muitos Passos 350/emissions/emissions.csv
│       ├── Poucos Passos 135/emissions/emissions.csv
│       └── Poucos Passos 350/emissions/emissions.csv
├── A40/
│   └── Emissão/  (same sub-structure)
└── RTX6000/
    └── Emissão/  (same sub-structure)

Factorial Analysis Values
--------------------------
Fill in factorial_data_per_gpu below with values extracted from your
XLSX files for each GPU × response variable combination.

Canonical experiment order (2² design):
  exp[0]: A=-1 B=-1  →  135M / Few steps  (10)
  exp[1]: A=+1 B=-1  →  360M / Few steps  (10)
  exp[2]: A=-1 B=+1  →  135M / Many steps (100)
  exp[3]: A=+1 B=+1  →  360M / Many steps (100)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# 0.  GLOBAL STYLE — ACM SIGCOMM / NSDI
# ══════════════════════════════════════════════════════════════════
BASE_FONT   = 16
SMALL_FONT  = 14
TITLE_FONT  = 17
LEGEND_FONT = 14

GPU_COLORS = {
    "A100":    "#1a4e8a",
    "A40":     "#c0392b",
    "RTX6000": "#2e7d32",
}

C_135_FEW  = "#1a4e8a"
C_360_FEW  = "#c0392b"
C_135_MANY = "#2e7d32"
C_360_MANY = "#e67e22"
CGRAY      = "#6c757d"
CLGRAY     = "#dee2e6"

HATCHES  = ["", "///", "...", "xxx"]
MARKERS  = ["o", "s", "^", "D"]

CONFIGS_ORDER  = [("135M", "Few"), ("360M", "Few"), ("135M", "Many"), ("360M", "Many")]
CONFIG_COLORS  = [C_135_FEW, C_360_FEW, C_135_MANY, C_360_MANY]
CONFIG_MARKERS = MARKERS

FACTOR_A_LABEL = "Model Size"
FACTOR_B_LABEL = "Optimization Steps"
LEVEL_A = {"low": "135M",        "high": "360M"}
LEVEL_B = {"low": "Few (10)",    "high": "Many (100)"}

GPUS = ["A100", "A40", "RTX6000"]

mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Linux Libertine O", "Linux Libertine", "Georgia",
                           "Times New Roman", "DejaVu Serif"],
    "font.size":          BASE_FONT,
    "axes.titlesize":     TITLE_FONT,
    "axes.labelsize":     BASE_FONT,
    "xtick.labelsize":    SMALL_FONT,
    "ytick.labelsize":    SMALL_FONT,
    "legend.fontsize":    LEGEND_FONT,
    "figure.titlesize":   TITLE_FONT,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     1.2,
    "axes.grid":          True,
    "grid.color":         CLGRAY,
    "grid.linewidth":     0.7,
    "grid.linestyle":     "--",
    "axes.axisbelow":     True,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "legend.framealpha":  0.92,
    "legend.edgecolor":   CGRAY,
    "legend.borderpad":   0.5,
})

OUT_DIR = "figures_paper"
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name, tight=True):
    path = os.path.join(OUT_DIR, name)
    if tight:
        fig.tight_layout()
    fig.savefig(path)
    print(f"  ✓  {path}")
    plt.close(fig)


def cfg_label(model, steps):
    p = "10" if steps == "Few" else "100"
    return f"{model} / {p} steps"


def safe_name(s):
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace(
        "/", "").replace("₂", "2").replace("²", "2")


def add_bar_labels(ax, bars, fmt="{:.3f}", fontsize=SMALL_FONT - 1,
                   pad_frac=0.015, color_inside="white", threshold_frac=0.55):
    ymax = max((b.get_height() for b in bars), default=1) or 1
    for bar in bars:
        h = bar.get_height()
        if h == 0:
            continue
        inside = h / ymax > threshold_frac
        ypos   = h / 2      if inside else h + ymax * pad_frac
        va     = "center"   if inside else "bottom"
        col    = color_inside if inside else "#222"
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                fmt.format(h), ha="center", va=va,
                fontsize=fontsize, color=col)


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTION — UPDATED FOR BBOX TEXT (LEGIBILITY FIXED)
# ══════════════════════════════════════════════════════════════════
def add_bar_labels(ax, bars, fmt="{:.3f}", fontsize=SMALL_FONT - 1,
                    pad_frac=0.015, color_inside="white", threshold_frac=0.55,
                    with_bbox=False):
    """
    Adiciona rótulos de texto às barras.
    
    Novos parâmetros:
    - with_bbox (bool): Se True, desenha uma caixa de texto com fundo branco
                        e borda transparente ao redor de cada rótulo para
                        garantir a legibilidade.
    """
    ymax = max((b.get_height() for b in bars), default=1) or 1
    
    # Configuração da caixa de texto (bbox)
    bbox_props = None
    if with_bbox:
        bbox_props = dict(
            boxstyle="round,pad=0.2", # Estilo da caixa (arredondada, com preenchimento)
            facecolor="white",        # Cor de fundo da caixa
            edgecolor="none",         # Cor da borda da caixa (transparente)
            alpha=0.85                # Opacidade da caixa (levemente transparente para ver o fundo)
        )
        
    for bar in bars:
        h = bar.get_height()
        if h == 0:
            continue
            
        # Determina se o texto fica dentro ou acima da barra com base no limiar
        inside = h / ymax > threshold_frac
        ypos   = h / 2      if inside else h + ymax * pad_frac
        va     = "center"   if inside else "bottom"
        
        # Cor do texto (preto se with_bbox=True para consistência)
        text_color = "black" if with_bbox else (color_inside if inside else "#222")
        
        # Adiciona o texto ao gráfico com a configuração da bbox
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                fmt.format(h), ha="center", va=va,
                fontsize=fontsize, color=text_color,
                bbox=bbox_props) # Aplica a bbox aqui

# ══════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════

BASE_RESULTS = "Resultados"

EMISSION_DIRS = {
    "A100":    os.path.join(BASE_RESULTS, "A100",    "Emissão"),
    "A40":     os.path.join(BASE_RESULTS, "A40",     "Emissão"),
    "RTX6000": os.path.join(BASE_RESULTS, "RTX6000", "Emissão"),
}

EMISSION_SUBFOLDERS = {
    ("135M", "Many"): "Muitos Passos 135",
    ("360M", "Many"): "Muitos Passos 350",
    ("135M", "Few"):  "Poucos Passos 135",
    ("360M", "Few"):  "Poucos Passos 350",
}

records_em = []
for gpu, base_em in EMISSION_DIRS.items():
    for (model, steps), subfolder in EMISSION_SUBFOLDERS.items():
        path = os.path.join(base_em, subfolder, "emissions", "emissions.csv")
        if os.path.exists(path):
            df_raw = pd.read_csv(path)
            row = df_raw.iloc[-1]
            records_em.append({
                "gpu":             gpu,
                "model":           model,
                "steps":           steps,
                "gpu_power_w":     float(row["gpu_power"]),
                "energy_kwh":      float(row["energy_consumed"]),
                "emissions_kgco2": float(row["emissions"]),
                "duration_s":      float(row["duration"]),
            })
        else:
            print(f"  ⚠  Not found: {path}")

df_em = pd.DataFrame(records_em)
print("\n── Emissions data ──\n", df_em.to_string(index=False) if not df_em.empty else "(empty)")

# ── 1b. Accuracy / F1 raw results ────────────────────────────────
BASE_BRUTO = os.path.join(BASE_RESULTS, "Bruto")

ACC_FNAMES = {
    ("135M", "Many"): "results_accs_experimentos_muitos_passos_135m_complete.csv",
    ("360M", "Many"): "results_accs_experimentos_muitos_passos_360m_complete.csv",
    ("135M", "Few"):  "results_accs_experimentos_poucos_passos_135m_complete.csv",
    ("360M", "Few"):  "results_accs_experimentos_poucos_passos_360m_complete.csv",
}
F1_FNAMES = {k: v.replace("accs", "f1") for k, v in ACC_FNAMES.items()}


def load_bruto(gpu, fnames):
    frames = []
    for (model, steps), fname in fnames.items():
        path = os.path.join(BASE_BRUTO, gpu, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["gpu"]   = gpu
            df["model"] = model
            df["steps"] = steps
            frames.append(df)
        else:
            print(f"  ⚠  Not found: {path}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


df_acc_all = pd.concat([load_bruto(g, ACC_FNAMES) for g in GPUS], ignore_index=True)
df_f1_all  = pd.concat([load_bruto(g, F1_FNAMES)  for g in GPUS], ignore_index=True)

# ── 1c. Factorial analysis data (extracted from XLSX) ─────────────
# ┌──────────────────────────────────────────────────────────────────┐
# │  Fill in each GPU × response variable block below.              │
# │                                                                  │
# │  exp order: [135M/Few, 360M/Few, 135M/Many, 360M/Many]          │
# └──────────────────────────────────────────────────────────────────┘
factorial_data = {'A100': {'GPU Power (W)': {'qA': 11.900525709297753,
                            'qB': 38.02806707802925,
                            'qAB': -2.184937802921766,
                            'SSA': 566.4900486306271,
                            'SSB': 5784.535542764369,
                            'SSAB': 19.095812810546377,
                            'SST': 6370.121404205543,
                            'infA': 0.08892923897127478,
                            'infB': 0.9080730453497211,
                            'infAB': 0.0029977156790040697,
                            'exp': [326.729527116294, 354.900454140733, 407.155536878196, 426.586712690948]},
          'Energy Consumed (kWh)': {'qA': 0.14642713674882674,
                                    'qB': 0.5006850758465202,
                                    'qAB': 0.11602532664628223,
                                    'SSA': 0.08576362550583842,
                                    'SSB': 1.002742180701743,
                                    'SSAB': 0.05384750569350596,
                                    'SST': 1.1423533119010874,
                                    'infA': 0.07507626984782131,
                                    'infB': 0.8777863820721055,
                                    'infAB': 0.04713734808007318,
                                    'exp': [0.155434215349196, 0.216237835554285, 0.924753713749672, 1.44965864053989]},
          'Emissions (kg CO2eq)': {'qA': 0.05388630714183655,
                                   'qB': 0.18425594037722337,
                                   'qAB': 0.04269820831515746,
                                   'SSA': 0.011614936389537379,
                                   'SSB': 0.13580100625717959,
                                   'SSAB': 0.0072925479732983255,
                                   'SST': 0.15470849062001527,
                                   'infA': 0.07507626984782119,
                                   'infB': 0.8777863820721062,
                                   'infAB': 0.04713734808007272,
                                   'exp': [0.0572009810109482,
                                           0.0795771786643064,
                                           0.34031644513508,
                                           0.533485476049068]}},
 'A40': {'GPU Power (W)': {'qA': 9.725507972450998,
                           'qB': 276.2700435768945,
                           'qAB': -9.72073018745101,
                           'SSA': 378.3420212888317,
                           'SSB': 305300.5479119168,
                           'SSAB': 377.97038150888545,
                           'SST': 306056.8603147145,
                           'infA': 0.001236182129359189,
                           'infB': 0.9975288500247308,
                           'infAB': 0.0012349678459101462,
                           'exp': [148.172987511309, 187.065463831113, 720.15453504, 720.16409061]},
         'Energy Consumed (kWh)': {'qA': 0.6151238162038017,
                                   'qB': 2.540577961621489,
                                   'qAB': 0.49338263937865334,
                                   'SSA': 1.5135092370445138,
                                   'SSB': 25.8181455163072,
                                   'SSAB': 0.9737057153609852,
                                   'SST': 28.3053604687127,
                                   'infA': 0.05347076355792994,
                                   'infB': 0.9121291899760563,
                                   'infAB': 0.03440004646601374,
                                   'exp': [0.611807364391719, 0.855289718042016, 4.70619800887739, 6.9232109200423]},
         'Emissions (kg CO2eq)': {'qA': 0.2768589590705173,
                                  'qB': 1.1434799813684577,
                                  'qAB': 0.22206489224371273,
                                  'SSA': 0.30660353287044145,
                                  'SSB': 5.230185871161634,
                                  'SSAB': 0.197251265468847,
                                  'SST': 5.7340406695009225,
                                  'infA': 0.05347076355793052,
                                  'infB': 0.9121291899760553,
                                  'infAB': 0.03440004646601422,
                                  'exp': [0.27536626870101, 0.384954402354619, 2.1181964469505, 3.11604414957896]}},
 'RTX6000': {'GPU Power (W)': {'qA': -133.86818081880315,
                               'qB': 142.40399245189514,
                               'qAB': -113.99274927802136,
                               'SSA': 71682.7593429431,
                               'SSB': 81115.58826495762,
                               'SSAB': 51977.38755184736,
                               'SST': 204775.7351597481,
                               'infA': 0.3500549480973539,
                               'infB': 0.39611914078432364,
                               'infAB': 0.2538259111183225,
                               'exp': [127.323772860167, 87.5729097786034, 640.11725632, 144.395396126351]},
             'Energy Consumed (kWh)': {'qA': 0.6432576197387858,
                                       'qB': 1.8829690797922363,
                                       'qAB': 0.47037409127146423,
                                       'SSA': 1.6551214614080334,
                                       'SSB': 14.182290221814483,
                                       'SSAB': 0.885007142957823,
                                       'SST': 16.722418826180338,
                                       'infA': 0.09897619947281808,
                                       'infB': 0.8481004075565269,
                                       'infAB': 0.052923392970655105,
                                       'exp': [0.590234675835736,
                                               0.936001732770379,
                                               3.41542465287728,
                                               5.64268807489778]},
             'Emissions (kg CO2eq)': {'qA': 0.13204094143336723,
                                      'qB': 0.38651545252841474,
                                      'qAB': 0.09655328741005773,
                                      'SSA': 0.06973924085843966,
                                      'SSB': 0.5975767801729809,
                                      'SSAB': 0.03729014923875685,
                                      'SST': 0.7046061702701774,
                                      'infA': 0.09897619947281831,
                                      'infB': 0.8481004075565267,
                                      'infAB': 0.05292339297065501,
                                      'exp': [0.121156967088266,
                                              0.192132275134885,
                                              0.70108129732498,
                                              1.15826975501183]}}}

# Flatten to DataFrame for heatmap / cross-GPU plots
factorial_flat = []
for gpu, resp_dict in factorial_data.items():
    for resp, v in resp_dict.items():
        factorial_flat.append({
            "gpu": gpu, "response": resp,
            "infA":  v["infA"],  "infB":  v["infB"],  "infAB": v["infAB"],
            "SSA":   v["SSA"],   "SSB":   v["SSB"],   "SSAB":  v["SSAB"],
            "SST":   v["SST"],
            "exp0":  v["exp"][0], "exp1": v["exp"][1],
            "exp2":  v["exp"][2], "exp3": v["exp"][3],
        })
df_fact = pd.DataFrame(factorial_flat)

# ── Helper ────────────────────────────────────────────────────────
def em_val(gpu, model, steps, col):
    if df_em.empty:
        return 0.0
    row = df_em[(df_em.gpu == gpu) & (df_em.model == model) & (df_em.steps == steps)]
    return float(row[col].values[0]) if len(row) else 0.0


# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Main Effects Plot
# ══════════════════════════════════════════════════════════════════
print("\n[Fig 1] Main Effects Plot …")

for gpu, resp_dict in factorial_data.items():
    for resp, fdata in resp_dict.items():
        exp = fdata["exp"]
        mA0 = np.mean([exp[0], exp[2]])
        mA1 = np.mean([exp[1], exp[3]])
        mB0 = np.mean([exp[0], exp[1]])
        mB1 = np.mean([exp[2], exp[3]])
        gm  = np.mean(exp)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
        fig.suptitle(f"Main Effects Plot — {resp}  [{gpu}]",
                     fontweight="bold", y=1.01)
        color = GPU_COLORS.get(gpu, C_135_FEW)

        for ax, (xlabel, lo, hi, lbl_lo, lbl_hi) in zip(axes, [
            (FACTOR_A_LABEL, mA0, mA1, LEVEL_A["low"],  LEVEL_A["high"]),
            (FACTOR_B_LABEL, mB0, mB1, LEVEL_B["low"],  LEVEL_B["high"]),
        ]):
            ax.plot([lbl_lo, lbl_hi], [lo, hi],
                    marker="o", color=color, linewidth=2.4, markersize=10, zorder=3)
            ax.axhline(gm, color=CGRAY, linestyle=":", linewidth=1.5,
                       label=f"Grand mean = {gm:.3f}")
            ax.set_xlabel(xlabel, labelpad=6)
            if ax is axes[0]:
                ax.set_ylabel(resp)
            ax.tick_params(axis="x", length=0)
            for xv, yv in [(lbl_lo, lo), (lbl_hi, hi)]:
                ax.annotate(f"{yv:.3f}", xy=(xv, yv),
                            xytext=(0, 11), textcoords="offset points",
                            ha="center", fontsize=SMALL_FONT,
                            color=color, fontweight="bold")
            ax.legend(loc="best", frameon=True, fontsize=SMALL_FONT - 1)

        save(fig, f"fig01_main_effects_{safe_name(gpu + '_' + resp)}.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Interaction Plot (A × B lines)
# ══════════════════════════════════════════════════════════════════
print("[Fig 2] Interaction Plot …")

for gpu, resp_dict in factorial_data.items():
    for resp, fdata in resp_dict.items():
        exp = fdata["exp"]
        fig, ax = plt.subplots(figsize=(7, 4.8))

        # A = 135M (low)
        ax.plot([LEVEL_B["low"], LEVEL_B["high"]], [exp[0], exp[2]],
                marker=MARKERS[0], color=C_135_FEW, linewidth=2.4,
                markersize=10, label="A = 135M")
        # A = 360M (high)
        ax.plot([LEVEL_B["low"], LEVEL_B["high"]], [exp[1], exp[3]],
                marker=MARKERS[1], color=C_360_FEW, linewidth=2.4,
                markersize=10, linestyle="--", label="A = 360M")

        ax.set_xlabel(FACTOR_B_LABEL)
        ax.set_ylabel(resp)
        ax.set_title(f"Interaction Plot — {resp}  [{gpu}]", fontweight="bold")
        ax.legend(title=FACTOR_A_LABEL, frameon=True)
        save(fig, f"fig02_interaction_{safe_name(gpu + '_' + resp)}.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Factor Influence (% contribution) per GPU (BBOX TEXT FIXED)
# ══════════════════════════════════════════════════════════════════
print("[Fig 3] Factor Influence per GPU …")

if not df_fact.empty:
    for gpu in df_fact["gpu"].unique():
        sub = df_fact[df_fact.gpu == gpu]
        if sub.empty:
            continue
            
        responses = sub["response"].tolist()
        infA  = sub["infA"].values  * 100
        infB  = sub["infB"].values  * 100
        infAB = sub["infAB"].values * 100

        # Mantendo o tamanho maior para o layout limpo
        fig, ax = plt.subplots(figsize=(10, 6)) 
        x = np.arange(len(responses))
        w = 0.25

        # Mantendo as cores e o layout corrigido
        b1 = ax.bar(x - w, infA,  w, label=f"Factor A: {FACTOR_A_LABEL}",
                    color=C_135_FEW, edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x,     infB,  w, label=f"Factor B: {FACTOR_B_LABEL}",
                    color=C_360_FEW, hatch="///", edgecolor="white", linewidth=0.5)
        b3 = ax.bar(x + w, infAB, w, label="Interaction AB",
                    color=C_135_MANY, hatch="...", edgecolor="white", linewidth=0.5)

        # ── ATUALIZAÇÃO DE LEGIBILIDADE COM BBOX ──────────────────────────────
        # Ativamos with_bbox=True para colocar um fundo branco atrás de cada número.
        # Definimos color_inside="black" para que o texto preto contraste com a bbox branca.
        for bars in [b1, b2, b3]:
            add_bar_labels(ax, bars, fmt="{:.2f}%", threshold_frac=0.6, 
                           color_inside="black", with_bbox=True)
        # ───────────────────────────────────────────────────────────────────

        ax.set_xticks(x)
        ax.set_xticklabels(responses, fontsize=SMALL_FONT)
        ax.set_ylabel("Contribution (%)")
        ax.set_ylim(0, 120) 
        
        ax.set_title(f"Factor Influence on Response Variables — {gpu}", 
                     fontweight="bold", pad=35) 

        # Legenda no topo, centralizada e fora do eixo
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
                  frameon=True, ncol=3, fontsize=SMALL_FONT - 2)
        
        save(fig, f"fig03_influence_{safe_name(gpu)}.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Cross-GPU Influence Comparison (FIXED)
# ══════════════════════════════════════════════════════════════════
print("[Fig 4] Cross-GPU Factor Influence …")

if not df_fact.empty:
    for resp in df_fact["response"].unique():
        sub = df_fact[df_fact.response == resp]
        if sub.empty:
            continue
            
        gpus_here = sub["gpu"].tolist()
        # Increased height and adjusted width logic for clarity
        fig, ax = plt.subplots(figsize=(max(8, len(gpus_here) * 3), 6))
        x = np.arange(len(gpus_here))
        w = 0.26

        b1 = ax.bar(x - w, sub["infA"].values  * 100, w,
                    label=f"Factor A: {FACTOR_A_LABEL}",
                    color=C_135_FEW,  hatch="",    edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x,     sub["infB"].values  * 100, w,
                    label=f"Factor B: {FACTOR_B_LABEL}",
                    color=C_360_FEW,  hatch="///", edgecolor="white", linewidth=0.5)
        b3 = ax.bar(x + w, sub["infAB"].values * 100, w,
                    label="Interaction AB",
                    color=C_135_MANY, hatch="...", edgecolor="white", linewidth=0.5)

        # ── LEGIBILITY WITH BBOX ──
        # Same logic: white background boxes for all numbers
        for bars in [b1, b2, b3]:
            add_bar_labels(ax, bars, fmt="{:.2f}%", threshold_frac=0.6, 
                           color_inside="black", with_bbox=True)

        ax.set_xticks(x)
        ax.set_xticklabels(gpus_here, fontsize=SMALL_FONT)
        ax.set_ylabel("Contribution (%)")
        ax.set_ylim(0, 125) # Extra headroom for the labels and legend
        
        ax.set_title(f"Factor Influence Across GPUs — {resp}", 
                     fontweight="bold", pad=35)

        # Legend moved to top center to match Figure 3
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
                  frameon=True, ncol=3, fontsize=SMALL_FONT - 2)
        
        save(fig, f"fig04_influence_xgpu_{safe_name(resp)}.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — GPU Power: all GPUs × all configs
# ══════════════════════════════════════════════════════════════════
print("[Fig 5] GPU Power — all GPUs …")

if not df_em.empty:
    w = 0.22
    offsets = np.linspace(-(len(GPUS) - 1) / 2 * w,
                           (len(GPUS) - 1) / 2 * w, len(GPUS))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(CONFIGS_ORDER))

    for gpu, offset in zip(GPUS, offsets):
        vals  = [em_val(gpu, m, s, "gpu_power_w") for m, s in CONFIGS_ORDER]
        bars  = ax.bar(x + offset, vals, w, label=gpu,
                       color=GPU_COLORS[gpu], edgecolor="white", linewidth=0)
        add_bar_labels(ax, bars, fmt="{:.1f}", fontsize=SMALL_FONT - 2)

    ax.set_xticks(x)
    ax.set_xticklabels([cfg_label(m, s) for m, s in CONFIGS_ORDER],
                       fontsize=SMALL_FONT, rotation=12, ha="right")
    ax.set_ylabel("Average GPU Power (W)")
    ax.set_title("Average GPU Power per Configuration and GPU", fontweight="bold")
    ax.legend(frameon=True, ncol=3)
    save(fig, "fig05_gpu_power_all.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 6 — Energy Consumed: all GPUs × all configs
# ══════════════════════════════════════════════════════════════════
print("[Fig 6] Energy Consumed — all GPUs …")

if not df_em.empty:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(CONFIGS_ORDER))

    for gpu, offset in zip(GPUS, offsets):
        vals = [em_val(gpu, m, s, "energy_kwh") for m, s in CONFIGS_ORDER]
        bars = ax.bar(x + offset, vals, w, label=gpu,
                      color=GPU_COLORS[gpu], edgecolor="white", linewidth=0)
        add_bar_labels(ax, bars, fmt="{:.4f}", fontsize=SMALL_FONT - 3)

    ax.set_xticks(x)
    ax.set_xticklabels([cfg_label(m, s) for m, s in CONFIGS_ORDER],
                       fontsize=SMALL_FONT, rotation=12, ha="right")
    ax.set_ylabel("Energy Consumed (kWh)")
    ax.set_title("Energy Consumed per Configuration and GPU", fontweight="bold")
    ax.legend(frameon=True, ncol=3)
    save(fig, "fig06_energy_consumed_all.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 7 — CO₂ Emissions: all GPUs × all configs
# ══════════════════════════════════════════════════════════════════
print("[Fig 7] CO₂ Emissions — all GPUs …")

if not df_em.empty:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(CONFIGS_ORDER))

    for gpu, offset in zip(GPUS, offsets):
        vals = [em_val(gpu, m, s, "emissions_kgco2") for m, s in CONFIGS_ORDER]
        bars = ax.bar(x + offset, vals, w, label=gpu,
                      color=GPU_COLORS[gpu], edgecolor="white", linewidth=0)
        add_bar_labels(ax, bars, fmt="{:.5f}", fontsize=SMALL_FONT - 3)

    ax.set_xticks(x)
    ax.set_xticklabels([cfg_label(m, s) for m, s in CONFIGS_ORDER],
                       fontsize=SMALL_FONT, rotation=12, ha="right")
    ax.set_ylabel("CO₂ Emissions (kg CO₂eq)")
    ax.set_title("CO₂ Emissions per Configuration and GPU", fontweight="bold")
    ax.legend(frameon=True, ncol=3)
    save(fig, "fig07_co2_emissions_all.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 8 — Trade-off: Accuracy × Energy (CLEAN VERSION)
# ══════════════════════════════════════════════════════════════════
print("[Fig 8] Accuracy vs Energy Trade-off — Clean Version …")

if not df_acc_all.empty and not df_em.empty:
    # Aumentamos um pouco a altura para acomodar a legenda inferior sem apertar
    fig, axes = plt.subplots(1, len(GPUS), figsize=(6 * len(GPUS), 6.5), sharey=True)
    axes = [axes] if len(GPUS) == 1 else list(axes)

    for ax, gpu in zip(axes, GPUS):
        df_acc_gpu = df_acc_all[df_acc_all.gpu == gpu]
        
        for (model, steps), color, marker in zip(CONFIGS_ORDER, CONFIG_COLORS, CONFIG_MARKERS):
            sub = df_acc_gpu[(df_acc_gpu.model == model) & (df_acc_gpu.steps == steps)]
            ene = em_val(gpu, model, steps, "energy_kwh")
            
            if sub.empty or ene == 0:
                continue
                
            acc = sub["top1"].mean() * 100
            
            # Plot apenas do símbolo (sem texto)
            # Adicionamos uma borda branca (edgecolor) para destacar pontos sobrepostos
            ax.scatter(ene, acc, color=color, marker=marker, s=250, 
                       zorder=5, edgecolor='white', linewidth=1.5)

        ax.set_xlabel("Energy Consumed (kWh)", labelpad=12)
        if ax is axes[0]:
            ax.set_ylabel("Top-1 Accuracy (%)", labelpad=12)
        ax.set_title(f"GPU: {gpu}", fontweight="bold", pad=15)
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle("Trade-off: Top-1 Accuracy vs. Energy Consumed", 
                 fontweight="bold", y=0.98, fontsize=TITLE_FONT)
    
    # Criando a legenda global na parte inferior
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=m, color='w', label=cfg_label(mod, stp),
               markerfacecolor=c, markersize=12, markeredgecolor='none')
        for (mod, stp), c, m in zip(CONFIGS_ORDER, CONFIG_COLORS, CONFIG_MARKERS)
    ]
    
    # ncol=4 coloca todas as opções em uma única linha horizontal
    fig.legend(handles=legend_elements, loc="lower center", 
               bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=True, 
               fontsize=SMALL_FONT - 1, borderpad=0.8)

    # Ajusta o layout para que o título e a legenda não cortem
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    save(fig, "fig08_accuracy_vs_energy.png", tight=False)


# ══════════════════════════════════════════════════════════════════
# FIGURE 9 — Top-k Accuracy Profile (all GPUs, grouped bars)
# ══════════════════════════════════════════════════════════════════
print("[Fig 9] Top-k Accuracy Profile …")

TOPK_COLS   = ["top1", "top3", "top5", "top10"]
TOPK_LABELS = ["Top-1", "Top-3", "Top-5", "Top-10"]

if not df_acc_all.empty:
    fig, axes = plt.subplots(1, len(GPUS), figsize=(6 * len(GPUS), 5.5),
                             sharey=True)
    axes = [axes] if len(GPUS) == 1 else list(axes)

    for ax, gpu in zip(axes, GPUS):
        df_acc_gpu = df_acc_all[df_acc_all.gpu == gpu]
        if df_acc_gpu.empty:
            ax.set_title(f"{gpu} (no data)")
            continue

        ww = 0.16
        x  = np.arange(len(CONFIGS_ORDER))

        for i, (col, lbl) in enumerate(zip(TOPK_COLS, TOPK_LABELS)):
            vals = []
            for model, steps in CONFIGS_ORDER:
                sub = df_acc_gpu[(df_acc_gpu.model == model) & (df_acc_gpu.steps == steps)]
                vals.append(sub[col].mean() * 100 if len(sub) else 0)
            offset = (i - (len(TOPK_COLS) - 1) / 2) * ww
            ax.bar(x + offset, vals, ww, label=lbl,
                   color=CONFIG_COLORS[i], hatch=HATCHES[i],
                   edgecolor="white", linewidth=0)

        ax.set_xticks(x)
        ax.set_xticklabels([cfg_label(m, s) for m, s in CONFIGS_ORDER],
                           fontsize=SMALL_FONT - 2, rotation=15, ha="right")
        if ax is axes[0]:
            ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 112)
        ax.set_title(gpu, fontweight="bold")
        ax.legend(loc="lower right", frameon=True, ncol=2,
                  fontsize=SMALL_FONT - 2)

    fig.suptitle("Top-k Accuracy Profile per GPU and Configuration",
                 fontweight="bold", y=1.02)
    save(fig, "fig09_topk_accuracy_profile.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 10 — F1-Score Profile (GLOBAL LEGEND REFINED)
# ══════════════════════════════════════════════════════════════════
print("[Fig 10] F1-Score Profile …")

if not df_f1_all.empty:
    f1_cols = [c for c in df_f1_all.columns
               if c not in ("gpu", "model", "steps", "label", "round", "k")]

    if f1_cols:
        # Aumentamos a altura levemente para acomodar a legenda inferior
        fig, axes = plt.subplots(1, len(GPUS), figsize=(6 * len(GPUS), 6), sharey=True)
        axes = [axes] if len(GPUS) == 1 else list(axes)
        pal = [C_135_FEW, C_360_FEW, C_135_MANY, C_360_MANY, CGRAY]

        for ax, gpu in zip(axes, GPUS):
            df_f1_gpu = df_f1_all[df_f1_all.gpu == gpu]
            if df_f1_gpu.empty:
                ax.set_title(f"{gpu} (no data)")
                continue

            ww = 0.65 / len(f1_cols)
            x = np.arange(len(CONFIGS_ORDER))

            for i, col in enumerate(f1_cols):
                vals = []
                for model, steps in CONFIGS_ORDER:
                    sub = df_f1_gpu[(df_f1_gpu.model == model) & (df_f1_gpu.steps == steps)]
                    vals.append(sub[col].mean() * 100 if len(sub) else 0)
                
                offset = (i - (len(f1_cols) - 1) / 2) * ww
                ax.bar(x + offset, vals, ww, label=col.upper(),
                       color=pal[i % len(pal)], hatch=HATCHES[i % len(HATCHES)],
                       edgecolor="white", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels([cfg_label(m, s) for m, s in CONFIGS_ORDER],
                               fontsize=SMALL_FONT - 2, rotation=15, ha="right")
            if ax is axes[0]:
                ax.set_ylabel("F1-Score (%)", labelpad=10)
            ax.set_title(f"GPU: {gpu}", fontweight="bold", pad=15)

        fig.suptitle("F1-Score, Precision and Recall per GPU and Configuration",
                     fontweight="bold", y=0.98, fontsize=TITLE_FONT)
        
        # ── LEGENDA GLOBAL INFERIOR ──
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", 
                   bbox_to_anchor=(0.5, 0.02), ncol=len(f1_cols), 
                   frameon=True, fontsize=SMALL_FONT - 1)

        # Ajuste para evitar que a legenda e os rótulos do eixo X se sobreponham
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        save(fig, "fig10_f1_score_profile.png", tight=False)


# ══════════════════════════════════════════════════════════════════
# FIGURE 11 — Factorial Heatmap (RESTORED & FIXED)
# ══════════════════════════════════════════════════════════════════
print("[Fig 11] Factorial Heatmap — Restoring data …")

if not df_fact.empty:
    metric_cols = ["infA", "infB", "infAB"]
    metric_xlbls = [
        "Factor A\n(Model Size)",
        "Factor B\n(Optim. Steps)",
        "Interaction\nAB"
    ]
    
    all_responses = df_fact["response"].unique().tolist()
    gpus_w_data = df_fact["gpu"].unique().tolist()
    n_gpus = len(gpus_w_data)

    fig, axes = plt.subplots(
        1, n_gpus,
        figsize=(6 * n_gpus, max(5, len(all_responses) * 1.5)),
        sharey=True
    )
    axes = [axes] if n_gpus == 1 else list(axes)
    im_ref = None

    for ax, gpu in zip(axes, gpus_w_data):
        # GARANTIA DE DADOS: Filtra e reconstrói a matriz explicitamente
        sub = df_fact[df_fact.gpu == gpu].set_index("response")
        
        matrix = []
        for r in all_responses:
            row = []
            for col in metric_cols:
                # Recupera o valor e converte para porcentagem
                val = sub.loc[r, col] * 100 if r in sub.index else 0
                row.append(val)
            matrix.append(row)
        
        matrix = np.array(matrix)

        # Plot com vmin/vmax fixos para garantir que as cores apareçam
        im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=100)
        im_ref = im

        # Formatação do Eixo X
        ax.set_xticks(range(len(metric_xlbls)))
        ax.set_xticklabels(metric_xlbls, fontsize=SMALL_FONT - 2, ha="center")
        
        # Formatação do Eixo Y
        ax.set_yticks(range(len(all_responses)))
        if ax is axes[0]:
            ax.set_yticklabels(all_responses, fontsize=SMALL_FONT - 1)
        else:
            ax.set_yticklabels([])

        ax.set_title(f"GPU: {gpu}", fontweight="bold", pad=20)
        
        # Remove molduras e grades que podem cobrir o heatmap
        ax.spines[:].set_visible(False)
        ax.grid(False)

        # Adição dos Textos (Valores) sobre as células
        for i in range(len(all_responses)):
            for j in range(len(metric_xlbls)):
                val = matrix[i, j]
                # Cor do texto dinâmica para contraste
                text_col = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=BASE_FONT - 1, color=text_col, fontweight="bold")

    # Barra de Cores Global
    if im_ref is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5]) # [left, bottom, width, height]
        cbar = fig.colorbar(im_ref, cax=cbar_ax)
        cbar.set_label("Influence (%)", fontsize=SMALL_FONT)

    fig.suptitle("Factorial Analysis — Factor Influence Heatmap", 
                 fontweight="bold", y=0.98, fontsize=TITLE_FONT)

    save(fig, "fig11_factorial_heatmap.png", tight=False)


# ══════════════════════════════════════════════════════════════════
# FIGURE 12 — Training Duration per GPU and Config
# ══════════════════════════════════════════════════════════════════
print("[Fig 12] Training Duration …")

if not df_em.empty:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(CONFIGS_ORDER))

    for gpu, offset in zip(GPUS, offsets):
        vals = [em_val(gpu, m, s, "duration_s") / 60 for m, s in CONFIGS_ORDER]
        bars = ax.bar(x + offset, vals, w, label=gpu,
                      color=GPU_COLORS[gpu], edgecolor="white", linewidth=0)
        add_bar_labels(ax, bars, fmt="{:.1f}", fontsize=SMALL_FONT - 2)

    ax.set_xticks(x)
    ax.set_xticklabels([cfg_label(m, s) for m, s in CONFIGS_ORDER],
                       fontsize=SMALL_FONT, rotation=12, ha="right")
    ax.set_ylabel("Training Duration (min)")
    ax.set_title("Training Duration per Configuration and GPU", fontweight="bold")
    ax.legend(frameon=True, ncol=3)
    save(fig, "fig12_training_duration.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 13 — Energy Efficiency: Top-1 Accuracy per kWh
# ══════════════════════════════════════════════════════════════════
print("[Fig 13] Energy Efficiency (Accuracy / kWh) …")

if not df_acc_all.empty and not df_em.empty:
    fig, axes = plt.subplots(1, len(GPUS), figsize=(6 * len(GPUS), 5.5),
                             sharey=True)
    axes = [axes] if len(GPUS) == 1 else list(axes)

    for ax, gpu in zip(axes, GPUS):
        df_acc_gpu = df_acc_all[df_acc_all.gpu == gpu]
        vals, labels, colors = [], [], []
        for (model, steps), color in zip(CONFIGS_ORDER, CONFIG_COLORS):
            sub = df_acc_gpu[(df_acc_gpu.model == model) & (df_acc_gpu.steps == steps)]
            ene = em_val(gpu, model, steps, "energy_kwh")
            if sub.empty:
                continue
            acc = sub["top1"].mean() * 100
            vals.append(acc / ene if ene > 0 else 0)
            labels.append(cfg_label(model, steps))
            colors.append(color)

        x_loc = np.arange(len(vals))
        bars  = ax.bar(x_loc, vals, color=colors, edgecolor="white", linewidth=0)
        add_bar_labels(ax, bars, fmt="{:.1f}", fontsize=SMALL_FONT - 2)
        ax.set_xticks(x_loc)
        ax.set_xticklabels(labels, fontsize=SMALL_FONT - 2,
                           rotation=15, ha="right")
        if ax is axes[0]:
            ax.set_ylabel("Top-1 Accuracy (%) per kWh")
        ax.set_title(gpu, fontweight="bold")

    fig.suptitle("Energy Efficiency: Top-1 Accuracy per kWh",
                 fontweight="bold", y=1.02)
    save(fig, "fig13_energy_efficiency.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 14 — Radar Chart: multi-metric overview per GPU
# ══════════════════════════════════════════════════════════════════
print("[Fig 14] Radar Chart — multi-metric overview …")

RADAR_METRICS = [
    "Top-1 Acc", "Top-5 Acc",
    "Energy\nEfficiency", "Low Duration", "Low Emissions",
]


def norm01(arr):
    mn, mx = np.min(arr), np.max(arr)
    return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr, dtype=float)


if not df_acc_all.empty and not df_em.empty:
    for gpu in GPUS:
        df_acc_gpu = df_acc_all[df_acc_all.gpu == gpu]
        if df_acc_gpu.empty:
            continue

        raw = {m: [] for m in RADAR_METRICS}
        valid_cfgs = []

        for model, steps in CONFIGS_ORDER:
            sub = df_acc_gpu[(df_acc_gpu.model == model) & (df_acc_gpu.steps == steps)]
            ene = em_val(gpu, model, steps, "energy_kwh")
            dur = em_val(gpu, model, steps, "duration_s")
            ems = em_val(gpu, model, steps, "emissions_kgco2")
            if sub.empty or ene == 0:
                continue
            acc1 = sub["top1"].mean() * 100
            acc5 = sub["top5"].mean() * 100
            raw["Top-1 Acc"].append(acc1)
            raw["Top-5 Acc"].append(acc5)
            raw["Energy\nEfficiency"].append(acc1 / ene)
            raw["Low Duration"].append(dur)
            raw["Low Emissions"].append(ems)
            valid_cfgs.append((model, steps))

        if len(valid_cfgs) < 2:
            continue

        data_norm = {}
        for m in RADAR_METRICS:
            arr = np.array(raw[m], dtype=float)
            n   = norm01(arr)
            if m.startswith("Low"):
                n = 1 - n
            data_norm[m] = n

        N      = len(RADAR_METRICS)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})
        ax.set_facecolor("white")
        ax.spines["polar"].set_visible(False)

        for i, (model, steps) in enumerate(valid_cfgs):
            vals = [data_norm[m][i] for m in RADAR_METRICS] + [data_norm[RADAR_METRICS[0]][i]]
            ax.plot(angles, vals, color=CONFIG_COLORS[i], linewidth=2.2,
                    marker=CONFIG_MARKERS[i], markersize=7,
                    label=cfg_label(model, steps))
            ax.fill(angles, vals, color=CONFIG_COLORS[i], alpha=0.10)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_METRICS, fontsize=SMALL_FONT)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                           fontsize=SMALL_FONT - 3, color=CGRAY)
        ax.set_title(f"Multi-Metric Overview — {gpu}",
                     fontweight="bold", pad=18)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20),
                  ncol=2, frameon=True, fontsize=SMALL_FONT - 2)
        save(fig, f"fig14_radar_{safe_name(gpu)}.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 15 — Experimental Response Values (dot + line per GPU)
# ══════════════════════════════════════════════════════════════════
print("[Fig 15] Experimental Response Values …")

if not df_fact.empty:
    for resp in df_fact["response"].unique():
        sub = df_fact[df_fact.response == resp]
        if sub.empty:
            continue

        exp_labels_local = [cfg_label(m, s) for m, s in CONFIGS_ORDER]
        fig, ax = plt.subplots(figsize=(9, 5))

        for i, row in sub.iterrows():
            gpu  = row["gpu"]
            vals = [row["exp0"], row["exp1"], row["exp2"], row["exp3"]]
            ax.plot(exp_labels_local, vals,
                    marker=MARKERS[list(sub["gpu"]).index(gpu)],
                    color=GPU_COLORS.get(gpu, CGRAY),
                    linewidth=2.2, markersize=9, label=gpu)
            for xv, yv in zip(exp_labels_local, vals):
                ax.annotate(f"{yv:.3f}", xy=(xv, yv),
                            xytext=(0, 10), textcoords="offset points",
                            ha="center", fontsize=SMALL_FONT - 2,
                            color=GPU_COLORS.get(gpu, CGRAY))

        ax.set_xlabel("Experiment Configuration")
        ax.set_ylabel(resp)
        ax.set_title(f"Experimental Response Values — {resp}", fontweight="bold")
        ax.legend(frameon=True)
        save(fig, f"fig15_response_values_{safe_name(resp)}.png")


# ══════════════════════════════════════════════════════════════════
print(f"\n✅  All figures saved to '{OUT_DIR}/'  (300 DPI png, LaTeX-ready)")
print("   15 figure slots — some generate multiple files (one per GPU or response).")
print("\n   ⚠  Remember to fill in factorial_data for A40 and RTX6000,")
print("      and for Energy Consumed + Emissions response variables.")
