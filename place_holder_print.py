#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pprint import pprint

import pandas as pd


BASE_DIR = "Resultados"  # ajuste se você rodar de outro lugar
GPUS = ["A100", "A40", "RTX6000"]

# arquivos esperados dentro de cada pasta da GPU
METRIC_FILES = {
    "GPU Power (W)": "gpu-power.xlsx",
    "Energy Consumed (kWh)": "energy.xlsx",
    "Emissions (kg CO2eq)": "emissions.xlsx",
}


def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)) and pd.notna(x):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    # aceita decimal com vírgula, se aparecer
    s = s.replace(".", "").replace(",", ".") if re.search(r"\d+,\d+", s) else s
    try:
        return float(s)
    except Exception:
        return None


def read_all_sheets(xlsx_path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    frames = []
    for sh in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sh, header=None)
        df["__sheet__"] = sh
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def find_value_in_table(df: pd.DataFrame, keys):
    """
    Procura qualquer ocorrência de uma chave em qualquer célula,
    e retorna o primeiro número na mesma linha, em outra coluna.
    """
    keys_n = {_norm(k) for k in keys}

    for i in range(df.shape[0]):
        row = df.iloc[i, :].tolist()
        row_norm = [_norm(v) for v in row]

        hit = None
        for j, cell in enumerate(row_norm):
            if cell in keys_n:
                hit = j
                break

        if hit is None:
            continue

        # tenta pegar um número em qualquer outra célula da linha
        for j, v in enumerate(row):
            if j == hit:
                continue
            fv = _to_float(v)
            if fv is not None:
                return fv

    return None


def read_experiments_values(df: pd.DataFrame):
    """
    Captura os 4 valores numéricos da tabela EXPERIMENTO 1..4.
    Procura linhas onde a primeira célula é 1, 2, 3, 4 e pega o último número da linha.
    """
    exp = [None, None, None, None]

    for i in range(df.shape[0]):
        first = df.iloc[i, 0]
        fnum = _to_float(first)
        if fnum is None:
            continue
        if int(fnum) in {1, 2, 3, 4}:
            row = df.iloc[i, :].tolist()
            nums = [_to_float(v) for v in row]
            nums = [n for n in nums if n is not None]
            # a linha tem vários números (fator A, fator B, resposta), queremos o último
            if len(nums) >= 1:
                exp[int(fnum) - 1] = nums[-1]

    # se não achou tudo, tenta estratégia alternativa: coletar respostas da coluna mais à direita
    if any(v is None for v in exp):
        # tenta achar uma coluna que tenha muitos floats e pegar os 4 primeiros depois do cabeçalho
        best_col = None
        best_count = 0
        for c in range(df.shape[1]):
            col = df.iloc[:, c].tolist()
            floats = [_to_float(v) for v in col]
            cnt = sum(1 for v in floats if v is not None)
            if cnt > best_count:
                best_count = cnt
                best_col = c
        if best_col is not None:
            floats = [_to_float(v) for v in df.iloc[:, best_col].tolist()]
            floats = [v for v in floats if v is not None]
            # pode ter q0, qA etc depois, então tenta pegar a janela que parece ser 4 valores de experimento
            # aqui: pega os 4 primeiros, mas só se fizer sentido (não preencher com q0 etc)
            if len(floats) >= 4 and all(v is None for v in exp):
                exp = floats[:4]

    return exp


def parse_metric_xlsx(xlsx_path: str):
    df = read_all_sheets(xlsx_path)

    qA = find_value_in_table(df, ["qA"])
    qB = find_value_in_table(df, ["qB"])
    qAB = find_value_in_table(df, ["qAB", "Qab", "QAB", "q ab"])

    SSA = find_value_in_table(df, ["SSA"])
    SSB = find_value_in_table(df, ["SSB"])
    SSAB = find_value_in_table(df, ["SSAB", "SSab", "SS A B", "SS A*B"])
    SST = find_value_in_table(df, ["SST"])

    infA = find_value_in_table(df, ["INFLU A", "influ a", "influence a", "influência a"])
    infB = find_value_in_table(df, ["INFLU B", "influ b", "influence b", "influência b"])
    infAB = find_value_in_table(df, ["INFLU AB", "influ ab", "influence ab", "influência ab"])

    exp = read_experiments_values(df)

    out = {
        "qA": qA,
        "qB": qB,
        "qAB": qAB,
        "SSA": SSA,
        "SSB": SSB,
        "SSAB": SSAB,
        "SST": SST,
        "infA": infA,
        "infB": infB,
        "infAB": infAB,
        "exp": exp,
    }

    # checagem simples: se algum campo crítico vier None, avisa no console
    critical = ["qA", "qB", "qAB", "SSA", "SSB", "SSAB", "SST", "infA", "infB", "infAB"]
    missing = [k for k in critical if out[k] is None]
    if missing:
        print(f"[warn] {os.path.basename(xlsx_path)} campos não encontrados: {missing}")

    return out


def build_factorial_data(base_dir: str, gpus):
    factorial_data = {}
    for gpu in gpus:
        gpu_dir = os.path.join(base_dir, gpu)
        gpu_block = {}

        for metric_name, fname in METRIC_FILES.items():
            fpath = os.path.join(gpu_dir, fname)
            if not os.path.exists(fpath):
                print(f"[warn] arquivo não encontrado: {fpath}")
                continue
            gpu_block[metric_name] = parse_metric_xlsx(fpath)

        factorial_data[gpu] = gpu_block

    return factorial_data


if __name__ == "__main__":
    data = build_factorial_data(BASE_DIR, GPUS)
    pprint(data, width=120, sort_dicts=False)