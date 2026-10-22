# -*- coding: utf-8 -*-
"""
TESIS UCV - ANÁLISIS MACHINE LEARNING COMPLETO
Autor: Alan Rossi Rivas Carhuancho
Título: Modelo de machine learning para apoyar la emisión de medidas de protección
        en expedientes de violencia familiar en Huamanga, 2026.

ALINEACIÓN CON EL PROYECTO
--------------------------
Objetivo general:
Desarrollar un modelo de machine learning que contribuya a optimizar la emisión de
medidas de protección en expedientes de violencia familiar en Huamanga durante 2026.

Objetivos específicos cubiertos por este script:
1) Celeridad / sobrecarga: predecir expedientes con demora > 24 horas y resumir el
   tiempo de emisión.
2) Priorización: identificar expedientes de riesgo severo y generar un ranking
   automatizado de prioridad.
3) Consistencia: cuantificar la uniformidad de decisiones por riesgo, juzgado y juez.

Modelo prespecificado en el proyecto: RANDOM FOREST.
Se comparan algoritmos de referencia, pero los resultados principales se reportan con
Random Forest para mantener consistencia con el proyecto aprobado.

REGLAS DE CALIDAD
-----------------
- No se usan variables posteriores a la resolución para predecir riesgo.
- Para predecir demora se usan variables disponibles durante la tramitación.
- No se usa MEDIDA OTORGADA para predecir riesgo o demora.
- Se evita fuga de información (data leakage).
- Se usa partición entrenamiento/prueba 80/20 estratificada y validación cruzada de 5 pliegues.
- Las métricas se guardan tal como resultan, sin alterar datos para inflarlas.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS_CV = 5
PRESPECIFIED_MODEL = "Random Forest"

EXPECTED_COLUMNS = [
    "NRO DE EXPEDIENTE",
    "ORGANO JURISDICCIONAL",
    "TIPO DE VIOLENCIA",
    "NIVEL DE RIESGO",
    "REINCIDENCIA",
    "HORA INGRESO",
    "HORA EMISIÓN",
    "TIEMPO DE EMISIÓN (HRS)",
    "CRITERIO DE CONSISTENCIA",
    "MEDIDA OTORGADA",
    "x_juez",
    "x_especialista",
]


def one_hot():
    """OneHotEncoder compatible con versiones nuevas y antiguas de scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def find_input_xlsx(base_dir: Path) -> Path:
    candidates = []
    for p in base_dir.glob("*.xlsx"):
        name = p.name.upper().replace("%20", " ")
        if "BASE" in name and ("DEPURADA" in name or "EXPEDIENTES" in name):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            "No se encontró una base Excel en la carpeta del script. Use --input RUTA.xlsx"
        )
    candidates.sort(key=lambda p: ("BASE_DEPURADA_ALAN_RIVAS" not in p.name.upper(), len(p.name)))
    return candidates[0]


def load_validate(path: Path) -> pd.DataFrame:
    # Admite el Excel original o una exportación CSV auditable del mismo.
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype={"NRO DE EXPEDIENTE": str})
    else:
        df = pd.read_excel(path, sheet_name="Base_Depurada", engine="openpyxl", dtype={"NRO DE EXPEDIENTE": str})
    df = df.dropna(how="all").copy()
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    if df.empty:
        raise ValueError("La base está vacía.")
    if df["NRO DE EXPEDIENTE"].isna().any():
        raise ValueError("Existen registros sin número de expediente.")
    dups = df["NRO DE EXPEDIENTE"].duplicated(keep=False)
    if dups.any():
        examples = df.loc[dups, "NRO DE EXPEDIENTE"].astype(str).head(10).tolist()
        raise ValueError("Se detectaron expedientes duplicados: " + ", ".join(examples))
    return df.reset_index(drop=True)


def derive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Tanto los valores de fecha de Excel (números seriales) como fechas
    # exportadas como texto deben conservar hora, día y mes reales.
    fecha_origen = out["HORA INGRESO"]
    # Los XLSX pueden devolver fechas datetime; los CSV exportados entregan seriales.
    if pd.api.types.is_datetime64_any_dtype(fecha_origen):
        ingreso = pd.to_datetime(fecha_origen, errors="coerce")
    else:
        numerica = pd.to_numeric(fecha_origen, errors="coerce")
        if numerica.between(30000, 80000).mean() > 0.90:
            ingreso = pd.Timestamp("1899-12-30") + pd.to_timedelta(numerica, unit="D")
        else:
            ingreso = pd.to_datetime(fecha_origen, errors="coerce")
    if ingreso.isna().mean() > 0.05:
        raise ValueError("No se pudieron interpretar las fechas de ingreso de los expedientes.")

    out["INGRESO_HORA"] = ingreso.dt.hour + ingreso.dt.minute / 60 + ingreso.dt.second / 3600
    out["INGRESO_DIA_SEMANA"] = ingreso.dt.dayofweek
    out["INGRESO_MES"] = ingreso.dt.month
    out["INGRESO_DIA"] = ingreso.dt.day
    out["INGRESO_SEMANA_ANIO"] = ingreso.dt.isocalendar().week.astype(float)
    out["INGRESO_FIN_SEMANA"] = ingreso.dt.dayofweek.isin([5, 6]).astype(int)

    out["TIEMPO_HRS"] = pd.to_numeric(out["TIEMPO DE EMISIÓN (HRS)"], errors="coerce")
    out["RIESGO_SEVERO"] = (
        out["NIVEL DE RIESGO"].astype(str).str.upper().str.strip().eq("SEVERO")
    ).astype(int)
    out["DEMORA_MAYOR_24H"] = (out["TIEMPO_HRS"] > 24).astype(int)
    out["MEDIDA_BINARIA"] = np.where(
        out["MEDIDA OTORGADA"].astype(str).str.upper().str.contains("NO OTORGA", na=False),
        0,
        1,
    ).astype(int)
    out["RIESGO_ORDINAL"] = (
        out["NIVEL DE RIESGO"]
        .astype(str)
        .str.upper()
        .str.strip()
        .map({"LEVE": 1, "MODERADO": 2, "SEVERO": 3})
    )
    return out


def optional_stratified_sample(df: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df.copy()

    # Muestreo reproducible proporcional por estrato observado de riesgo severo y demora >24h.
    # La selección no sustituye el control de elegibilidad y validez del marco de datos.
    strata = df["RIESGO_SEVERO"].astype(str) + "_" + df["DEMORA_MAYOR_24H"].astype(str)
    parts = []
    remaining = sample_size
    counts = strata.value_counts()
    allocations = {}
    for s, count in counts.items():
        n_s = int(round(sample_size * count / len(df)))
        n_s = max(1, min(n_s, count))
        allocations[s] = n_s
    # Ajuste exacto al tamaño solicitado.
    while sum(allocations.values()) > sample_size:
        key = max(allocations, key=lambda k: allocations[k])
        if allocations[key] > 1:
            allocations[key] -= 1
        else:
            break
    while sum(allocations.values()) < sample_size:
        room = [k for k in allocations if allocations[k] < counts[k]]
        if not room:
            break
        key = max(room, key=lambda k: counts[k] - allocations[k])
        allocations[key] += 1

    for s, n_s in allocations.items():
        idx = df.index[strata == s]
        parts.append(df.loc[idx].sample(n=n_s, random_state=RANDOM_STATE))

    sampled = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return sampled


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", one_hot()),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )


def candidate_models() -> dict[str, object]:
    return {
        "Regresión logística": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Árbol de decisión": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"variable_{i+1}" for i in range(model.n_features_in_)]

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
    else:
        return pd.DataFrame(columns=["variable", "importancia"])

    n = min(len(names), len(values))
    return (
        pd.DataFrame({"variable": names[:n], "importancia": values[:n]})
        .sort_values("importancia", ascending=False, ignore_index=True)
    )


def evaluate_binary_task(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    categorical: list[str],
    task_slug: str,
    positive_label: str,
    negative_label: str,
    outdir: Path,
):
    numeric = [c for c in features if c not in categorical]
    X = df[features].copy()
    y = df[target].astype(int).copy()

    if y.nunique() != 2:
        raise ValueError(f"La variable {target} no tiene dos clases válidas.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    comparison = []
    fitted = {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("pre", make_preprocessor(categorical, numeric)), ("model", model)])
        cvres = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        comparison.append(
            {
                "modelo": name,
                "cv_accuracy": np.mean(cvres["test_accuracy"]),
                "cv_balanced_accuracy": np.mean(cvres["test_balanced_accuracy"]),
                "cv_precision": np.mean(cvres["test_precision"]),
                "cv_recall": np.mean(cvres["test_recall"]),
                "cv_f1": np.mean(cvres["test_f1"]),
                "cv_roc_auc": np.mean(cvres["test_roc_auc"]),
                "test_accuracy": accuracy_score(y_test, pred),
                "test_balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "test_precision": precision_score(y_test, pred, zero_division=0),
                "test_recall": recall_score(y_test, pred, zero_division=0),
                "test_specificity": specificity,
                "test_f1": f1_score(y_test, pred, zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, proba),
                "test_mcc": matthews_corrcoef(y_test, pred),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
        fitted[name] = (pipe, pred, proba)

    metrics = pd.DataFrame(comparison).sort_values(
        ["cv_roc_auc", "cv_f1"], ascending=False, ignore_index=True
    )
    metrics.to_csv(outdir / f"{task_slug}_comparacion_modelos.csv", index=False, encoding="utf-8-sig")

    # Se conserva Random Forest como modelo principal por estar prespecificado en el proyecto.
    if PRESPECIFIED_MODEL not in fitted:
        raise RuntimeError("No se encontró el modelo prespecificado Random Forest.")
    final_name = PRESPECIFIED_MODEL
    pipe, pred, proba = fitted[final_name]

    report = classification_report(
        y_test,
        pred,
        labels=[0, 1],
        target_names=[negative_label, positive_label],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        outdir / f"{task_slug}_reporte_random_forest.csv", encoding="utf-8-sig"
    )

    test_pred = df.loc[X_test.index, ["NRO DE EXPEDIENTE"] + [c for c in features if c in df.columns]].copy()
    test_pred["real"] = y_test.values
    test_pred["predicho"] = pred
    test_pred["probabilidad_clase_positiva"] = proba
    test_pred.to_csv(
        outdir / f"{task_slug}_predicciones_prueba.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        display_labels=[negative_label, positive_label],
        values_format="d",
    )
    plt.title(f"Matriz de confusión - {task_slug.replace('_', ' ').title()}")
    plt.tight_layout()
    if not (outdir / f"{task_slug}_matriz_confusion.png").exists():
        plt.savefig(outdir / f"{task_slug}_matriz_confusion.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, proba, name="Random Forest")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"Curva ROC - {task_slug.replace('_', ' ').title()}")
    plt.tight_layout()
    if not (outdir / f"{task_slug}_curva_roc.png").exists():
        plt.savefig(outdir / f"{task_slug}_curva_roc.png", dpi=220, bbox_inches="tight")
    plt.close()

    imp = feature_importance(pipe)
    imp.to_csv(outdir / f"{task_slug}_importancia_variables.csv", index=False, encoding="utf-8-sig")
    if not imp.empty:
        top = imp.head(15).sort_values("importancia", ascending=True)
        plt.figure(figsize=(9, 6))
        plt.barh(top["variable"], top["importancia"])
        plt.xlabel("Importancia")
        plt.title(f"Importancia de variables - {task_slug.replace('_', ' ').title()}")
        plt.tight_layout()
        if not (outdir / f"{task_slug}_importancia_variables.png").exists():
            plt.savefig(outdir / f"{task_slug}_importancia_variables.png", dpi=220, bbox_inches="tight")
        plt.close()

    # Métricas exactas del Random Forest.
    rf_row = metrics.loc[metrics["modelo"] == PRESPECIFIED_MODEL].iloc[0].to_dict()

    # Reentrenar modelo final con todos los registros disponibles del análisis.
    final_model = clone(pipe)
    final_model.fit(X, y)
    joblib.dump(final_model, outdir / f"{task_slug}_random_forest.joblib")

    return {
        "task": task_slug,
        "model": final_model,
        "features": features,
        "metrics": rf_row,
        "best_benchmark": metrics.iloc[0]["modelo"],
        "best_benchmark_auc": float(metrics.iloc[0]["cv_roc_auc"]),
        "test_index": X_test.index,
        "y_test": y_test,
        "pred_test": pred,
        "proba_test": proba,
    }


def data_quality(df: pd.DataFrame, outdir: Path, source: Path, original_n: int, sample_size: int | None):
    rows = []
    for c in df.columns:
        rows.append(
            {
                "columna": c,
                "n": len(df),
                "faltantes": int(df[c].isna().sum()),
                "porcentaje_faltantes": float(df[c].isna().mean() * 100),
                "valores_unicos": int(df[c].nunique(dropna=True)),
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "00_calidad_datos.csv", index=False, encoding="utf-8-sig")

    summary = {
        "archivo": source.name,
        "registros_base_original": int(original_n),
        "registros_analizados": int(len(df)),
        "sample_size_solicitado": sample_size,
        "expedientes_unicos": int(df["NRO DE EXPEDIENTE"].nunique()),
        "duplicados": int(df["NRO DE EXPEDIENTE"].duplicated().sum()),
        "riesgo": df["NIVEL DE RIESGO"].value_counts(dropna=False).to_dict(),
        "demora_mayor_24h": df["DEMORA_MAYOR_24H"].value_counts(dropna=False).to_dict(),
        "medida_binaria": df["MEDIDA_BINARIA"].value_counts(dropna=False).to_dict(),
        "tiempo_emision_horas": {
            "media": float(df["TIEMPO_HRS"].mean()),
            "mediana": float(df["TIEMPO_HRS"].median()),
            "desv_std": float(df["TIEMPO_HRS"].std()),
            "min": float(df["TIEMPO_HRS"].min()),
            "max": float(df["TIEMPO_HRS"].max()),
            "p95": float(df["TIEMPO_HRS"].quantile(0.95)),
            "p99": float(df["TIEMPO_HRS"].quantile(0.99)),
        },
    }
    (outdir / "00_resumen_base.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def descriptives_and_consistency(df: pd.DataFrame, outdir: Path):
    # Indicadores directamente vinculados a la matriz de objetivos.
    summary = pd.DataFrame(
        [
            {
                "indicador": "Tiempo promedio de emisión (TPE)",
                "valor": df["TIEMPO_HRS"].mean(),
                "unidad": "horas",
            },
            {
                "indicador": "Mediana de tiempo de emisión",
                "valor": df["TIEMPO_HRS"].median(),
                "unidad": "horas",
            },
            {
                "indicador": "Expedientes emitidos dentro de 24 h",
                "valor": (df["TIEMPO_HRS"] <= 24).mean(),
                "unidad": "proporción",
            },
            {
                "indicador": "Expedientes con riesgo severo",
                "valor": df["RIESGO_SEVERO"].mean(),
                "unidad": "proporción",
            },
            {
                "indicador": "Tasa global de otorgamiento",
                "valor": df["MEDIDA_BINARIA"].mean(),
                "unidad": "proporción",
            },
        ]
    )
    summary.to_csv(outdir / "03_indicadores_generales.csv", index=False, encoding="utf-8-sig")

    by_risk = (
        df.groupby("NIVEL DE RIESGO", dropna=False)
        .agg(
            expedientes=("NRO DE EXPEDIENTE", "count"),
            otorgamiento=("MEDIDA_BINARIA", "mean"),
            tiempo_promedio_horas=("TIEMPO_HRS", "mean"),
            tiempo_mediana_horas=("TIEMPO_HRS", "median"),
            demora_24h=("DEMORA_MAYOR_24H", "mean"),
        )
        .reset_index()
    )
    by_risk.to_csv(outdir / "03_resultados_por_nivel_riesgo.csv", index=False, encoding="utf-8-sig")

    by_court = (
        df.groupby("ORGANO JURISDICCIONAL", dropna=False)
        .agg(
            expedientes=("NRO DE EXPEDIENTE", "count"),
            otorgamiento=("MEDIDA_BINARIA", "mean"),
            tiempo_promedio_horas=("TIEMPO_HRS", "mean"),
            demora_24h=("DEMORA_MAYOR_24H", "mean"),
        )
        .reset_index()
    )
    by_court.to_csv(outdir / "03_resultados_por_juzgado.csv", index=False, encoding="utf-8-sig")

    by_judge = (
        df.groupby(["x_juez", "NIVEL DE RIESGO"], dropna=False)
        .agg(
            expedientes=("NRO DE EXPEDIENTE", "count"),
            tasa_otorgamiento=("MEDIDA_BINARIA", "mean"),
            tiempo_promedio_horas=("TIEMPO_HRS", "mean"),
        )
        .reset_index()
    )
    by_judge.to_csv(outdir / "03_consistencia_por_juez_y_riesgo.csv", index=False, encoding="utf-8-sig")

    # Consistencia modal: proporción que coincide con la decisión más frecuente dentro de cada nivel de riesgo.
    consistency_rows = []
    weighted_num = 0
    weighted_den = 0
    for risk, g in df.groupby("NIVEL DE RIESGO"):
        p = g["MEDIDA_BINARIA"].mean()
        modal = max(p, 1 - p)
        consistency_rows.append(
            {
                "nivel_riesgo": risk,
                "n": len(g),
                "tasa_otorgamiento": p,
                "consistencia_modal": modal,
            }
        )
        weighted_num += modal * len(g)
        weighted_den += len(g)
    consistency_rows.append(
        {
            "nivel_riesgo": "TOTAL_PONDERADO",
            "n": weighted_den,
            "tasa_otorgamiento": df["MEDIDA_BINARIA"].mean(),
            "consistencia_modal": weighted_num / weighted_den if weighted_den else np.nan,
        }
    )
    pd.DataFrame(consistency_rows).to_csv(
        outdir / "03_indice_consistencia_decisiones.csv", index=False, encoding="utf-8-sig"
    )

    # Pruebas estadísticas complementarias.
    tests = []
    contingency = pd.crosstab(df["NIVEL DE RIESGO"], df["MEDIDA_BINARIA"])
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        stat, p, dof, _ = chi2_contingency(contingency)
        tests.append({"prueba": "Chi-cuadrado: riesgo vs medida otorgada", "estadistico": stat, "gl": dof, "p_valor": p})

    groups = [
        g["TIEMPO_HRS"].dropna().to_numpy()
        for _, g in df.groupby("NIVEL DE RIESGO")
        if g["TIEMPO_HRS"].notna().sum() > 1
    ]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        tests.append({"prueba": "Kruskal-Wallis: tiempo por nivel de riesgo", "estadistico": stat, "gl": len(groups)-1, "p_valor": p})

    courts = list(df["ORGANO JURISDICCIONAL"].dropna().unique())
    if len(courts) == 2:
        a = df.loc[df["ORGANO JURISDICCIONAL"] == courts[0], "TIEMPO_HRS"].dropna().to_numpy()
        b = df.loc[df["ORGANO JURISDICCIONAL"] == courts[1], "TIEMPO_HRS"].dropna().to_numpy()
        if len(a) > 1 and len(b) > 1:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            tests.append({"prueba": "Mann-Whitney U: tiempo entre juzgados", "estadistico": stat, "gl": np.nan, "p_valor": p})

    corr = df[["RIESGO_ORDINAL", "TIEMPO_HRS"]].dropna()
    if len(corr) > 3:
        stat, p = spearmanr(corr["RIESGO_ORDINAL"], corr["TIEMPO_HRS"])
        tests.append({"prueba": "Spearman: riesgo ordinal vs tiempo", "estadistico": stat, "gl": np.nan, "p_valor": p})

    tests_df = pd.DataFrame(tests)
    if not tests_df.empty:
        tests_df["decision_alpha_0_05"] = np.where(
            tests_df["p_valor"] < 0.05,
            "Significativo",
            "No significativo",
        )
    tests_df.to_csv(outdir / "03_pruebas_estadisticas_complementarias.csv", index=False, encoding="utf-8-sig")


def create_priority(df: pd.DataFrame, risk_info: dict, delay_info: dict, outdir: Path):
    priority = df.copy()
    priority["PROB_RIESGO_SEVERO_ML"] = risk_info["model"].predict_proba(priority[risk_info["features"]])[:, 1]
    priority["PROB_DEMORA_24H_ML"] = delay_info["model"].predict_proba(priority[delay_info["features"]])[:, 1]

    # Puntaje de apoyo: riesgo potencial tiene mayor peso que la demora potencial.
    priority["PUNTAJE_PRIORIDAD_ML"] = (
        0.70 * priority["PROB_RIESGO_SEVERO_ML"]
        + 0.30 * priority["PROB_DEMORA_24H_ML"]
    )
    priority["PRIORIDAD_ML"] = pd.cut(
        priority["PUNTAJE_PRIORIDAD_ML"],
        bins=[-np.inf, 0.40, 0.65, np.inf],
        labels=["BAJA", "MEDIA", "ALTA"],
        right=False,
    ).astype(str)

    cols = [
        "NRO DE EXPEDIENTE",
        "ORGANO JURISDICCIONAL",
        "NIVEL DE RIESGO",
        "TIEMPO DE EMISIÓN (HRS)",
        "MEDIDA OTORGADA",
        "PROB_RIESGO_SEVERO_ML",
        "PROB_DEMORA_24H_ML",
        "PUNTAJE_PRIORIDAD_ML",
        "PRIORIDAD_ML",
    ]
    priority = priority[cols].sort_values(
        "PUNTAJE_PRIORIDAD_ML", ascending=False, ignore_index=True
    )
    priority.to_csv(outdir / "04_priorizacion_automatizada_expedientes.csv", index=False, encoding="utf-8-sig")

    summary = (
        priority.groupby(["PRIORIDAD_ML", "NIVEL DE RIESGO"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    summary.to_csv(outdir / "04_resumen_priorizacion.csv", index=False, encoding="utf-8-sig")

    # Tasa de priorización de casos urgentes en el test del modelo de riesgo = Recall de SEVERO.
    tpcu = risk_info["metrics"]["test_recall"]
    pd.DataFrame(
        [
            {
                "indicador": "Tasa de priorización de casos urgentes (TPCU)",
                "valor": tpcu,
                "formula_operativa": "Casos severos detectados / Total de casos severos reales en prueba",
            }
        ]
    ).to_csv(outdir / "04_indicador_tpcu.csv", index=False, encoding="utf-8-sig")


def write_summary(
    source: Path,
    original_n: int,
    df: pd.DataFrame,
    sample_size: int | None,
    risk_info: dict,
    delay_info: dict,
    outdir: Path,
):
    r = risk_info["metrics"]
    d = delay_info["metrics"]
    sample_note = f"Muestra estratificada reproducible: n = {len(df)}; marco de datos: N = {original_n}."


    text = f"""TESIS UCV - EJECUCIÓN MACHINE LEARNING COMPLETADA
================================================

Archivo de entrada:
{source.name}

Registros disponibles en la base: {original_n}
Registros analizados: {len(df)}
{sample_note}

1. MODELO RANDOM FOREST PARA IDENTIFICAR RIESGO SEVERO
------------------------------------------------------
Accuracy prueba: {r['test_accuracy']:.4f}
Balanced accuracy prueba: {r['test_balanced_accuracy']:.4f}
Precision prueba: {r['test_precision']:.4f}
Recall / sensibilidad prueba: {r['test_recall']:.4f}
Especificidad prueba: {r['test_specificity']:.4f}
F1-score prueba: {r['test_f1']:.4f}
ROC-AUC prueba: {r['test_roc_auc']:.4f}
MCC prueba: {r['test_mcc']:.4f}
Mejor benchmark por CV: {risk_info['best_benchmark']} (AUC CV={risk_info['best_benchmark_auc']:.4f})
Modelo principal conservado: Random Forest, por estar prespecificado en el proyecto.

2. MODELO RANDOM FOREST PARA DETECTAR DEMORA > 24 HORAS
-------------------------------------------------------
Accuracy prueba: {d['test_accuracy']:.4f}
Balanced accuracy prueba: {d['test_balanced_accuracy']:.4f}
Precision prueba: {d['test_precision']:.4f}
Recall prueba: {d['test_recall']:.4f}
Especificidad prueba: {d['test_specificity']:.4f}
F1-score prueba: {d['test_f1']:.4f}
ROC-AUC prueba: {d['test_roc_auc']:.4f}
MCC prueba: {d['test_mcc']:.4f}
Mejor benchmark por CV: {delay_info['best_benchmark']} (AUC CV={delay_info['best_benchmark_auc']:.4f})

3. CELERIDAD OBSERVADA
----------------------
Tiempo promedio de emisión: {df['TIEMPO_HRS'].mean():.4f} horas
Mediana de tiempo de emisión: {df['TIEMPO_HRS'].median():.4f} horas
Porcentaje emitido dentro de 24 h: {(df['TIEMPO_HRS'] <= 24).mean()*100:.2f}%
Porcentaje con demora >24 h: {(df['TIEMPO_HRS'] > 24).mean()*100:.2f}%

4. PRIORIZACIÓN
---------------
La TPCU operativa se corresponde con el Recall del modelo de riesgo severo:
TPCU = {r['test_recall']:.4f} ({r['test_recall']*100:.2f}%).
Además, se generó un ranking de prioridad que combina 70% probabilidad de riesgo severo
y 30% probabilidad de demora mayor de 24 horas.

5. CONSISTENCIA JUDICIAL
------------------------
Se generaron tablas de consistencia por nivel de riesgo, juzgado y juez, además de un
índice modal descriptivo. Esta base NO contiene las respuestas del cuestionario Likert
previsto en el proyecto; por ello, el script no inventa una medición perceptual de
consistencia ni afirma causalmente una mejora antes/después que no haya sido observada.

ADVERTENCIA METODOLÓGICA IMPORTANTE
-----------------------------------
Se analizó exclusivamente la muestra n={len(df)} extraída de N={original_n}
registros disponibles de enero a septiembre de 2026. El valor N=1595 del documento
anterior no coincidía con la base entregada y requiere contrastarse con el registro
institucional. Para N={original_n} y n={len(df)}, un nivel de confianza de 90% y
p=q=0.5, el error teórico nominal es próximo a 5.11%; no se debe afirmar
5.00% sin actualizar n. La partición de prueba es de {int(r['tn'] + r['fp'] + r['fn'] + r['tp'])}
expedientes y no de 447. Las seis figuras antiguas del Word permanecen sin editar
y no representan estos indicadores nuevos. No se observó comparación antes/después,
por lo que no se atribuyen mejoras causales al algoritmo.
"""
    (outdir / "05_RESUMEN_EJECUCION.txt").write_text(text, encoding="utf-8")


def create_html_report(outdir: Path, source: Path, original_n: int, df: pd.DataFrame, risk_info: dict, delay_info: dict, base_dir: Path) -> Path:
    """Crea un informe HTML interactivo, autónomo y editable."""
    import json
    import base64
    import zlib
    import math
    import pandas as pd

    compressed = (
        'c-q~4U2ogSmf(AS1>@v~VnoT*w=9|VrW5z=%ycI~obG!uZcRc-l*L3MH6$h5Een{}eVzsOd4R<Ni^V+6!vHgHxBtofl0E0t2dhZRc5dH`U2K'
        '3@WU)R@ojP^C>QvS7<7qTWZ&toBPnV06N5}YMEWGg2J@AtQOz^!a{4D*{GbVE{PW;q8Sf{gE|KL!hhu+e64>o~+wTj~Oz?ei~>W7f;Dww8oc'
        'j|9~iC?1^+X%uS4ZKAynRpA|ZBQ*~kS_d__kOZoq~0`2jH$mcyztk5UIa;y7^~Qy29q?{MDW4zuUGI2_4|qOc?}<+Ha|H&WR*NRPSP9v9mG+'
        'TZfmvB+uRJEG<wZRYcj0Wg7C}WNwd}Jv^(&&T&MnY@TBYX+WkJfFM`k?JeiLDjz5Oi@#WyjtUc?^j^HI($1`sNpJ%ffep{^JxjE@}`(ysvgr'
        'XDQo3-IR^}Pi&+v#_DRCo^kx*9k}YjtfjI`HR-Giy0*$MJTLo^Qv|bu9^g3Bt?4IEtr!T!WMy4#qZ_#L;3=8~bx_6GZVKSwg?&yKywV*)F~K'
        'G6)CGa1t%x)5eP{)I)1H_9kC0<7gdDdFuGm8qQ!`295UW`mj;&bc};@|1$E8A73BXFJeHVop@nVOZ+&P?dA=8-n8c}dnFs$jn`=!g?13G)~T'
        'KN3xASssmdUnL!p#s*iP1Q0?@3Y0LCchse@^dc;kgX6`#W>t$B+@bmdQn(aM_y>CK?t**&T!bAPeeLiW|dyBS=@!E}g!0HaGtN&OmhcD)RfL'
        '9@{ZJT8N9>0MVGyD^KcA-vV*esDR5jyld}zN;s})E|5CR>25TgWqm)tUd8t{uIV+6(vBK0BMp2lP@>JG+F^>)yjs^#*b$J^1xfC(QxHWr#NC'
        'w1f$VgUGM6P=rSUl;Ar{-9B9IN?TS^?cbwtEPXR5p1p5r7>rSKNFNa8Z67o6_r>NaA8V=Oo>_SzNHUOi`K?9PLXc0_}CoR9z>ULQBMzZu21d'
        'bQc<csL4cQl?&rb7k7W)4E2$RQi?8X};Jy+Rb$7T(xjsC@&d3VmsGp)0(tzVCVCDX(nMM4E3P$VWw-$Wa1TvdIm-w;Hqr5SYW;^7^MoFiZVm'
        '>R+d|sXvKgk47(yLSG_}P=~D!b#!n<#@;koC)i09C3<s%IgYxmB+R18I??E<Ic|I1Hb+z%rQSl!nih2%q3G+4Al`XL#{J27+E+^(Ng7AtrGo'
        '8du@9{D)jUZ3U47{VA+&kTxGYTaJ`N9mah%_YVQ>tMbX#3EgDUNMXLDtD0b0xGA#6C2SaApDB#;8mApyW_;a?9wuVJFhZfcy{50D6a;8W?9?'
        '}tNg5nP5ffNhyj_F9s9aZ1D4M9?^maZ+v)o%uKZIF7EiG^A+^+-MfX%fWiJ^5cn@_(gK-WaQR%dd;I|VYcN6R&N%QfYm8@ZV^>$bFtCOw_B_'
        'CI#icHclXH2hO^Oaj$6P%oB81@)Y2}r)smI;i9_YUMJ&x*>LhIECR48?gu#<$yWQw?fvnbJ5{+Wsbep|i|45zWG(oc70NQXgya~>yL?3i@SS'
        'Kih;{qMfX!{|Xx<vJua23NF{!tq0w5g$S8gAKyR9V`jH|<VmxtR(~HaE9h#c8GC!%^_jX>@0*WTd$>q`x#OJ}?gikw9YeUYq71COWe|eXXzJ'
        '05Eu?vw=pp(VM9gQ-Ci(pbT0CCCxOtz)g-=t|XeJThZeV=bnP%`@IHeEqW7NcTrH%;5e>NL9K(PgedSjjdm|r+MiAP(+<mylqtr=QPD;-bLG'
        'V_*m5v>3Rul<zu#auBvx75fEEX$->!q+mg;HSXq}SM)yzh|-8k~bogx2Hi)J%WIxz##vQI?0RfxBRQ-CW;N^-NU<gc*_@U@hzb2KDpHsA8}t'
        'j*D4y9TfVJrfALFRLJVh>{cgE8j~iZClfGECvnX1ZpZyAExvn{z`N~uE|=t$wdrO&aNY_)MzRdDKw<4GJTfUM=utEV$>I7O-*QIl+if?f^BQ'
        '_v)gfav5lt|YnEwFf;S$uypE=&l<6YCUS%BsX5hhZxL$&yIvJ$ic)jpqd{0>Oq|_GCsFM1g8Nh(zDzGx|v=rV6k(L8h%bJH(d{l%x%5&vWYK'
        'pn>Y;)>hofCRYmcaEGRi_kupw~ERdq--QpUnER2^c<NzM`77tD{Aj){t_)f|i@5v`qRlt8e1PQybV->e1t0h`JkhkglrkWLo|_m`?prK!ujD'
        'AO9toh&Io9IL~mhH8TXTri_B9mk5GXV0by%jCC`8eRGMNNs}AME6DKFjTo8PfZv?WppY1!|D-!2A@cWDt0Q%)Ty=F&V;407cdL|OWFs=6qQ|'
        ';8^OnKlW^k8DO46gY@<Kl|Zb;eZ<4la8ltd)12@zySsf6tTVNccS>(R;&b!32s9fSc5j1ZumUa#5j-l=?KT#7UPBOIO-?cf>{Sz9q0wd)MK8'
        'RB=8cbf_6g||xlfp}1dNQYyW&Y2ob6gpV=vsCZ2gQ|?+s4c(i^tuJub6mVcQA+1_I+b*GB#f=ScB<d^{aJIDqW1!r>tNv}K+W@D0jzeavm!<'
        '+sYG|H(;jywkU#^qEwHBPkUJKz+sb_I5Y6klM2#@SYAVz7CbQnmlNpNk-fT8&Py53X-DNyI8h83w6bzX(!Z}(*+fW&q7_rUAv86Hwtdw|S>@'
        'NYrdo?j(b@z_Oz_}GkTCixeCOTOY91}u61e>-E!Z~BZs=y^?BtIE-{L5y?pJ^j<?kUuoY|f%{rRi`ErnRZfBZbGbJC2@{O`?n>y3KrssbV{~'
        'x+<4sik=tVT!9$&wf1Kby0FTuq<c-GBLB&mHHx=K=WAjLWf;H1-@}+7IXSJnygV%_T9R6TqGh|Ue8%XM;<Z-MvL?C;qZ;CCI|~*IFg%a?j<C'
        '=bk%lhhd)gxt2DFLvM)I1bVsUr9e&fg~xNug`>K;wB*#kpN*Khp(tU2oqHJRyTqrkNYYBt9wp(T_WHKYd(KEp9IG`E|3N0pmxQvK?;k6L+_U'
        'JpW=$IKIV?+AgQiGaOsrC)OvT>I0Z@VKBQL(GRgX7o?)AhgWwet&bNDOgV$e(R_`F0*hse)_tS<)GWkK)z<6+72wd)ygQnEliAGfa<3HH43t'
        '!D%yD$>V6zYI*ac0TjN$wd{5@lRZd<%sxPCd2O0?Rv>XnC5T$@4+if};TYcj68eX%1)YgD`h_c6%We-UmQvlI53uFtxQ9#d$F1H20L{5Ee<u'
        'OazOph%k^c&-5zwZpQ5$vD?&_>N5(|^bZOWtbyqbhPooxFEjz-1KuSo%z-F7mu9>0@g2`u(HajLfqLl5VrWUrdWG-)=E`TV^*3zUbzDHyE(<'
        'UUw==vc(Jj!q+f_*vMw5=7&0(_I&3KKU7a@Jq6EFrkx4C<G}nm(tx?Z<u?~Nq(9m!YPP^>xqy*o#9m2L?xY~DLnF3|eO$29gnB6gN)>78dea'
        '114@FrCrlh|LhoUG)Bo#KuX{|4^sM-_2OkLV9oEaGcSGc@<fWHT^sQA-5&92w;O1LOi#ytdVNC`4^upa%FC8&yCMbX}y=tZ@*d^xM!Af^r*U'
        '<7B&RC2Din6r<&_SLl}?mW8F#%|?o?A@*aVZWmSn6q{SII7_%nP}`olfgz=W@*XAuWZ?>&%KpGaaGrmY;~zb=lEqSL6YV*&Fs2JA;N!~Efxu'
        'Qx&ocKr3~I*G0RHl6I;4aXUfsTMxjZ_*W4s~2mwhzA84}IQ9^bKixkMmO9r&VDeexqpdChT=rq*ju<#`oJaLZhm@#yBNTCwLVIF<)q0bemot'
        'PxXDh3^MOAFWqmYG-j?jlZGjGCQui`a@SZ%aeW))aOdfwWcGY;%1{CCA6X5<?oiFfA^zX|A_Y*R+Z{Hu#)v?{Re5+?OkbJaccvzV`(b73}rE'
        '+eE>%L`qE?)1&EB_vR>^waHdmuzJOv+iT{QQ}z=;iP<d_G@4E&vRWtE7E!?QA&<^DKIDNVNCxmT4K~JP;U$TCKtqn9Cm!}#rm`_VI8n1A$2m'
        'B8Q4%O}Oh(bkS!t|@@uz<?zFT`sFTC^sBFcGu2=!`}XWZ)G<o((CkKccI@#^h4&j;M;<k&Oju|IPUp2U8#Ui#s|$?Fi>fT4>qAjMO1T*lrkm'
        '_*6J$?xM||Ct{33_ytf0>A&j9`vtMAI#ifg*}5TTA)n!K`#QcEQ*6)ya^EH!O1)M_Upff`HYA(%!Uy^@02&ncSv2$<Apdk0K4{+H1>?xzYOq'
        'U>K(JCYa^KQoQVfK$Uit~wwgHl_)#GHkVY7zh+t}o>hW0w9{I6IWn^|h)k<t1=ZB0hARnBF(Z!K&I?XO`aNgh*;H0i$OKgQkhAbhCqq6`SjM'
        'tD%U)_T<OpcB8*MCXZi-@9=SRMYsZ;e`-H+1Ck#oB|Dmyi$up|BytD&oLTFucPDW)<0xvWx)rehBKY;V+?0o)5X~)I)FphA8zXtcDw7r4m67'
        'sJa*v{Hx7J&M*PpERsj01OGJmA|@M`^=c7$(;CYK9RQX9NCWPMq{&iC>R@FPZ{yc6-U%ki8dxC|{O-dK-#@I}=LFk>xsI&*p)HaWlEA)RVpo'
        'DVE5^~m$PmNI7=IV_ppa!1!4q-!0|QrX7{_oA6ixt`JLIK{_3&3#RmLSWnot)L;nTc%at17V&ByVQF%%C66c~rG6buCN9fMjVn?irc!O6G5<'
        'vIrH*g&mO%&8Ewm9W4}bTRmpcZr~f^-Qw2dph<qTCTh}qfl>g2|%Vx(Vq-NS!<1EZ1S}b=`8ggu=aA?B23wL|IdU>=iK)|rZIMy<D>(D8Ut('
        'wShF|%^<OZYtsGR=P<r!c!>J06dof57f)Hr<8Tl5;RP}OY4S|lf4pW*+Ia<v?VQud13#xyaM9F-1ss^!_Q>!|I&1DL!5S1AeB^vKUph%rO$R'
        'rrOfqn`C1x6-aCxT-DsQ^5n$e}#R*TOW=8Y!x>GDU3_RWjR`S4rQ65X;!o!k><BWR+rcl!4=w;E1V>3z4L8bR4(U$vOH^kOn+s9YRx+ISM?S'
        '!(cH_q8Q3Ukz@@F2dV~cXm}v667tEz&y6u-{0QXFN|!t2BLPqda$^?75*&1vz@YKAUY`GC8-M)%{EtRt{LT5>H^+xm;^detcC{N+&LDrw0ek'
        'i)6MvPu2lYv^vFq21<l3&2G&wAjgbX4kcjms26;cA}w1~XSa43bF3W0JGeI{moc0KVI5?bbrQo-ffk%aNneu&E20zr6*rC;G-0nR~+ue}8@i'
        'ujHeMWMTz<c^~QwVX&Rr-ph-ExL2-V6T+s4NSanf_nH%`t{;Fq`R(@i1JZ53*zO0A)=5`;FGpc5csMV#5g&th=;ACu_kE9Tdfu#c(DtIOb7Y'
        '+@%`Dmx9>lEeD&hPi;o`xk5q%K>g18RPJ9gS1a4&x9|2rRYCJVuXv1(%jA=AM)i$leFq|!XeEjC-^|TU9t>L4EpBl(Ge8`T3USDB2e&uP!RD'
        'm<CIxa81WC|@WHd)8GNXR-3#l!VtAzGg;Q@0Y@Q@3fkCtH@bao?<$V?VBh)-d+dbsQQjp&lgPVrzb7W1Vim`uje@IR10!P<UA}{iOEp%(T(S'
        '43_KVw=o;wSHWeFCWEOBCTUsv&Kj8i`H%n2+zFV%_0qitXo6V<Q60^UYu9znkYF$^4$}1yzT*e%&h0Hefnn;;KzvL^o;Ti`AtBD?FkS<GfI='
        '!(9aQ**H}NZnKPQKmwrN^eT~Y_OwOE4`ub8%JJ$v?8O{trfmAHv43R!GS)R9z4EZGy^P23aSbaHwMtz_+9JUc!)Fh|h%#678O&1dGod<N#y5'
        'Gp;!=S7N-C-`_tj|cep59<hD4$K31dD1!>n!Ah1$g)I}$zpw}wrbVGC|=^^2Y}yv`$_}h=e5&voErXgXCsCO<@4WrwO^dt(WrWO8QA7WBrX{'
        '8D?fhe0qMymk(<83T_EO%rcpDFT1Knszyq5NXT}dNUViub&6)B2*^BpYUcdQ$9aViv|L9OyCGIv3k}m`Ey=vx|HfVUjQ_WY*+F5!xQH;#9**'
        '4~okO0lloTw91gRxOwZg01wv3@}Dl?=8!o0J3P6`!ER-(tdLL`#*Qul$L-ePm!?$dreML^FEFCJ}tQxG<S&41@bZ=(^^}zAz`7k8w2l=&dLC'
        '>BpDG%bzG6T2DWM+ybFFxq(k-*EpBb^)&kmU{3)4D1^`N*fVHa*>_0t77O@z?uSV*4i;c8D#d3F`1EZ<$*;k&0;OW?nWNFpeq_+-<odaHsQ&'
        '-xGLG!`Fox>LW(1V@m%sjdXoR5CEi$7$>-wA5Kb?JVygD=9zdk$v{aa|@O)e|KppY8%H+v%p10NVU^f^LPKl;eAYkr_~vi~EaGPfu*a}#`Qw'
        '&#$856uY<34M*g2E6o^SUHZzTlJN77xeBpE$u4E664o@h5TUz^8wJptpRV5b@%Q2-@kbC*7%zr-@iV8_4?(@*Kgmv_#PVi^`A=1{Xf=sB!#T'
        '&?}!RCy9(YW5&E}KQlp-&5IH9ro0nsK!=eFMWY{;wK%>u>NTHVu4Xcl&Fhjym_-m|qj~JPM^cJhR_tE+2bRdTu6b`pA?6N6|wpTD-OS@Wg1!'
        'XY5d;j|F``52ty!!aV_XIiS5;Nstq#Nc3jZ+a!JFrT_Ne0wQnHDhT1^KL4|9l0JG?zY#iS)2Wky5<G`<NT!50HjMwPa#F3M?PVdV<^<fGm;q'
        'CLrey)R7T~;_opo{m_p|Hpo5%;xkk$vr^02C{xNhB$V6ckL)31J6vf69XM-OBrzOP^zfM5t9t~F$U}|tM{GC9szNB0H-*=KB?#K2`^^v_4&F'
        'zTY(}m_v~J$NeR)TBXDdYSx85Qlo`q|en~0MLefr6dr|9G>exAc<&Z7m5#c~|W@h42|2c*2ye+4*5$?#xiUrR{n^%oomw0YLdDZC#!vCk+S<'
        '{nHapR?f5zr+RJhzZka-H-{ynK*Frt4vUJf`@@sj;g-~RER75ps@4Ief2dMGAhntgUJ`BAI?Ra0r#}48KSYrwT+Kxqxv5?SWGZvU&(^?jfRT'
        'DJHfqQ%lf|n`6FE!e*G$O-;O`~leGTDzey@mL(Xg!RM+2m>71+oT$SW1A_{g)g`WvIh6Av&0X$XfRI*ZXK5ve17HgJF#}_r1XBV|=`6X2q>&'
        '<+UiKsZAQ&g#E6<n@ZC1|Chs$2$96U(9Sg<>U3`XI4b$R?_i`4oj0rRtajo0+9T7Oo^JpIHigVmg%ag%aR{R8o|0B8{rdGzP6+nGBf}L9FD-'
        'Qq2(%<ZK->C-$YEe!x5xAVzD;P4<ieFCcjYw(oQerf(%0f409^?g{)}d>Pr4&Dv#^B}!#BI$o?T`WaSku0k8k7M3=gMR5i51g<j-j#u@CA6}'
        ';OT4NYgs}^R1{}-!^U^E)Cr!328<YF62`!Leqf)b1*Rf(gk#Nw_S&!yMf+l!G^UwW%bjJ}1NLAYL4;ySVc%T6Ysr|^_JCc3q=tU8$XsUKr4c'
        '@I!rzp<V@E2dKImNjJXymLLJ4k+WgZo}dov)`0_b;MWn$2@*_d;9ne_vT!Tc%`I610vjudc7X{SH`)YRy@&N3BstJz*MbNuI(Eb$(!1>WgB!'
        'XLE>I)K^_krJ7s@`gdH^M4*qNGMl@WUnTgLvG?Q%$gr}>r9zJ`<p*V46Y1!*pSrO;1B0~mteYH;J6#{*>I($RWr>nzvSqFC3aK|Z|712rN#d'
        '=a<=^4Es4tXL`%}9W@DvmkQ%&HFUTO~u!KBoB5(lC!7uBjAmYpdMQ<Lxa=PP1exilgrFZXY)be!TV4b(EO|ljqoz^aAq;GE`jx#e<oYH41Mq'
        'jTq?9v+vksRryXJ95#hR4{B2o$XH_OwEM-qxNvL-L?`}Li6=&)kyDaqzj4+1xN^l@WKx-M02Ki|()WmSEtj;itY0o}Mi;A5wJJyb7X@A%mRE'
        'iZ2Gg`+Z8MO*AV?QNATWkMV*G2#J!niE0OzAg_#%$I8yIN%tt{(6Nicqo>478Pq8C?XOUt?<*sRiu7)DuL>^ZBBy}LQ2=PKRpv`#AAo-Hz_K'
        '9g`2R*j92i-{W&+S!NG`h_#1KLCt_Gqm`v!N29oHI#Dcs^l9hCt24fnEGzTv&S$q#;4xlc!0#shHsmO&^9yyHQC`H4Xd+Nbrx3PO*p^?tcFU'
        'uszPQZ2hfr5f@OK29>`$X%mJ254=sDv5NXZlkcxj;g_iBDCj-th5NkV?llUPRDu4U#by5-7L}GGtI9Exjjv4rv+Qkpp0@FzDtb}ru*22C}HN'
        'p#we-n;&3jBpRs=5it+q?8)bjHk<X$9D8v|2A{LC1D1P)Fn|>BKzRNn~b`O_Y&6Ql^(t_!+H1=4D|nAk0s_3~z54oUDFSg{Y9`y+T(r59bk0'
        'd0>-xm;QI4R3%o`L|e&J)2Na&5d-v}|Mh<`dx{ml3etHHHrvk|j)R4N2Zco%70g*aym$EKzr<B{*2V0d3ZY-LXf?^60`b@_%f^)%%&odz?<&'
        'T~6d|=699B^UZ&0@wu^_|fJq_+Fwy(j_@X^zXDPns}3%G9R#~7335-E|cQMq8z%s{X<>UuDtKOK8<8cY_x5-5}lS{yG9PM&W43YE2X#ju-=M'
        '>(LzXV2izW7lPra{7s{X{&)oN=;&264(C@3<@j7DS&yP8u*M8%V$L(0aw%{9h{cX%7ACmDDAX_A_sh_31;lPp2ol>Dopu3J#|1A*yRK1q&Mu'
        '~@Z{3~pyEg<@3Vp#+$+XWqeE~0&Mb%4{Em#`f#x3M&3zh}_{azB<daqZ3|deHR{~+|mFPu%P1Vg!f6SbPsO0i~(kOjkJlz&~pCZt>^xI`5TP'
        'VWlwxrCbJjwpU+K+E&NgiyW7mEch6yuH)Nf{%^1X{tFH<?#}mQMUd-J4F&HURkdWMqL7XinyMZis1DaCsP2gwUmh=a>Fs!Ph&X9Fdj{4_G<a'
        'M#OxX%rB^ksQMi{);5eV*1lrC4lhA_rI3N}*MDCEo{SAH!A*fPU<eB?uV(UwOUj+X<1cdUj@(>=PAuUeQ1>8&ie{)aLzQ6Y3{~51bNLOfZ(i'
        'ARn@5J)lSSeJWeBtJ(=lWcWchRpUxCW6mx~Zy=V`hc93Ea>UDdBz^(ekPY=TK~2nApY2L9DI(Y1Tv7!Hi=)eip5cOr-<SYGO3gd$ov3)eG5j'
        '4I9p4jd3MRDSYF=>jc$p5d0sN@nWvZgqj#wOWq-={Uv!5CEppJ^+=Je%P6P2c2H?z+iFW?m^3O07A%jbbw2}02kZrGd`e`s=oLbqCE4Dd>;d'
        '6ko>VyDuuOn8!~W-?+86G*cRvzl-(GnbugV`ht7kLMHjIEB9P8JxG_3D!~!LC_q!6L-qDeB<io3oI)~5AJuUs11{e%u{ONylmPF0|=|7AJlG'
        'He-KmA9YxBW@A{^!1zf&@6<TnfdKW~YS_3y+2N$2D1weRV{*QUa3(bdpSAlq#s5PiyFDjgovq)f|y%+7xX!jMx}}C|`QlE^%ty9|X1^LIJ|h'
        'zH*T;=5DLm0~M;#bnI)#ZS*?!jpKHD%=+3~x}W4AJl$S9Rit|uh-Rl8(_(Piy>6#B)(5A>U#~cs^<Yp>n7fi&LSBB{n=NYX>Gp=TEKdn@58?'
        '<JLZx=(7|oW!b69^-sL)yy)G&Nx=>D80nV%x<Go*cnw9k-6Tyx_ho-cyu&FbfoO^4OFn?o1^o0jt2dc&#K`_6Ov!s+AM1q6S7p4c}o^d@+o3'
        '}FDOZZ>aW1a<-;J2`hjVANW>Z2F3KoWUK@_xN(`Re%XJoQ~aSwe6={f}@^B-{KzGO3Omz$>}|t&D6OK4Oc5rE11d>fNFBstaf+%FrJ~f+03E'
        '1kF%M9^>j-JIH`A>QxjVS(fy<`Y4<uabI0^16u%pgyQn&`vDiH_IAV0tRDg4;&F4wA*?s_c$TS3fjJrT0*ep$N6eDCbpPFx^@;WeuzP-y+D?'
        '!nycb+GIH8{=%5GNW4NS-AqsWnuLF#Kp>fSmm+;zDm;Am+CNa0e22JQ+<s>Dy(tw^tpb_EttYI7Y3O0&C0J?E|al^t(p@R%p1ub&xpGxKm-s'
        'iC@$z;MvkWb0|u5&m4dpOkLG8m-%zAx%U>`S+;QQqVGy^*_OEPR`u%=zul^S!`yDvV!`#M2`-j}Irlgk>({k${rdLStKY!GjSz0CI6*<uXrd'
        'r#HSKe^*=dkCxde?W=M{H5#6~&G3=TUsF~QCAbH)TKjoQX~em<lz!$5K4x*_>sgN+tyVojOY<U@GAQ`ni`@uxF9d|0c)$tua}U?q7q<#i~uq'
        'tMKGwb6gbG?%>ZNyqQ?`()s3Bp+^Ad!~|;G2ZsQv*@9104rsqw@MS2jWyC7kkY58+X~=9Dh`g#9>HL0*hp6}DweFVY3)FwRM>#+D+5%QFC=X'
        '+1B|f37%St$Mx|_s9CeJQjsCdX6A7SuCP2)w2YBllU75_~W~1K4(OgA~8)}tdjKaCI8k>#o<mf10v?UX;Bz6H6k~IzkUvL1>0H-^&c!4DpS<'
        'bm8tj??L=2wb{Uboq6ADoB~msu{*prAo_mDUdTHdd}+wYr8ZXgw67O-X!%fJ!JdZHIx?%Pf&br<18NU2cf<Gee{`>+6O{*YD08ZEClh4kH}k'
        '(?m1vc8}4^QTG8utIY6rr^oCL(pXBn-D-DQU62Pa*YU=~kdQp)+sy{6qv_nQd5T&a{no4l(-ys8LeA*xId<zlp`(eM8CWpu*gu~mD^H}2m<0'
        '{XAmG0T>GJX8p&l9h_Y*HICSDg(UcdSM`?K@6`Sgz%X>4dYbKzwMlwiJ-0pwE`HRSp`icGc!L$(R-$DFd&EV<k=%0wnv)WTC0*F_8x#bn#$z'
        'Tje1i9)(}1(eEx@MMi^a&gQ));ykxd?7sFy_VsX+v_@Z=xX_GJ_{s2ZT3ljuQq$nLkop$tL@l-a9dDRH}af@4QSz&$yHJg{PRY=(IR>Liz<('
        'q1>?3e6NV)5AD(|cY!83gBYttN=4CH~y4G%M!n23RSnUy>yhOA0b>b5=vDQ%wMH|iz=s1FEH&x-rQ6QXNmDmo1TkFQ{w!y{%ZH4(>v<VWkka'
        'Ek$y#p>i^FcXjJE!bDSp-1Hj@{~8Rr5dp@qd~FN@uv*E4v%5`-OKsb8@t2-Bz!0aKi6QKzm5Qwi9<CU?ReqX%33L=#(0aiiqZPfu`y6KUj(x'
        '+2}Lgo3Xm)v^Vzp<-iOClj5$yP6prcA;JU0VF8~iV5j<rOcDyx$tmBB3?2P$qqoQPaH0edO2$}Zi`0bui*05nR%-l)oNe!cp?nJygCa-x!g('
        'CKT0{v>aQLmm_$7Y`(s_kNy<M9YpXe1RlIiiB_9|QSDhvXQ002`;gtQIJcQT+&#JLS(bOxO|wh*6?(7*(AU-FLeCbYFW@^TWK=(DAO6=0CB@'
        '#rfI7Q>Aji~+J~o<6u+lpShtc`dX@NA%#-87Rpz8t=3*pxsAMD|~*;Htz_W7G2AXcz8t*^e3N3i%4!a1fKru|B7)7tP)iz-OA)iPA>#HMi<c'
        '1h%ru5-*5vIUs>Prh&6KiyGohU`UQtigr`J{>Fb1n>GJ(cyjZr03A>`?6Bf^((f^H=b^UC?Iu82Q*axdT&ia3MrwM#_Gu-)$#5c-6A}D<Gl<'
        'F;?0JN{z-bbP>C?Lp**_3Nv?$Hu&a-UX~WXgSACLxC-jy^vwZvNteJd>@3fPO9dzbqQOB<Nd-u7GZciEEI{30tCt!X{fNWR2?E6)+Gzr9Uh9'
        'GowHFs>H5lajE5P71z>)lEJ17%_hD%;Mp_!cH;1dV+X^R8loybVi6vV5CVvSx2Uu@vhtg13fpM(eKXlMncV&uZPN~mamH}w`w(fjwf@`@ZCW'
        'hak=!mzYpE<dKZ%g{BUuVIGg<gvC^iJ*N3kbR1PdhZCaFugl+Ay?Gl9|kRmkkwGa7AgoKRpAHEdZY4U4og=%6c;v#p1?jBjR)pI|<PytKH5Y'
        'pQD8(icAqR-^-OT(l24!%uL}nvBh5QH!WSy9Lol@ZjD(S6h}<Phv%F0AJbqGe1p*FF?a5I`-!{jX_z!X6qQNOyY2YYv-U&p%O8{bqrJ&=IWy'
        'C4nB((#i35PR~d4_pO~Il?R2Qnr<`XAd!z=6ZLuGud=!DH;D$;9Xqv6@l?%aV&nD!F#9(VMB?cz0&zYdGJhZ^4J!j7VM2p2_?oYnB2Q*j*pL'
        'PtXF+vmU_!9qwE$B~*%k03Irylc}7kD%c@I^49oui2AX%5xX5>)fe)-1|^ADtqz(HX)VCiWu0R2MYQ7Qr$I0XR_LybNG1_2k$8u58N&$Y^*!'
        '?bv{s@J0g+zlblsTiawioxSPw8!G9RW1lcyvTGbK+Tg%L6&b}=jc)-Y3C<^G#o-LH1pTP{@s$(C5b+Y4-Q7)2l1<j6Jjmwd`A@m+>7Rxqp8r'
        '}cmo>=Q3pQECQC@oSRMm?2@C~_Yo{Nr=XP@aYGV*X^Jc^>lAf+?iIK7B~majRsiA@A`e{fyd4oqC{_EzVDb*czv13M$^=~m$(JEGMjJ3nmRI'
        '{8hge!A7zw53L-<QMjT@ZvA)70qOvjAHSb4FSEpf)@2qPWl<yV-T)?>)*KaS|u$D*%6QSKNS;YAU@!Zf>ZN7{8e^(`KBe;gYz}0OXC8Py0%E'
        'J*Ui^Hvb-3>H_*L2*~cth2x~UjTh>DPP9qb;J_n}s$eDIl&Hnvs=8n#09qlFF>=3_$l4FzD$x(R%rf*0Fd>no|5i48VFL<R$$d~|X>Yx72Fs'
        'p2XJ*NO9ETq)}z~cDCgL~*00o8jz3?39=548HJkX9#ZWW`+vb|_UNJ*hRHod5Lu@)J7L(}viYhx}lc&1Frk03`e7vLwbXJ0X&vF+|4!sU@|z'
        'GjaI?!+*vl1i~e!C`@IZgBf2tdGcL0vvD5bvC!`jg2WZO8LVM0+V&-#&o1L_akM<Jpknta#c1(^xlEcC?28h!m=E@@&lVnV5nv(i$%UA@Z2O'
        'SD!->=(ny)7s=V5&33r70Opt)RtdDbZb^9o^Qm^Q0l$N*Sb+f|*gj1gxTepWHDped&J*v<N*1+-N@U1;R^Wp^<8M0Jj>aXs<V7b)#|1`FBL2'
        'o%+H7!1&IiXqK3W*L%6)uymv^{6q^X6XDu+cL%9Mr@mQ{tz%xhHsgY8>I9dP5Ry?whvD`qsvs1vb|K<jy~lGEYTuH%?lXmd8BPmxCo{gyPAw'
        '(#8`K&r+D6sLOl4H=s=VDindsAHUR<+80x}7i@D%PT^M7Ap^*#qQz?I_kCD<W(g3QrUW$>78@V?J*ozCVb#|~3&{>oK#wsS`DD1=+$U}|_c1'
        '0SC8N~WT9l#y(pU8J3sk=V+62zFLBmuU4AUr&`hcq_j6OuTl7y3qSnH(S(Cm{L~UE}mY24`0DdH7^O{t5X`b29}C@#AeE?=-p!@uWsjZ2f#='
        'RTKZB%kj$i;}74T`xtT<`~o(#RZqmrhe)gvL2bw`yh|Sg5UbbA#ZX)lOm_sE`}zgC5cDhAFkQFsX-_FY9kW<5)HnCRF8E{Z*xl|_N}nK&=QK'
        'Pt`K%Z|;nD|G!FE{1C0<XOJC?Q8_s2jMIv$#5A}$L?2K^2sx5b??7o*{yQ<U$5fzrDyUR6gA(4WHV6ax&wRkq_2%H$lNB-6sl&lGlvO?tC~>'
        '9sfoN<}rR*|NQddx?c}@_Uc%p!$WPa(q<A@Ej3fcm8D*|N1Xp-ZAUdlNBwh04#Q6|Cl5<;e<8^JqV}Puz*Na5Xj1OrMgU+9zJlB(!E=w)+<1'
        '-|NdLvrUFgw%U#geOPF@~T~AW;W?N(8nG0#z25Gh&F$1ZCOM-H7lpWBMJM2kug~cF9qa_AFVvIg>O+Joi+*LP>XLZfJ=q_=N@)!A4IveS;e3'
        'Rdq?~y*TNBL#epww9-%gbDltfBRU(pkMJRXAI*5-WOPUuLW+_9a~nnJR{!l5c$M8M&;7@<_H~WISCA6@Yc&Z`fI7{)S7MQ$+h@?kA^pm}u<m'
        'Y(SGzx6G;v#GQ3lb3ok4x*y!!vuBWjZ;Cj>3lB_d3n_S5AJqHGpLy#A>X{iR<HZ^n{SMdV?(256PB|d_oU=vNC8P#_aasq-Nsg|960j`_evH'
        '^xFv0z?x3>>DoXb@BA-2m|)hXeD1Z=KF)nrn33droEa%k>qhHA@qOmQ#dnxWMDn|X}wl#&vkX`H)bMR~?fb28O460g_!JR2-cEpy$BP(}GPz'
        'S)+pBxYMcq=c~4eIOHCul+B_#rI1b$uz%O9_%Jeg}jzh%fGHK{UiYuN6IFidAH*!cTJWF$S<(FR0UPtq^jv+AuhzyS_K+cQtu;^@S8<6u3X^'
        '9k-f#wJIh|j3-`zO-`6Lx58Y)4ZNNL&KOmhYOzbN^O%hBY=&x6&v)`dSJPHt8Y(C>Z=whvVE6`h~6%UJ0Pb&DtQ~-FyGZ2EM54@m)yMquGU@'
        'p-Y1r}(-vKt-8QH2^Re#yjJ;o9X^JDqVu;W&H-99|3$evW?*M~B4ZD+!F;fmzLtUpzd(FrtG4)2iaph)J#qrdyJ8k}p&zZi)IQmLDdxqvDQ*'
        '55&G}{=A;~vzb}Fpt8&c8iF2Guh+>v$ir*2nq_}u^0#NDtPy|4ewxVs6?%Y%K_7?0CZh@y{~p47LCIFt7jNT#mo695h6T=CI3m=J7SkN_8Rn'
        'P(oiLivsNghU=U-%)KSs>|{PWszS`#&l>m*C})%0gFz1fwvJ*^~}$B%*NkfiDT1n^LmR45)e`PPBdm{FPs7doUPPaH{UW2GE*_*lT5&`#g86'
        'aDGjdE=4}p`S*msnV(H6K^>Vq6CdyG{VGj_0Yw=b6$)D+f67As=uoQP9m)nCOOYOdZ9-$?_(zO=$K;DE*mA<z-l+aGRq%Oo9{c{x|kh=e;;s'
        'HpeROHbt+P^_~Ej2vNiI7ICEKXq%M<7n=Wq?%Z)-_LAz88@ULk=E|eIi%pt@M6s2qi-tcBtD^%C1Y0xh)aHOP#@5O=(WuQR4^4ZPtsbc-6-E'
        'QhE3Yg~f<@rx+b(pSFb@7i9ugLfS#WfD~&=AarhD%yG{5d}TIXtvvY*<^&eVpxf;;-BaE5)RI<#2d-X)&dohnMBP+qv(jz(oe8Z4!f`<DkKi'
        'MJ5IoBjaW8izmkVYzT6Tq6DkDC`ycyx^NND*$=c>2^<y`PO9hGGxoLG7>Q5RZWW)g5*9+j+I&nuEQMDFoqqO=rCYe5!Ypp8!WxPm$P%pmpJV'
        'g(mOX`ZC@(KA;QJXvAvCZsAHXuqHvlCtDT$Uo?vWE>M+5+XVud`MIRWruXokrgT)e-N-EXd)tt?=wLVI4q<Y&US_GMNdKO#d!qylUwvb}5B-'
        'ZeXCn6>$sB5Rm7xYPqrp<lF~AGk|00V&`os}+TpQYl)4KY~8Mtw9+VdyO9ClCvS-zs2{nKfXJA_4@41hci5<@{A7ueD@xbUVewUet7%p?0bC'
        'p>BalkFTVL6bG?28Uthd>`<}A?c=pW;deBa$EI(IN39Goa<P37C^Yh@?vF1qi!ha6h!17`qjPe~-u_uRU$*1rf2+aDb0kHVC9n8xQLz~6X5-'
        'c)2z_OCX9DJd8z>6<9ZDI0Z9V6U5uN`3ZW20L^8E&s+$Tq%StVXx&7wG@^xWCiNw3_)cm9j$Or3o*Mu6S#~TeEGQoSZnjM;<EaM&_LeSKj0c'
        '9=>0pZ57|FXETgt6Gsc7=K3SZiatB<Ma^SR>FWoWw6FY0G({Jq4a(*ZLFg*csN!s&TDQ~b^q3w7x!hVq;bbyXvpEb-Tvp34sMRbOc#Z%MvFp'
        '0dwbOO*DrpCdVW=ly_iK?&7fyjh(m9~tI70zbF~uDNbX33bcdy@pU<1(r-HRr;J)OZr`~F=%I39MQ+Hfq4(sX6xl-RXwx^PtZB{M!uRWJ^$f'
        '(%vQ1O}j+PJ@6ZXsOkupU$Ic)-2?uW+_Hit)be4R8;N(NqY8XA5eac_l4-`EhsRFmZ)*%zkpi7F45MTXg66C?zBFo(X4oU!qC?)%hauMFpwW'
        '1A}&ITEZnlb5a_)69bPwBb+aN>2542lRiH(b*cC9a=F)K=&LWpJgURe9BTN(#W#r!xD^vKP&e|SoL%;{1?X+kBH2T0v>`gtQj38u(7I<Q99P'
        'LgQ=8sd&P*`ntE$tOXV7#8W0$WtbroUB!2vI`(i{AK{;~8mU@(X-JE%|%^cwdJXuh#+Nl}?FaG#y5Cf-&fb(rLg0FL72<Zgk24WAnWw3RW4p'
        'Y35L|z$+X5OFY&CYxSnf0JLd5@fdgh$lIx?*&$c3af`pWGXz?r%RD&xpy3sIiHS?R=P#Aa6o&qS5|j{J6|24qR{jToXMH)ROeN5RHto;oI>g'
        'w!0+A?ryToChU0r!W21s4uUKQfY2<Fw;71@CN%<IMBRTO_2N70x1HJS%RbXQ{?SF~Xk*ch$2|N5_Vffw{-Jpph$y32$k0ji+>zk2(_JJj5fM'
        '}kS@LOQ$s0UgR*=v<9;g`b+YV_Z`%%btV!BWBcKuVad7l~l^dXYqr#(pOm6QD^A-%I>d7zPb6}U81i2=DcD)r_-w0nM%n#Sdk|SE4qlWjIxa'
        'J8VGFU=2JhAgHg<|B+VlqxA*x|*2ScrK?~SLO_-Lzl9Z{rQ>PR$@V%_hSuYbh1<<*(k4b<IEohV}d$=j-Zlpk;m?9FoC!yvqlgq4};^<l!>W'
        'l&srV&`t<ZieyqlnxmhD|9sEaL~HonTt}IUZn~!R>AF8#cpFKS?HYf9WN3<27Iqw-9=1LZ_hsmjzz;iktR$I7;_0R)~Vm@Prm@=1z76g#^>l'
        '?X8J4%5ZQ<KMfjYo5l2UN48?69TxVeEqEFc21*7ODaWi*4U`C0lR{S{Q`kICES$1Ixc1o`h@!N7E+%DjF<Z)0rVMZeIv+4f?;JSDm#yaE#w)'
        '!VblCE92Jft9l2w8pKrEIYTpU?2A)$Yu;wEkd@(JVf?=N1|{%%w~L<<y^(+g)b6k9+X6&jd@2o0~p;#)+XH@j8*ME>`ez;Iur700S|+js+Wt'
        ';XZQj~y`eqQwdN3?;U?-vHxKN?=ZL2j511+1vY#Y0+^i%Jf1l+h$je-CsUPOxb6)Z9zn`CFUv4_6xCVV`CwyW>Ivk1)vVFQ`z$=@&xwyFvbA'
        '(<ffc43^tn~*26Zh5MVoQDH8Xg%e!}M&hQgN$<>Tz=b&J0ax&dnv;e_Nal3K96!T&{6({;kHzM#;e6o9SSS@02Ig2l1tek>t_$gbqiF>m0SF'
        'J?GlApUb(j!{5leI9=O}&tg-<BX^Ui9l9N;B&RMVHkd><bi>ZE5B6E#6t2ANpL{EqS<n))32D=#!o^I|jO}+GSawq-;Qz%eS$&m=+OLXl;_a'
        '*pXidsWd2x<{P}Lq!v+?yP8C~9zw=i<+6NU|8hyUX6R8>+MSB>PMAbq<&nDrBzJ+tP44ChcC$ei)@ARxD+055iv@ag2~y^bX4i|PghTqoPbS'
        'oEY}1b?RA(38&G5zEu>B0tifko$5a1Vnrp^XK6WK@Y96ZJM0Fy9bwUTdLauBhCBNFwX0g5ft4uPkQ2M(~u?amUemutO{WzWGRP23ZTJd-}1B'
        '$q3HUl;R<i?vaSO!EGC@I8!k_c);|k$uVCDI)hP3DzX*plutXm^?*a=2yXC8-sAUG2=snjDg82<!u<ETJ;)?ht#Zaok5<B$yRV{mS_MsXg#E'
        '27WFMyldOtg-}I0g_^ckN+(Fi}64%RoMD<7vo1BPBJ~~EK=3uii8&>sx7@0|(kb2gVaaI|L%p8?oQKU3LOhsnTu6U7IlvzJi8lvH+btLV|<o'
        'gz1w%N32yU46PfwNgYoRg4^O?T5&zVIs>a{-xV`2~(@L4K7Y6cl$Uy0{8XYh2l-m?oO$HyBgKHXXSp9;0FowTlQ@fXz+?m|gVvn=rvwr#UjT'
        '+C}F5%k=rk8qnR~nM2LEpOsIztL?80=Mm95v{-CfU)9aLKWy%W(}mA4O^b;_@t?T`0>Phzjua$2XQoDdK*FqEWb8`Z1znt)85Z0?Wfqj`A*O'
        '^`&X1BK3DQgltU;UAM9~lflN#CGOJ@`WuZ%gjcohN3Ra%ma3HjWMaxvg?F(E*#ol$wZP3pz%KUS&KM1eR4CMBB67FR%~C1uFW7^qNonRvrmP'
        'F%+-_J@JW?M=*P4cSSp=a>OQS;P|BeWS;vCW-NA{Ml9R78OH=JmQ|Qib%u{@Y)QrEA*4zA=x^GKZA=AA5Lm@_dvE$qVTZ(m$afODiStpkm&;'
        '~Jm?+cfC5wM{<X{|63f6jMBIr9tb%tEV>f<tA4@t!dRMU@cXDaA{3`;SVqB#uGUEsF`MX63S%(9|tU?<ibV>wSC`o)njJ~riBVRg5RRi#}&S'
        'sO~P0|<UYT0-$VTtdVIMzrXL7Lj!*@ce}8x58}V^#IiaVbRe$}h!E>fw3>R1r>G|F;uRKPG-@-%tN6git>QdQl1qyKlrl0c#{+=+x@6O_D!{'
        'FQdhJ8F~qx3bWT35J3`+raT1>x38f;>MV1Q;~W-dk!S8c?H(1x+5SVU^Fq-okrew&D-^1lMtHq$$q`y?5%_EG-qqQ3@g`oDP1}mQ!br6Wo$X'
        'DRe@RFvNd%;cjT1R*j7h5~w^icH2>o}gGKwt0!%)~pAPN=-n~i#Xjccm(^aYq=1m#@H`dB&Q<21gYv=JV+&LN9-Y8S@1_xlEe0r9{PsF8RB&'
        ')OLBL>5BCQ=K6@iX0C{*pR3Uaarv4Hl+K7#0nq3`<Wn=+e$9bJ<o46$WCxjkLKVz#BB2h4Rgf;wN4uYM~$|_`+9!D2~>&21yFb1sdRd(f7ag'
        '98+@fY@!{qpJu-JA^>lUTb=-K5Na3I~c~gxo10m_yX*E`npFzZjf!#{O06;Y06T3bhpICSJy*1qa%}R<CRvO<}2gP28><x_!tYI&v1{!X6Vv'
        '|9}TyAe4m#`9x1!K^@Mt%d{4IfO_3G{+vL+oC`Y-|M;Te_F?TW^tixI#B!ER5fz$1c5##<Y!P>REiH?;e1ZYxize%rBYYM!rnw!jZ3sEOSQ;'
        'm7U-&H!7s%nG1im_ib%acBs(txqElJ;K#B}E+ReKZsK0BvwNd~`JQbd`Ic=V*^TmhVC1grr&~I^R|pi<Z!27?Q!n{cApCRk+$aYG`qwNzPmw'
        'u=G2?kG8PSvDa)m&r{Gi~SzO>uQV@vB{8h{Kj3dM%vz$=BIZMsC~_v|)wT=vBcx|YqaV9rw5!j&Hf(Uijb*v$s6W3xM2-~rUtc&aVQ$XlowF'
        '6g!%vE8Q_zBN(<nDn?WcGsU$c26i4)PTqN`*^TQFbgJLEGx9<al&rAx-dVyKKtPvixm6e_4(^x|LKiU`R@I>WsWR+>3hjK_La7NID7T_)eGb'
        '82W<4!3kjzN;AQOLZhDs&S8OW)@52j&RdP;ogvC726;{28y#DU*4su*ONM7MYf)?{k8TyGXw>UXzwo!1XTe^X{O)J2EGqw!&BD~{b5QUrreT'
        'D-EaP=WdfjQ$7-M#})7{oB(qnyZxHz1jY2`FvVD$;csFt|ab&TFi?kTRrqsuxPZ3szlqWtVm(6><Es%2?A?mnRNo(wk~^huuk7cWf|Fcf8jb'
        'mDAmlIMpeg(K#b9YkwCw<+GJ6Od=OtoE*E&w#DWGw2h&dNaVoz&d+{2dk=GtO%H}0K_sXw5WrC6k=$7i`BGPo4sh;z9cAT+yVFVIzaG))eYB'
        '5YKq<<q)!@1w+Uhw_kihYBDxMBy5|L5q&E`;y56|hf7`Q{3M&+{er?kvCka|uOp^VeB4dV68>FM%5t3y*{b$V&el|)HCczf7EuB`C!W}9MVr'
        'P6C#_(pEuLTct?%|6kkF#LiZH`Ef<+ksp`Nj#KVyp|mkf2ilBE8En1#qBcrI$8RCrIpLIT=qOHTodGO6@iR?6n^3f#d%NgM>*8w0xwV%QWxH'
        'K!{Qx6zp=thK8HJP!Bz<6D5)FT?me+|XW*tHc?Fv>q3c$P0vykKDYcO`fm`aa<~}&oeD)dg_nX+mM{I`~T`I+WN12yQFT=cFq)x!&DaSKtsP'
        'x82RlL!V>zyE*HX(5-?Oa@?P!<7YgdPo~3pEY(n-BQ{Lt){NX}!TOxX`(Kl+&T_ThS-;o!!R6hc%Wv?<nqp-;X5>&VuH(cekHbLN`(b;`bgo'
        'y40+f3wCU?IARe@jElkr;q0ayusyhir)0{`W3)yD?o!@>aC^Iyir}@3>MuKhGp7Y+JKbadAJzfPXWHq`uz4dC_Cx{*X%Y;!txlfSx0G&?rlo'
        '8YfAG#mB8t-M*E7QK7-8rr$0a|DO}{&>8g88+HP9O4Jw^)U;h|S?f@&J~Uof|6$>|K4m5TS?D)(Nb@-?Mbd5JGil$Yq<cZxUGR(6vyyN^~I4'
        'hHm}I}pGuSp}InBSSXSiJI_pfgXyY;w)2->aut>ylII}+_7jYi_PWdpO@^jW$U<m>Mhyhz>|Fb8e4i;0{oJm#@CfkHv{I?!QBtTN2TcBhfX%'
        'b6v|G3of%KJW#_u-uEWgVefQEME{PEjN-kYhP7ReKUT9BRrEKMK-};@h1v%R~XFXsy@>jUyWl`GTpEdaC_>i4ebOKLwvLgJ_R?J8L2TH8u6a'
    )
    template = zlib.decompress(base64.b85decode(compressed)).decode('utf-8')

    def cleaned(value):
        if isinstance(value, dict): return {str(k): cleaned(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)): return [cleaned(v) for v in value]
        if pd.isna(value): return None
        if hasattr(value, 'item'): value=value.item()
        if isinstance(value, float) and not math.isfinite(value): return None
        return value

    def data(filename, columns=None):
        path = outdir / filename
        if not path.is_file(): return []
        df1 = pd.read_csv(path, encoding='utf-8-sig')
        if columns is not None: df1 = df1[columns]
        return cleaned(df1.to_dict('records'))

    a = data('01_riesgo_severo_predicciones_prueba.csv', ['real','predicho','probabilidad_clase_positiva'])
    b = data('02_demora_24h_predicciones_prueba.csv', ['real','predicho','probabilidad_clase_positiva'])
    whole_n = int(len(df))
    train_size = int(risk_info.get('n_train', whole_n - len(a))) if isinstance(risk_info,dict) else whole_n-len(a)
    stats = {
        'population': int(original_n), 'sample': whole_n, 'train':train_size, 'test':len(a),
        'averageHours':float(df['TIEMPO_HRS'].mean()),
        'within24':float((df['TIEMPO_HRS'] <= 24).mean()),
        'severeCount':int(df['RIESGO_SEVERO'].sum())
    }
    state = {
        'version': 3,'stats':stats,
        'models':{
            'risk':{'predictions':a,'importance':data('01_riesgo_severo_importancia_variables.csv'),'comparison':data('01_riesgo_severo_comparacion_modelos.csv'),'roc':[],'matrix':[]},
            'delay':{'predictions':b,'importance':data('02_demora_24h_importancia_variables.csv'),'comparison':data('02_demora_24h_comparacion_modelos.csv'),'roc':[],'matrix':[]}
        },
        'tables':{
            'riskLevels':data('03_resultados_por_nivel_riesgo.csv'),
            'courts':data('03_resultados_por_juzgado.csv'),
            'tests':data('03_pruebas_estadisticas_complementarias.csv'),
            'priority':data('04_resumen_priorizacion.csv'),
            'consistency':data('03_indice_consistencia_decisiones.csv'),
            'indicators':data('03_indicadores_generales.csv')
        }
    }
    serialized = json.dumps(cleaned(state),ensure_ascii=False,separators=(',',':'),allow_nan=False)
    serialized=serialized.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    report = template.replace('__REPORT_DATA__', serialized)
    target = base_dir / 'T2 93 320 REPORTE MACHINE LEARNING.html'
    target.write_text(report, encoding='utf-8')
    return target

def main() -> int:
    parser = argparse.ArgumentParser(description="Machine Learning completo - Tesis Alan Rivas UCV")
    parser.add_argument("--input", type=str, default=None, help="Ruta al Excel depurado")
    parser.add_argument("--output", type=str, default=None, help="Carpeta de resultados")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=232,
        help="Tamaño muestral aprobado (por defecto: 232).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    source = Path(args.input).resolve() if args.input else find_input_xlsx(base_dir)
    outdir = Path(args.output).resolve() if args.output else base_dir / "RESULTADOS ML"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TESIS UCV - MACHINE LEARNING COMPLETO")
    print("=" * 78)
    print(f"Base: {source}")
    print(f"Resultados: {outdir}")

    full = derive(load_validate(source))
    original_n = len(full)
    if args.sample_size < 2 or args.sample_size > original_n:
        raise ValueError("--sample-size debe ser >= 2 y menor o igual que el marco disponible")
    df = optional_stratified_sample(full, args.sample_size)
    if len(df) != args.sample_size:
        raise RuntimeError(f"Muestra incorrecta: esperados {args.sample_size}, obtenidos {len(df)}")
    df.drop(columns=[c for c in ("INGRESO_HORA", "INGRESO_DIA_SEMANA", "INGRESO_MES", "INGRESO_DIA", "INGRESO_SEMANA_ANIO", "INGRESO_FIN_SEMANA", "TIEMPO_HRS", "RIESGO_SEVERO", "DEMORA_MAYOR_24H", "MEDIDA_BINARIA", "RIESGO_ORDINAL") if c in df], errors="ignore").to_csv(outdir / "00_muestra_232.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] Marco de datos validado: {original_n} expedientes únicos.")
    if args.sample_size:
        print(f"[OK] Muestra reproducible utilizada: {len(df)} expedientes.")
    else:
        print(f"[OK] Análisis con base completa: {len(df)} expedientes.")

    data_quality(df, outdir, source, original_n, args.sample_size)
    descriptives_and_consistency(df, outdir)
    print("[OK] Descriptivos, celeridad y consistencia generados.")

    # MODELO 1: Identificación de riesgo severo.
    risk_features = [
        "ORGANO JURISDICCIONAL",
        "INGRESO_HORA",
        "INGRESO_DIA_SEMANA",
        "INGRESO_MES",
        "INGRESO_DIA",
        "INGRESO_SEMANA_ANIO",
        "INGRESO_FIN_SEMANA",
    ]
    risk_info = evaluate_binary_task(
        df=df,
        target="RIESGO_SEVERO",
        features=risk_features,
        categorical=["ORGANO JURISDICCIONAL"],
        task_slug="01_riesgo_severo",
        positive_label="Severo",
        negative_label="No severo",
        outdir=outdir,
    )
    print("[OK] Modelo Random Forest de riesgo severo generado.")

    # MODELO 2: Demora >24 h, vinculada a celeridad y sobrecarga.
    delay_features = [
        "ORGANO JURISDICCIONAL",
        "NIVEL DE RIESGO",
        "x_juez",
        "x_especialista",
        "INGRESO_HORA",
        "INGRESO_DIA_SEMANA",
        "INGRESO_MES",
        "INGRESO_DIA",
        "INGRESO_SEMANA_ANIO",
        "INGRESO_FIN_SEMANA",
    ]
    delay_info = evaluate_binary_task(
        df=df,
        target="DEMORA_MAYOR_24H",
        features=delay_features,
        categorical=["ORGANO JURISDICCIONAL", "NIVEL DE RIESGO", "x_juez", "x_especialista"],
        task_slug="02_demora_24h",
        positive_label=">24 h",
        negative_label="≤24 h",
        outdir=outdir,
    )
    print("[OK] Modelo Random Forest de demora >24 h generado.")

    create_priority(df, risk_info, delay_info, outdir)
    print("[OK] Priorización automatizada generada.")

    write_summary(source, original_n, df, args.sample_size, risk_info, delay_info, outdir)
    print("[OK] Resumen final generado.")

    html_report = create_html_report(outdir, source, original_n, df, risk_info, delay_info, base_dir)
    print(f"[OK] Reporte HTML independiente generado: {html_report}")

    # Solo al terminar, mostrar archivos de resultados con nombres legibles.
    # El HTML anterior ya ha integrado los resultados necesarios.
    for result_file in outdir.iterdir():
        if result_file.is_file():
            clean_name = result_file.stem.replace("_", " ").replace("-", " ") + result_file.suffix
            if result_file.name != clean_name:
                result_file.rename(result_file.with_name(clean_name))

    print("=" * 78)
    print("PROCESO FINALIZADO CORRECTAMENTE")
    print(f"Archivos: {outdir}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nERROR DURANTE LA EJECUCIÓN:")
        print(str(exc))
        raise
