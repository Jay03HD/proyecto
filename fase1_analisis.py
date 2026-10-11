"""
Fase 1 - Dataset: Social Media Impact on Life (estudiantes)
Uso: python fase1_analisis.py Social_media_impact_on_life.csv
Genera: docs/data.json (dashboard) y reporte_fase1.txt (para el reporte)
"""
import sys, json, os
import pandas as pd

ruta = sys.argv[1] if len(sys.argv) > 1 else "Social_media_impact_on_life.csv"
out = []
def log(t=""):
    print(t); out.append(str(t))

df = pd.read_csv(ruta)

# ---------- 2. ESTRUCTURA
log("=== 2. ESTRUCTURA ===")
log(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
log(df.dtypes.to_string())
log("Periodo: el dataset NO contiene fechas (es una encuesta transversal, un registro por estudiante).")
for c in ["Gender", "Academic_Level", "Primary_Platform", "Device_Type",
          "Social_Comparison_Frequency", "Overall_Impact"]:
    log(f"{c}: {df[c].value_counts().to_dict()}")

# ---------- 3. CALIDAD
log("\n=== 3. CALIDAD ===")
log("Valores vacios:\n" + df.isna().sum()[df.isna().sum() > 0].to_string())
log(f"Duplicados exactos: {df.duplicated().sum()} | Student_ID repetidos: {df['Student_ID'].duplicated().sum()}")
log("Fechas: no aplica (sin columnas de fecha).")
log("Atipicos (IQR 1.5):")
for c in ["Daily_Usage_Hours", "Weekend_Extra_Hours", "Sleep_Duration_Hours",
          "Perceived_Stress_Score", "Mental_Health_Index", "Academic_Performance_GPA"]:
    q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
    n = ((df[c] < q1 - 1.5 * i) | (df[c] > q3 + 1.5 * i)).sum()
    log(f"  {c}: {n} ({n/len(df):.1%}) | min={df[c].min()} max={df[c].max()}")
log("Coherencia Age vs Academic_Level:")
log(df.groupby("Academic_Level")["Age"].agg(["min", "max"]).to_string())
log("DECISIONES: (a) vacios en Stress y GPA: NO imputar; se excluyen solo de los promedios "
    "de esas variables para no inventar datos. (b) atipicos: conservar, son valores posibles "
    "(14 h de uso, 3 h de sueno) y representan a los casos de mayor riesgo. "
    "(c) sin duplicados: no se elimina nada.")

# ---------- 4. PROCESAMIENTO
log("\n=== 4. PROCESAMIENTO ===")
df["Late_Night_Usage"] = df["Late_Night_Usage"].astype(int)
bins = [0, 2, 4, 6, 8, 10, 20]
labels = ["0-2 h", "2-4 h", "4-6 h", "6-8 h", "8-10 h", "10+ h"]
df["Usage_Range"] = pd.cut(df["Daily_Usage_Hours"], bins, labels=labels)
kpis = {
    "estudiantes": len(df),
    "uso_diario_prom": round(df["Daily_Usage_Hours"].mean(), 2),
    "salud_mental_prom": round(df["Mental_Health_Index"].mean(), 2),
    "sueno_prom": round(df["Sleep_Duration_Hours"].mean(), 2),
    "pct_impacto_negativo": round((df["Overall_Impact"] == "Negative").mean() * 100, 2),
}
log(f"KPIs globales (para validar el dashboard): {kpis}")
tabla = df.groupby("Usage_Range", observed=True).agg(
    n=("Student_ID", "count"), salud_mental=("Mental_Health_Index", "mean"),
    sueno=("Sleep_Duration_Hours", "mean"), estres=("Perceived_Stress_Score", "mean"),
    gpa=("Academic_Performance_GPA", "mean")).round(2)
log("Promedios por rango de uso diario:\n" + tabla.to_string())

cols = {"Age": "Age", "Gender": "Gender", "Academic_Level": "Level",
        "Primary_Platform": "Platform", "Daily_Usage_Hours": "Usage",
        "Sleep_Duration_Hours": "Sleep", "Late_Night_Usage": "Late",
        "Perceived_Stress_Score": "Stress", "Mental_Health_Index": "MH",
        "Academic_Performance_GPA": "GPA", "Overall_Impact": "Impact"}
rec = df[list(cols)].rename(columns=cols)
os.makedirs("docs", exist_ok=True)
with open("docs/data.json", "w", encoding="utf-8") as f:
    json.dump({"kpis": kpis, "registros": json.loads(rec.to_json(orient="records"))}, f)
log("Archivo generado: docs/data.json")

# ---------- 5. EVIDENCIA PARA HALLAZGOS
log("\n=== 5. EVIDENCIA ===")
num = df[["Daily_Usage_Hours", "Sleep_Duration_Hours", "Perceived_Stress_Score",
          "Mental_Health_Index", "Academic_Performance_GPA"]].corr().round(2)
log("H1/H2 Correlaciones:\n" + num.to_string())
log("H3 Late_Night_Usage:\n" + df.groupby("Late_Night_Usage")[["Sleep_Duration_Hours", "Mental_Health_Index"]].mean().round(2).to_string())
log("H4 Plataforma:\n" + df.groupby("Primary_Platform")[["Daily_Usage_Hours", "Mental_Health_Index", "Academic_Performance_GPA"]].mean().round(2).to_string())
log("H5 Impacto declarado:\n" + df.groupby("Overall_Impact")[["Daily_Usage_Hours", "Mental_Health_Index"]].mean().round(2).to_string())
log("H6 Comparacion social:\n" + df.groupby("Social_Comparison_Frequency")[["Perceived_Stress_Score", "Mental_Health_Index"]].mean().round(2).to_string())
log(f"H7 Estudiantes con 8+ h/dia: {(df['Daily_Usage_Hours']>=8).mean():.1%}")
log("NOTA: son correlaciones, no causalidad. El dataset parece sintetico/de encuesta.")

open("reporte_fase1.txt", "w", encoding="utf-8").write("\n".join(out))
