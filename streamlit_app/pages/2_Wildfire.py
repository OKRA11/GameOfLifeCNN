"""Страница прогноза пожара: CNN, U-Net и U-Net++ из финального ноутбука."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colormaps
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wildfire_model import (  # noqa: E402
    ALL_FEATURES, DATA_STATS, FEATURE_LABELS, SAMPLE_SIZE,
    load_wildfire_checkpoint, predict_wildfire,
)

# Сюда распаковывается streamlit_export.zip из ноутбука
ARTIFACTS = PROJECT_ROOT / "wildfire_final"
MODELS_DIR = ARTIFACTS / "models"
SAMPLES_PATH = ARTIFACTS / "demo_samples.npz"
FIGURES = ARTIFACTS / "figures"

MODEL_NAMES = {"unetpp": "U-Net++", "unet": "U-Net", "cnn": "CNN"}
GROUP_NAMES = {
    "best": "Удачный прогноз",
    "gain": "Модель лучше «копии вчера»",
    "random": "Случайный пример",
}
PREV = ALL_FEATURES.index("PrevFireMask")
ERROR_COLORS = {
    "tn": (255, 255, 255), "tp": (236, 70, 71), "fp": (242, 166, 0),
    "fn": (37, 125, 224), "unknown": (208, 208, 208),
}

st.set_page_config(page_title="Прогноз пожара", page_icon="🔥", layout="wide")
st.title("🔥 Прогноз распространения лесного пожара на следующий день")
st.caption("Датасет Next Day Wildfire Spread. Модели обучены на всех 14 979 примерах train; "
           "вход — карта 64×64 км, прогноз — центральные 32×32 км.")


def file_version(path):
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_resource
def load_models(version):
    models = {}
    for key in MODEL_NAMES:
        path = MODELS_DIR / f"{key}.pth"
        if path.exists():
            models[key] = load_wildfire_checkpoint(path)
    return models


@st.cache_data
def load_samples(version):
    with np.load(SAMPLES_PATH) as saved:
        data = {name: saved[name] for name in saved.files}
    if "group" not in data:
        data["group"] = np.array(["random"] * len(data["x"]))
    return data


def center(values):
    top = (values.shape[-2] - SAMPLE_SIZE) // 2
    left = (values.shape[-1] - SAMPLE_SIZE) // 2
    return values[..., top:top + SAMPLE_SIZE, left:left + SAMPLE_SIZE]


def metrics(prediction, target, valid):
    pred = prediction & valid
    true = target & valid
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {"IoU": iou, "F1": f1, "Precision": precision, "Recall": recall}


def error_rgb(prediction, target, valid):
    image = np.empty((*prediction.shape, 3), np.uint8)
    image[:] = ERROR_COLORS["tn"]
    image[prediction & target] = ERROR_COLORS["tp"]
    image[prediction & ~target] = ERROR_COLORS["fp"]
    image[~prediction & target] = ERROR_COLORS["fn"]
    image[~valid] = ERROR_COLORS["unknown"]
    return image


def physical_values(name, normalized):
    """Отменяет нормализацию, чтобы показать признак в настоящих единицах."""
    if name == "PrevFireMask":
        return normalized
    minimum, maximum, mean, std = DATA_STATS[name]
    values = np.clip(mean + (normalized * 6 - 3) * std, minimum, maximum)
    return values - 273.15 if name in ("tmmn", "tmmx") else values


FEATURE_UNITS = {
    "elevation": "м", "th": "°", "vs": "м/с", "tmmn": "°C", "tmmx": "°C",
    "sph": "кг/кг", "pr": "мм", "pdsi": "", "NDVI": "", "population": "чел./км²",
    "erc": "", "PrevFireMask": "0/1",
}
FEATURE_CMAPS = {"elevation": "terrain", "th": "twilight", "NDVI": "YlGn", "erc": "YlOrRd",
                 "tmmn": "coolwarm", "tmmx": "coolwarm", "pr": "Blues", "sph": "Blues",
                 "population": "Purples", "PrevFireMask": "Reds"}

# ---------- Загрузка ----------
if not MODELS_DIR.exists() or not SAMPLES_PATH.exists():
    st.error(f"Не найдены результаты ноутбука в папке `{ARTIFACTS}`.")
    st.markdown("Распакуйте `streamlit_export.zip` из Colab так, чтобы получилось:\n"
                "```\nOlya/wildfire_final/models/cnn.pth\n"
                "Olya/wildfire_final/models/unet.pth\nOlya/wildfire_final/models/unetpp.pth\n"
                "Olya/wildfire_final/demo_samples.npz\nOlya/wildfire_final/figures/...\n```")
    st.stop()

models = load_models(file_version(MODELS_DIR))
samples = load_samples(file_version(SAMPLES_PATH))
if not models:
    st.error(f"В `{MODELS_DIR}` нет файлов cnn.pth / unet.pth / unetpp.pth.")
    st.stop()

tab_demo, tab_results = st.tabs(["Прогноз на примере", "Итоговые результаты на test"])

# ---------- Вкладка 1: прогноз ----------
with tab_demo:
    order = sorted(range(len(samples["x"])),
                   key=lambda i: (list(GROUP_NAMES).index(str(samples["group"][i]))
                                  if str(samples["group"][i]) in GROUP_NAMES else 9,
                                  -float(center(samples["y"][i, 0]).sum())))
    labels = {
        f"#{int(samples['ids'][i])} · {GROUP_NAMES.get(str(samples['group'][i]), 'Пример')} · "
        f"огонь завтра: {int((center(samples['y'][i, 0]) * center(samples['valid'][i, 0])).sum())} клеток": i
        for i in order
    }

    left, right = st.columns([2, 1])
    with left:
        index = labels[st.selectbox("Пример из test", list(labels))]
    with right:
        model_key = st.selectbox("Модель", list(models), format_func=MODEL_NAMES.get)
    model, meta = models[model_key]
    default_threshold = float(meta.get("threshold", 0.5))

    with st.form("wildfire_form"):
        threshold = st.slider("Порог вероятности", 0.05, 0.95, default_threshold, 0.01,
                              help=f"Порог, выбранный на validation: {default_threshold:.2f}")
        active = st.multiselect(
            "Активные признаки (эксперимент: что будет, если «выключить» признак)",
            list(ALL_FEATURES), default=list(ALL_FEATURES),
            format_func=lambda name: FEATURE_LABELS[name],
        )
        st.form_submit_button("Получить прогноз", type="primary")

    x = samples["x"][index].copy()
    for c, name in enumerate(ALL_FEATURES):
        if name not in active:
            x[c] = 0.0 if name == "PrevFireMask" else 0.5   # 0.5 = среднее значение признака
    target = center(samples["y"][index, 0]) >= 0.5
    valid = center(samples["valid"][index, 0]) >= 0.5
    previous = center(x[PREV]) >= 0.5

    probabilities = predict_wildfire(model, x, meta["channel_indices"])
    prediction = probabilities >= threshold
    m = metrics(prediction, target, valid)
    m_persistence = metrics(previous, target, valid)

    columns = st.columns(5)
    for column, (name, value) in zip(columns, m.items()):
        column.metric(name, f"{value:.3f}")
    columns[4].metric("F1 «копии вчера»", f"{m_persistence['F1']:.3f}",
                      delta=f"{m['F1'] - m_persistence['F1']:+.3f} у модели")

    fig = Figure(figsize=(16, 4.4), layout="constrained")
    axes = fig.subplots(1, 4)
    reds = colormaps["Reds"].copy()
    reds.set_bad("#d0d0d0")
    axes[0].imshow(previous, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Пожар сегодня (вход)")
    axes[1].imshow(np.ma.array(target, mask=~valid), cmap=reds, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Факт: пожар завтра")
    heat = axes[2].imshow(probabilities, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"{MODEL_NAMES[model_key]}: вероятность")
    fig.colorbar(heat, ax=axes[2], fraction=.046, pad=.02)
    axes[3].imshow(error_rgb(prediction, target, valid), interpolation="nearest")
    axes[3].set_title(f"Ошибки при пороге {threshold:.2f}")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    st.pyplot(fig)
    st.caption("Карта ошибок: 🟥 верно предсказан огонь · 🟨 ложная тревога · "
               "🟦 пропущенный огонь · ⬜ серый — нет данных (не учитывается в метриках).")

    with st.expander("Сравнить все модели на этом примере"):
        rows = []
        for key, (other, other_meta) in models.items():
            p = predict_wildfire(other, x, other_meta["channel_indices"])
            row = metrics(p >= float(other_meta["threshold"]), target, valid)
            rows.append({"Модель": MODEL_NAMES[key], "Порог": float(other_meta["threshold"]), **row})
        rows.append({"Модель": "Копия вчерашнего пожара", "Порог": 0.5, **m_persistence})
        st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True)

    with st.expander("Входные признаки (карта 64×64 км)"):
        name = st.selectbox("Признак", list(ALL_FEATURES), format_func=lambda n: FEATURE_LABELS[n])
        values = physical_values(name, samples["x"][index, ALL_FEATURES.index(name)])
        fig = Figure(figsize=(5, 4.2), layout="constrained")
        ax = fig.subplots()
        image = ax.imshow(values, cmap=FEATURE_CMAPS.get(name, "viridis"), interpolation="nearest")
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
            (15.5, 15.5), 32, 32, fill=False, edgecolor="black", linestyle="--", linewidth=1))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(FEATURE_LABELS[name] + " (пунктир — зона прогноза)")
        fig.colorbar(image, ax=ax, label=FEATURE_UNITS[name])
        st.pyplot(fig, width="content")

# ---------- Вкладка 2: итоговые метрики ----------
with tab_results:
    table_path = ARTIFACTS / "test_comparison.csv"
    if table_path.exists():
        st.subheader("Метрики на всём test (1 689 примеров)")
        table = pd.read_csv(table_path)
        st.dataframe(table.style.format(precision=4, subset=[c for c in ["IoU", "F1", "Precision", "Recall", "AP"] if c in table]),
                     hide_index=True)
        st.caption("Эпоха выбрана по validation AP, порог — по validation F1; test использован один раз. "
                   "В скобках — 95% бутстреп-интервал разницы F1 с «копией вчерашнего пожара».")
    for title, filename in [("Сравнение метрик", "02_test_metrics.png"),
                            ("Precision–Recall кривые", "03_pr_curves.png"),
                            ("Кривые обучения", "01_learning_curves.png")]:
        if (FIGURES / filename).exists():
            st.subheader(title)
            st.image(str(FIGURES / filename))
    ablation_path = ARTIFACTS / "ablation.csv"
    if ablation_path.exists():
        st.subheader("Вклад признаков (U-Net)")
        st.dataframe(pd.read_csv(ablation_path).style.format(precision=4), hide_index=True)
        if (FIGURES / "08_feature_ablation.png").exists():
            st.image(str(FIGURES / "08_feature_ablation.png"))
    duplicate_path = ARTIFACTS / "duplicate_check.csv"
    if duplicate_path.exists():
        st.subheader("Проверка на запоминание мест")
        st.dataframe(pd.read_csv(duplicate_path).style.format(precision=4), hide_index=True)
        st.caption("Отдельно посчитаны примеры test, для которых в train есть почти та же местность. "
                   "Преимущество U-Net++ сохраняется на примерах без копии — модель прогнозирует, а не вспоминает.")
