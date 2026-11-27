import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch


# Файл страницы находится в Olya/streamlit_app/pages.
# parents[2] — это папка Olya, где лежат model.py и best_model.pth.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import GameOfLifeCNN


MODEL_PATH = PROJECT_ROOT / "best_model.pth"
GRID_SIZE = 10

FIGURES = {
    "Блок": np.array(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=int,
    ),
    "Мигалка": np.array(
        [
            [1, 1, 1],
        ],
        dtype=int,
    ),
    "Глайдер": np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=int,
    ),
}


def apply_conway_rules(grid: np.ndarray) -> np.ndarray:
    """Вычисляет следующее поколение на замкнутом поле."""
    neighbors = sum(
        np.roll(np.roll(grid, i, axis=0), j, axis=1)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        if (i, j) != (0, 0)
    )

    return (
        (neighbors == 3)
        | ((grid == 1) & (neighbors == 2))
    ).astype(int)


@st.cache_resource
def load_model() -> GameOfLifeCNN:
    """Загружает CNN один раз и сохраняет её между перезапусками Streamlit."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл с весами модели: {MODEL_PATH}"
        )

    model = GameOfLifeCNN()

    try:
        weights = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Для старых версий PyTorch, где ещё нет weights_only.
        weights = torch.load(
            MODEL_PATH,
            map_location="cpu",
        )

    model.load_state_dict(weights)
    model.eval()

    return model


def predict_cnn(
    model: GameOfLifeCNN,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает бинарный прогноз CNN и вероятности живых клеток."""
    input_tensor = (
        torch.from_numpy(grid)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    with torch.no_grad():
        probabilities = model(input_tensor)

    probabilities_array = probabilities[0, 0].cpu().numpy()
    prediction = (probabilities_array >= 0.5).astype(int)

    return prediction, probabilities_array


def clear_comparison() -> None:
    """Удаляет результат сравнения после изменения исходного поля."""
    st.session_state.pop("comparison", None)


def toggle_cell_callback(row: int, col: int) -> None:
    """Меняет состояние клетки после клика по игровому полю."""
    st.session_state.grid[row, col] = 1 - st.session_state.grid[row, col]
    clear_comparison()


def step_callback() -> None:
    """Применяет к полю один классический шаг Игры Жизнь."""
    st.session_state.grid = apply_conway_rules(
        st.session_state.grid
    )
    st.session_state.generation += 1
    clear_comparison()


def clear_callback() -> None:
    """Очищает игровое поле."""
    st.session_state.grid = np.zeros(
        (GRID_SIZE, GRID_SIZE),
        dtype=int,
    )
    st.session_state.generation = 0
    clear_comparison()


def random_callback() -> None:
    """Создаёт случайное игровое поле."""
    st.session_state.grid = np.random.randint(
        0,
        2,
        size=(GRID_SIZE, GRID_SIZE),
    )
    st.session_state.generation = 0
    clear_comparison()


def load_figure_callback(figure_name: str) -> None:
    """Помещает выбранную фигуру в центр пустого поля."""
    figure = FIGURES[figure_name]
    height, width = figure.shape
    top = (GRID_SIZE - height) // 2
    left = (GRID_SIZE - width) // 2

    grid = np.zeros(
        (GRID_SIZE, GRID_SIZE),
        dtype=int,
    )
    grid[top:top + height, left:left + width] = figure

    st.session_state.grid = grid
    st.session_state.generation = 0
    clear_comparison()


def life_grid_rgb(grid: np.ndarray) -> np.ndarray:
    """Преобразует поле 0/1 в хорошо различимое цветное изображение."""
    dead = np.array([245, 247, 250], dtype=np.uint8)
    alive = np.array([35, 45, 65], dtype=np.uint8)
    return np.where(grid[..., None] == 1, alive, dead)


def error_grid_rgb(error_map: np.ndarray) -> np.ndarray:
    """Показывает ошибки красным, а совпадения светло-серым."""
    correct = np.array([235, 239, 244], dtype=np.uint8)
    error = np.array([230, 70, 70], dtype=np.uint8)
    return np.where(error_map[..., None] == 1, error, correct)


def probability_grid_rgb(probabilities: np.ndarray) -> np.ndarray:
    """Строит тепловую карту вероятности живой клетки."""
    values = np.clip(probabilities, 0.0, 1.0)[..., None]
    low = np.array([245, 247, 250], dtype=np.float32)
    high = np.array([47, 109, 179], dtype=np.float32)
    return (low + values * (high - low)).astype(np.uint8)


def compare_callback() -> None:
    """Сравнивает прогноз CNN с классическим алгоритмом."""
    input_grid = st.session_state.grid.copy()
    true_next = apply_conway_rules(input_grid)
    cnn_next, probabilities = predict_cnn(
        load_model(),
        input_grid,
    )

    error_map = (cnn_next != true_next).astype(int)
    error_count = int(error_map.sum())
    total_cells = int(error_map.size)
    accuracy = 1 - error_count / total_cells

    st.session_state.comparison = {
        "input_grid": input_grid,
        "true_next": true_next,
        "cnn_next": cnn_next,
        "probabilities": probabilities,
        "error_map": error_map,
        "error_count": error_count,
        "total_cells": total_cells,
        "accuracy": accuracy,
    }


st.set_page_config(
    page_title="Game of Life",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Игра «Жизнь»: CNN и классические правила")
st.caption(
    "Создайте поле, затем сравните прогноз обученной CNN "
    "с точным математическим алгоритмом."
)

# У Streamlit нет отдельного квадратного виджета-клетки. Класс st-key-life_grid
# создаётся параметром key у контейнера ниже, поэтому правило не затрагивает
# кнопки управления и готовых фигур.
st.html(
    """
    <style>
    .st-key-life_grid div[data-testid="stButton"] button {
        aspect-ratio: 1 / 1;
        min-height: 0;
        padding: 0;
        font-size: clamp(0.8rem, 1.6vw, 1.25rem);
    }
    </style>
    """
)


if (
    "grid" not in st.session_state
    or st.session_state.grid.shape != (GRID_SIZE, GRID_SIZE)
):
    st.session_state.grid = np.zeros(
        (GRID_SIZE, GRID_SIZE),
        dtype=int,
    )
    st.session_state.generation = 0
    clear_comparison()

if "generation" not in st.session_state:
    st.session_state.generation = 0


# Модель загружается здесь один раз. Последующие перезапуски используют кэш.
model = load_model()


st.subheader("Управление")

button_step, button_clear, button_random, button_compare = st.columns(4)

with button_step:
    st.button(
        "Следующий шаг",
        on_click=step_callback,
        width="stretch",
    )

with button_clear:
    st.button(
        "Очистить поле",
        on_click=clear_callback,
        width="stretch",
    )

with button_random:
    st.button(
        "Случайное поле",
        on_click=random_callback,
        width="stretch",
    )

with button_compare:
    st.button(
        "Сравнить CNN и правила",
        type="primary",
        on_click=compare_callback,
        width="stretch",
    )


st.markdown("**Готовые фигуры**")

figure_block, figure_blinker, figure_glider = st.columns(3)

with figure_block:
    st.button(
        "Блок",
        on_click=load_figure_callback,
        args=("Блок",),
        width="stretch",
    )

with figure_blinker:
    st.button(
        "Мигалка",
        on_click=load_figure_callback,
        args=("Мигалка",),
        width="stretch",
    )

with figure_glider:
    st.button(
        "Глайдер",
        on_click=load_figure_callback,
        args=("Глайдер",),
        width="stretch",
    )


field_column, result_column = st.columns([1, 1.35])

with field_column:
    st.subheader("Текущее поле")
    st.caption("Нажимайте на клетки: ● — живая, · — мёртвая.")

    with st.container(key="life_grid"):
        for row in range(GRID_SIZE):
            columns = st.columns(GRID_SIZE, gap="small")

            for col, column in enumerate(columns):
                with column:
                    is_alive = bool(st.session_state.grid[row, col])
                    st.button(
                        "●" if is_alive else "·",
                        key=f"life_cell_{row}_{col}",
                        on_click=toggle_cell_callback,
                        args=(row, col),
                        type="primary" if is_alive else "secondary",
                        help=f"Клетка {row + 1}, {col + 1}",
                        width="stretch",
                    )

    metric_live, metric_generation = st.columns(2)

    with metric_live:
        st.metric(
            "Живых клеток",
            int(st.session_state.grid.sum()),
        )

    with metric_generation:
        st.metric(
            "Поколение",
            st.session_state.generation,
        )


with result_column:
    st.subheader("Прогноз следующего поколения")

    comparison = st.session_state.get("comparison")

    if comparison is None:
        st.info(
            "Нажмите «Сравнить CNN и правила», чтобы получить "
            "два прогноза и увидеть ошибки модели."
        )
    else:
        metric_accuracy, metric_errors = st.columns(2)

        with metric_accuracy:
            st.metric(
                "Точность CNN",
                f"{comparison['accuracy']:.1%}",
            )

        with metric_errors:
            st.metric(
                "Ошибочных клеток",
                (
                    f"{comparison['error_count']} "
                    f"из {comparison['total_cells']}"
                ),
            )

        cnn_column, rules_column, error_column = st.columns(3)

        with cnn_column:
            st.caption("Прогноз CNN")
            st.image(
                life_grid_rgb(comparison["cnn_next"]),
                width="stretch",
            )

        with rules_column:
            st.caption("Правила Конвея")
            st.image(
                life_grid_rgb(comparison["true_next"]),
                width="stretch",
            )

        with error_column:
            st.caption("Карта ошибок")
            st.image(
                error_grid_rgb(comparison["error_map"]),
                width="stretch",
            )
            st.caption("Красная клетка — ошибка CNN")

        with st.expander("Вероятности, предсказанные CNN"):
            st.image(
                probability_grid_rgb(comparison["probabilities"]),
                width="stretch",
            )
            st.caption(
                "Чем светлее клетка, тем выше уверенность модели, "
                "что она будет живой."
            )
