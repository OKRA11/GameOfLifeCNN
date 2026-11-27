import streamlit as st
import numpy as np

st.title("Заголовок страницы")
st.header("Заголовок раздела")
st.subheader("Подзаголовок")
st.write('Обычный текст')
st.markdown("**Жирный текст**, *курсив*, списки")

name = st.text_input("Введите имя")
age = st.slider("Введите возраст", min_value=0, max_value=100, value=18)
is_ready = st.checkbox("Готово?")
choice = st.selectbox("Выберите", ["Опция 1", "Опция 2", "Опция 3"])
clicked = st.button("Нажми меня")

if clicked:
    st.write(f"Привет, {name}! Тебе {age} лет, {is_ready} и ты выбрал {choice}")

col1, col2, col3 = st.columns([1, 2, 2])
with col1:
    st.write("Это в левой колонке")

with col2:
    st.write("Это в правой колонке")

st.sidebar.title("Настройки")
value = st.sidebar.slider("Параметр", 0, 10)

tab1, tab2 = st.tabs(["Первая", "Вторая"])
with tab1:
    st.write("Контент вкладки 1")

GRID_SIZE = 5
for row in range(GRID_SIZE):
    cols = st.columns(GRID_SIZE)

    for col_index, col in enumerate(cols):
        col.checkbox("", key=f"cell_{row}_{col_index}")

if "grid" not in st.session_state:
    st.session_state.grid = np.zeros((5, 5), dtype=int)

if st.button("Случайное поле"):
    st.session_state.grid = np.random.randint(0, 2, (5, 5))

st.write(st.session_state.grid)

@st.cache_resource

def load_model():
    model = ...
    return model

@st.cache_data

def load_dataset():
    return

array = np.random.rand(50, 50)
st.image(array, width=300, clamp=True)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(array)
st.pyplot(fig)

st.metric(label="Точность", value="94,2%")