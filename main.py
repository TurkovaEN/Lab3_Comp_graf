# Подключение модулей.
import argparse  # Параметры запуска.
import math  # Тригонометрия.
import time  # Время кадра.
from concurrent.futures import ThreadPoolExecutor  # Подготовка без блокировки окна.
from pathlib import Path  # Пути файлов.
import tkinter as tk  # Оконный интерфейс.
from tkinter import ttk  # Элементы управления.
import numpy as np  # Операции с массивами.
from PIL import Image, ImageDraw, ImageTk  # Изображения.
from fast_render import render_surface, draw_edges  # Ускоренный конвейер.

# Основные параметры.
NEAR = 0.12  # Ближняя плоскость.
FOV = math.radians(55)  # Вертикальный угол обзора.
LIGHT = np.array([3.5, 5.0, 4.0])  # Точечный источник.
MODES = ('Плоская', 'Гуро', 'Фонг')  # Режимы закраски.

# Нормирование векторов.
def unit(v):  # Единичное направление.
    length = np.linalg.norm(v, axis=-1, keepdims=True)  # Длина вектора.
    return v / np.maximum(length, 1e-12)  # Защита от нуля.

# Образующая кривая.
def profile(t):  # Радиус и высота.
    r = 0.82 + 0.29 * np.cos(2 * np.pi * t) + 0.14 * np.sin(3 * np.pi * t)  # Радиус.
    return r, 3.4 * t - 1.7  # Диапазон высот.

# Построение поверхности вращения.
def build_mesh(rows=24, sectors=40):  # Размер сетки.
    vertices = []  # Мировые координаты.
    faces = []  # Индексы треугольников.
    for i in range(rows + 1):  # Точки образующей.
        r, y = profile(i / rows)  # Текущие параметры.
        for j in range(sectors):  # Угловые отсчёты.
            angle = 2 * np.pi * j / sectors  # Угол вращения.
            vertices.append((r * np.cos(angle), y, r * np.sin(angle)))  # Новая вершина.
    for i in range(rows):  # Полосы поверхности.
        for j in range(sectors):  # Ячейки полосы.
            a = i * sectors + j  # Нижняя вершина.
            b = i * sectors + (j + 1) % sectors  # Замыкание шва.
            c = a + sectors  # Верхняя вершина.
            d = b + sectors  # Верхний сосед.
            faces.extend(((a, c, b), (b, c, d)))  # Два треугольника.
    p = np.asarray(vertices, dtype=float)  # Массив координат.
    f = np.asarray(faces, dtype=int)  # Массив индексов.
    normals = np.zeros_like(p)  # Нормали вершин.
    cross = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])  # Нормали граней.
    for corner in range(3):  # Каждая вершина в треугольнике
        np.add.at(normals, f[:, corner], cross)  # Прибавить нормаль грани
    return p, f, unit(normals), unit(cross)  # Готовая сетка.

# Расчёт интенсивности.
def lighting(p, n, eye, specular=True):  # Модель отражения.
    n = unit(n)  # Нормирование нормали.
    l = unit(LIGHT - p)  # Направление к свету.
    v = unit(eye - p)  # Направление к камере.
    cosine = np.maximum(np.sum(n * l, axis=-1), 0)  # Закон Ламберта.
    reflection = 2 * cosine[..., None] * n - l  # Отражённый луч.
    shine = np.maximum(np.sum(reflection * v, axis=-1), 0) ** 32  # Зеркальный блик.
    shine = np.where(cosine > 0, shine, 0)  # Светлая сторона.
    intensity = 0.16 + 0.68 * cosine + (0.28 * shine if specular else 0)  # Сумма компонент.
    return np.clip(intensity, 0, 1)  # Допустимая яркость.

# Закреплённая камера.
class Camera:  # Положение и ориентация.
    # Начальные настройки.
    def __init__(self):  # Начальные настройки.
        self.reset()  # Исходный ракурс.

    # Возврат к объекту.
    def reset(self):  # Возврат к объекту.
        self.eye = np.array([4.5, 2.6, 6.5])  # Положение камеры.
        direction = unit(-self.eye)  # Взгляд к началу.
        self.yaw = math.atan2(direction[0], direction[2])  # Азимут.
        self.pitch = math.asin(direction[1])  # Наклон.

    # Ортонормальный базис.
    def basis(self):  # Ортонормальный базис.
        cp = math.cos(self.pitch)  # Косинус наклона.
        forward = np.array([math.sin(self.yaw) * cp, math.sin(self.pitch), math.cos(self.yaw) * cp])  # Вперёд.
        right = unit(np.cross(forward, [0, 1, 0]))  # Вправо.
        up = np.cross(right, forward)  # Вверх.
        return np.array([right, up, forward])  # Строки базиса.

# Матрица поворота модели.
def model_rotation(angles):  # Повороты вокруг трёх осей.
    x, y, z = angles  # Углы в радианах.
    cx, sx = math.cos(x), math.sin(x)  # Поворот вокруг X.
    cy, sy = math.cos(y), math.sin(y)  # Поворот вокруг Y.
    cz, sz = math.cos(z), math.sin(z)  # Поворот вокруг Z.
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])  # Матрица X.
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])  # Матрица Y.
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])  # Матрица Z.
    return rz @ ry @ rx  # Общий поворот.

# Преобразования модели.
class Model:  # Вращение и масштаб.
    # Исходное положение.
    def __init__(self):  # Настройки объекта.
        self.reset()  # Начальные преобразования.

    # Сброс преобразований.
    def reset(self):  # Возврат модели.
        self.angles = np.zeros(3)  # Углы поворота.
        self.scale = 1.0  # Исходный масштаб.

    # Поворот мышью.
    def rotate(self, dx, dy):  # Изменение ориентации.
        self.angles += np.array([dy, dx, 0]) * 0.008  # Наклон и разворот.
        self.angles %= 2 * np.pi  # Ограничение накопления.

    # Масштабирование модели.
    def zoom(self, steps):  # Изменение размера.
        self.scale = float(np.clip(self.scale * 1.1 ** np.clip(steps, -20, 20), 0.25, 2.5))  # Допустимый диапазон.

    # Непрерывное управление.
    def update(self, keys, dt):  # Клавиши поворота.
        x = int(bool(keys & {'s', 'down'})) - int(bool(keys & {'w', 'up'}))  # Поворот вокруг X.
        y = int(bool(keys & {'d', 'right'})) - int(bool(keys & {'a', 'left'}))  # Поворот вокруг Y.
        z = int('e' in keys) - int('q' in keys)  # Поворот вокруг Z.
        zoom = int(bool(keys & {'plus', 'equal'})) - int('minus' in keys)  # Изменение масштаба.
        speed = 2.4 if 'shift' in keys else 1.2  # Угловая скорость.
        self.angles = (self.angles + np.array([x, y, z]) * dt * speed) % (2 * np.pi)  # Новая ориентация.
        self.zoom(zoom * dt * 8)  # Непрерывный масштаб.
        return bool(x or y or z or zoom)  # Запрос нового кадра.

# Отсечение треугольника.
def clip_near(polygon):  # Пакет содержит все атрибуты.
    result = []  # Вершины после отсечения.
    for i, current in enumerate(polygon):  # Рёбра многоугольника.
        previous = polygon[i - 1]  # Начало ребра.
        inside = current[2] >= NEAR  # Видимость конца.
        before = previous[2] >= NEAR  # Видимость начала.
        if inside != before:  # Пересечение плоскости.
            t = (NEAR - previous[2]) / (current[2] - previous[2])  # Доля ребра.
            result.append(previous + t * (current - previous))  # Интерполяция атрибутов.
        if inside:  # Конец внутри объёма.
            result.append(current)  # Сохранение вершины.
    return result  # Новый многоугольник.

# Программная визуализация.
class Renderer:  # Буферы и алгоритмы.
    # Начальная детализация.
    def __init__(self, rows=24, sectors=40):  # Начальная детализация.
        self.mesh = build_mesh(rows, sectors)  # Геометрия поверхности.

    # Один кадр.
    def render(self, camera, mode=2, wire=False, specular=True, width=900, height=650, model=None):  # Один кадр.
        self.w, self.h = width, height  # Размер изображения.
        self.scale = height / (2 * math.tan(FOV / 2))  # Фокусное расстояние.
        self.camera = camera  # Текущая камера.
        self.basis = camera.basis()  # Видовой базис.
        self.rgb = np.full((height, width, 3), (16, 16, 24), dtype=np.uint8)  # Фон сцены.
        self.depth = np.full((height, width), np.inf)  # Буфер глубины.
        self.wires = []  # Видимые рёбра.
        p, faces, normals, face_normals = self.mesh  # Данные сетки.
        self.axis_rotation = np.eye(3)  # Исходные направления осей.
        if model is not None:  # Преобразования только объекта.
            rotation = model_rotation(model.angles)  # Матрица ориентации.
            self.axis_rotation = rotation  # Оси вместе с моделью.
            p = (p @ rotation.T) * model.scale  # Поворот и масштаб вершин.
            normals = normals @ rotation.T  # Поворот нормалей вершин.
            face_normals = face_normals @ rotation.T  # Поворот нормалей граней.
        view = (p - camera.eye) @ self.basis.T  # Видовые координаты.
        self.wires = render_surface(p, faces, normals, face_normals, view, camera.eye, LIGHT, mode, specular, wire, self.scale, NEAR, self.rgb, self.depth)  # Скомпилированный обход сетки.
        if wire:  # Наложение сетки.
            draw_edges(self.wires, self.rgb, self.depth)  # Сетка без циклов Python.
        for axis, color in enumerate(((255, 107, 107), (123, 237, 123), (107, 182, 255))):  # Оси сцены.
            a = np.zeros(3)  # Начало координат.
            b = np.zeros(3)  # Конец оси.
            a[axis], b[axis] = -3, 3  # Пределы оси.
            self.world_line(a @ self.axis_rotation.T, b @ self.axis_rotation.T, color)  # Поворот оси.
        image = Image.fromarray(self.rgb)  # Растровое изображение.
        draw = ImageDraw.Draw(image)  # Текст и указатели.
        for axis, name in enumerate(('X', 'Y', 'Z')):  # Подписи мировых осей.
            endpoint = np.zeros(3)  # Положение подписи.
            endpoint[axis] = 3.12  # Отступ от конца.
            endpoint = endpoint @ self.axis_rotation.T  # Подпись вместе с осью.
            v = (endpoint - camera.eye) @ self.basis.T  # Видовая точка.
            if v[2] >= NEAR:  # Подпись перед камерой.
                x, y = self.project(v)[:2]  # Экранная позиция.
                if 5 < x < width - 15 and 5 < y < height - 15:  # Внутри окна.
                    draw.text((x, y), name, fill=(218, 220, 235))  # Имя оси.
        self.triad(draw)  # Постоянный указатель осей.
        return image  # Готовый кадр.

    # Перспективное деление.
    def project(self, v):  # Перспективное деление.
        return np.array([self.w / 2 + self.scale * v[0] / v[2], self.h / 2 - self.scale * v[1] / v[2], v[2]])  # Экран и глубина.

    # Линия с Z-проверкой.
    def screen_line(self, a, b, color):  # Линия с Z-проверкой.
        delta = b[:2] - a[:2]  # Экранное направление.
        low, high = 0.0, 1.0  # Параметр видимого отрезка.
        for k, limit in enumerate((self.w - 1, self.h - 1)):  # Границы окна.
            if abs(delta[k]) < 1e-12:  # Параллельный отрезок.
                if not 0 <= a[k] <= limit:  # Вне полосы.
                    return  # Линия невидима.
            else:  # Пересечение границ.
                ends = sorted(((0 - a[k]) / delta[k], (limit - a[k]) / delta[k]))  # Параметры входа.
                low, high = max(low, ends[0]), min(high, ends[1])  # Общий интервал.
        if low > high:  # Нет пересечения.
            return  # Пропуск линии.
        count = int(np.max(np.abs(delta)) * (high - low)) + 2  # Число отсчётов.
        t = np.linspace(low, high, count)  # Параметры точек.
        xy = np.rint(a[:2] + t[:, None] * delta).astype(int)  # Экранные пиксели.
        x, y = xy[:, 0], xy[:, 1]  # Индексы пикселей.
        z = 1 / ((1 - t) / a[2] + t / b[2])  # Глубина отрезка.
        visible = z <= self.depth[y, x] + 0.018  # Допуск наложения.
        self.rgb[y[visible], x[visible]] = color  # Видимая часть линии.

    # Линия в мировых координатах.
    def world_line(self, a, b, color):  # Линия в мировых координатах.
        a, b = (np.array([a, b]) - self.camera.eye) @ self.basis.T  # Видовые концы.
        if a[2] < NEAR and b[2] < NEAR:  # За ближней плоскостью.
            return  # Отрезок невидим.
        if a[2] < NEAR:  # Отсечение начала.
            a += (b - a) * (NEAR - a[2]) / (b[2] - a[2])  # Точка пересечения.
        if b[2] < NEAR:  # Отсечение конца.
            b += (a - b) * (NEAR - b[2]) / (a[2] - b[2])  # Точка пересечения.
        self.screen_line(self.project(a), self.project(b), color)  # Экранный отрезок.

    # Указатель ориентации.
    def triad(self, draw):  # Указатель ориентации.
        origin = np.array([60, self.h - 65])  # Место указателя.
        orientation = self.basis @ self.axis_rotation  # Оси модели в камере.
        draw.rectangle((8, self.h - 125, 125, self.h - 8), fill=(25, 25, 35))  # Фон указателя.
        for k, (name, color) in enumerate(zip(('X', 'Y', 'Z'), ((255, 107, 107), (123, 237, 123), (107, 182, 255)))):  # Все оси.
            endpoint = origin + 40 * np.array([orientation[0, k], -orientation[1, k]])  # Направление оси.
            draw.line((tuple(origin), tuple(endpoint)), fill=color, width=2)  # Проекция оси.
            draw.text(tuple(endpoint + 3), name, fill=color)  # Имя направления.

# Оконное приложение.
class Application:  # Интерфейс и события.
    # Создание окна.
    def __init__(self, root):  # Создание окна.
        self.root = root  # Главное окно.
        root.title('Поверхность вращения — освещение и вращение модели')  # Заголовок.
        root.geometry('1180x840')  # Исходный размер.
        root.minsize(900, 650)  # Минимальный размер.
        root.configure(bg='#202028')  # Фон панелей.
        self.renderer = Renderer()  # Программный рендерер.
        self.camera = Camera()  # Закреплённая камера.
        self.model = Model()  # Управляемый объект.
        self.keys = set()  # Нажатые клавиши.
        self.drag = None  # Положение мыши.
        self.dirty = True  # Обновление кадра.
        self.last = time.perf_counter()  # Время предыдущего шага.
        self.frame_time = None  # Предыдущее начало кадра.
        self.mode = tk.StringVar(value=MODES[2])  # Исходный режим Фонга.
        self.wire = tk.BooleanVar(value=False)  # Видимость сетки.
        self.specular = tk.BooleanVar(value=True)  # Видимость блика.
        self.rows = tk.IntVar(value=24)  # Отрезки образующей.
        self.sectors = tk.IntVar(value=40)  # Угловые секторы.
        self.style = ttk.Style(root)  # Единое оформление.
        self.style.theme_use('clam')  # Настраиваемая тема.
        self.style.configure('.', background='#202028', foreground='#e4e4ee', font=('Segoe UI', 10))  # Базовая палитра.
        self.style.configure('TFrame', background='#202028')  # Фон контейнеров.
        self.style.configure('TLabel', background='#202028', foreground='#d5d5e4')  # Подписи.
        self.style.configure('Muted.TLabel', foreground='#9696ac')  # Вторичный текст.
        self.style.configure('Title.TLabel', font=('Segoe UI Semibold', 19), foreground='#f3f3fa')  # Заголовок окна.
        self.style.configure('Badge.TLabel', font=('Segoe UI Semibold', 9), foreground='#aebaff')  # Название камеры.
        self.style.configure('TButton', background='#333342', foreground='#ececf4', borderwidth=0, padding=(14, 8))  # Плоские кнопки.
        self.style.map('TButton', background=[('active', '#444459'), ('pressed', '#51516a')])  # Реакция на мышь.
        self.style.configure('Accent.TButton', background='#6676db', foreground='white')  # Основное действие.
        self.style.map('Accent.TButton', background=[('active', '#7d8cef'), ('pressed', '#5263c7')])  # Активное действие.
        self.style.configure('TCheckbutton', background='#202028', foreground='#c9c9db', padding=4)  # Переключатели.
        self.style.map('TCheckbutton', background=[('active', '#202028')], indicatorcolor=[('selected', '#8190ef'), ('!selected', '#3b3b50')])  # Цвет индикатора.
        self.style.configure('TSpinbox', fieldbackground='#14141e', foreground='#e6e6f2', arrowcolor='#c0c0d5', padding=5)  # Числовые поля.
        header = ttk.Frame(root, padding=(20, 14, 20, 8))  # Заголовочная панель.
        header.pack(fill='x')  # Верх окна.
        ttk.Label(header, text='Поверхность вращения', style='Title.TLabel').pack(side='left')  # Название сцены.
        ttk.Label(header, text='ЗАКРЕПЛЁННАЯ КАМЕРА', style='Badge.TLabel').pack(side='right')  # Режим управления.
        toolbar = ttk.Frame(root, padding=(20, 8, 20, 8))  # Настройки изображения.
        toolbar.pack(fill='x')  # Вторая строка.
        ttk.Label(toolbar, text='ЗАКРАСКА', style='Muted.TLabel').pack(side='left', padx=(0, 12))  # Группа режимов.
        for name in MODES:  # Переключатели закраски.
            button = tk.Radiobutton(toolbar, text=name, value=name, variable=self.mode, indicatoron=False, command=self.invalidate, takefocus=False, bg='#30303e', fg='#dadae8', selectcolor='#606fcb', activebackground='#44445b', activeforeground='white', relief='flat', bd=0, padx=19, pady=7, font=('Segoe UI', 10))  # Сегмент режима.
            button.pack(side='left', padx=(0, 3))  # Расположение сегмента.
        ttk.Checkbutton(toolbar, text='Триангуляция', variable=self.wire, command=self.invalidate, takefocus=False).pack(side='left', padx=(18, 8))  # Рёбра сетки.
        ttk.Checkbutton(toolbar, text='Блик', variable=self.specular, command=self.invalidate, takefocus=False).pack(side='left')  # Зеркальная компонента.
        ttk.Button(toolbar, text='Сброс модели  R', command=self.reset, takefocus=False).pack(side='right')  # Начальное положение.
        density = ttk.Frame(root, padding=(20, 2, 20, 12))  # Детализация поверхности.
        density.pack(fill='x')  # Третья строка.
        ttk.Label(density, text='СЕТКА', style='Muted.TLabel').pack(side='left', padx=(0, 20))  # Группа геометрии.
        for label, variable, low, high in (('По высоте', self.rows, 6, 60), ('По окружности', self.sectors, 8, 100)):  # Параметры сетки.
            ttk.Label(density, text=label).pack(side='left', padx=(0, 8))  # Имя параметра.
            ttk.Spinbox(density, from_=low, to=high, textvariable=variable, width=5).pack(side='left', padx=(0, 16))  # Число отсчётов.
        ttk.Button(density, text='Перестроить', command=self.rebuild, style='Accent.TButton', takefocus=False).pack(side='left')  # Обновление геометрии.
        self.mesh_info = ttk.Label(density, text='1000 вершин  /  1920 треугольников', style='Muted.TLabel')  # Текущая детализация.
        self.mesh_info.pack(side='right')  # Сводка геометрии.
        self.help_label = ttk.Label(root, text='ЛКМ / ПКМ  вращение     W/S  вокруг X     A/D  вокруг Y     Q/E  вокруг Z     Колесо / +/−  масштаб     R  сброс', style='Muted.TLabel', padding=(20, 9))  # Памятка управления.
        self.help_label.pack(fill='x')  # Полоса подсказок.
        self.canvas = tk.Canvas(root, bg='#101018', highlightthickness=0, takefocus=True)  # Тёмное поле сцены.
        self.canvas.pack(fill='both', expand=True, padx=12)  # Основная область.
        self.item = self.canvas.create_image(0, 0, anchor='nw')  # Единственное изображение.
        self.status = ttk.Label(root, text='Подготовка сцены…', style='Muted.TLabel', padding=(20, 10))  # Нижняя строка.
        self.status.pack(fill='x')  # Состояние приложения.
        self.canvas.bind('<Configure>', self.invalidate)  # Изменение размеров.
        self.canvas.bind('<ButtonPress-1>', self.drag_start)  # Начало вращения.
        self.canvas.bind('<B1-Motion>', self.drag_move)  # Поворот модели.
        self.canvas.bind('<ButtonRelease-1>', lambda event: setattr(self, 'drag', None))  # Завершение вращения.
        self.canvas.bind('<ButtonPress-3>', self.drag_start)  # Начало вращения.
        self.canvas.bind('<B3-Motion>', self.drag_move)  # Поворот модели.
        self.canvas.bind('<ButtonRelease-3>', lambda event: setattr(self, 'drag', None))  # Завершение вращения.
        self.canvas.bind('<MouseWheel>', self.wheel)  # Масштабирование колесом.
        root.bind('<KeyPress>', self.key_down)  # Нажатие клавиши.
        root.bind('<KeyRelease>', self.key_up)  # Отпускание клавиши.
        root.bind('<FocusOut>', lambda event: self.keys.clear())  # Сброс при потере фокуса.
        self.canvas.focus_set()  # Начальный фокус.
        self.executor = ThreadPoolExecutor(max_workers=1)  # Однократная подготовка.
        self.warmup = self.executor.submit(lambda: Renderer().render(Camera(), wire=True, width=96, height=96))  # Прогрев вычислений.
        root.after(50, self.tick)  # Запуск цикла.

    # Запрос обновления.
    def invalidate(self, event=None):  # Запрос обновления.
        self.dirty = True  # Новый кадр.

    # Восстановление ракурса.
    def reset(self):  # Восстановление ракурса.
        self.model.reset()  # Исходная модель.
        self.keys.clear()  # Нет активного движения.
        self.invalidate()  # Обновление изображения.

    # Изменение детализации.
    def rebuild(self):  # Изменение детализации.
        try:  # Проверка введённых значений.
            rows, sectors = self.rows.get(), self.sectors.get()  # Чтение параметров.
            if not (6 <= rows <= 60 and 8 <= sectors <= 100):  # Допустимые границы.
                raise ValueError('Вне диапазона')  # Отказ от неверной сетки.
            self.renderer = Renderer(rows, sectors)  # Новая поверхность.
            self.mesh_info.config(text=f'{(rows + 1) * sectors} вершин  /  {2 * rows * sectors} треугольников')  # Обновление сводки.
            self.canvas.focus_set()  # Возврат управления.
            self.invalidate()  # Новый кадр.
        except (ValueError, tk.TclError):  # Ошибка ввода.
            self.status.config(text='Допустимо: 6–60 отсчётов по высоте, 8–100 по окружности.')  # Подсказка.

    # Независимость от раскладки Windows.
    def key_name(self, event):  # Независимость от раскладки Windows.
        codes = {87: 'w', 83: 's', 65: 'a', 68: 'd', 81: 'q', 69: 'e', 82: 'r', 16: 'shift', 37: 'left', 38: 'up', 39: 'right', 40: 'down', 187: 'equal', 189: 'minus', 107: 'plus', 109: 'minus'}  # Физические клавиши.
        return codes.get(event.keycode, event.keysym.lower())  # Имя действия.

    # Начало движения.
    def key_down(self, event):  # Начало движения.
        if self.root.focus_get() != self.canvas:  # Ввод в поле настроек.
            return  # Камера неподвижна.
        name = self.key_name(event)  # Текущая клавиша.
        if name == 'r':  # Команда сброса.
            self.reset()  # Начальный ракурс.
        self.keys.add(name)  # Сохранение нажатия.

    # Завершение движения.
    def key_up(self, event):  # Завершение движения.
        self.keys.discard(self.key_name(event))  # Удаление клавиши.

    # Начало поворота.
    def drag_start(self, event):  # Начало поворота.
        self.canvas.focus_set()  # Фокус на сцене.
        self.drag = (event.x, event.y)  # Начальная точка.

    # Вращение при перетаскивании.
    def drag_move(self, event):  # Вращение при перетаскивании.
        if self.drag is not None:  # Доступна прошлая точка.
            self.model.rotate(event.x - self.drag[0], event.y - self.drag[1])  # Поворот объекта.
            self.drag = (event.x, event.y)  # Новая точка.
            self.invalidate()  # Новый ракурс.

    # Изменение масштаба модели.
    def wheel(self, event):  # Изменение масштаба модели.
        self.model.zoom(event.delta / 120)  # Масштаб объекта.
        self.invalidate()  # Обновление сцены.

    # Кадровый цикл.
    def tick(self):  # Кадровый цикл.
        if not self.warmup.done():  # Подготовка ещё не завершена.
            self.root.after(16, self.tick)  # Окно доступно для событий.
            return  # Отрисовка после подготовки.
        self.warmup.result()  # Проверка успешной подготовки.
        self.executor.shutdown(wait=False)  # Освобождение рабочего потока.
        now = time.perf_counter()  # Текущий момент.
        dt = min(now - self.last, 0.15)  # Ограничение скачка.
        self.last = now  # Время следующего шага.
        self.dirty |= self.model.update(self.keys, dt)  # Непрерывное вращение.
        if self.dirty:  # Есть изменения.
            width = max(300, min(self.canvas.winfo_width(), 1400))  # Предел вычислений.
            height = max(240, round(width * self.canvas.winfo_height() / max(self.canvas.winfo_width(), 1)))  # Соотношение сторон.
            image = self.renderer.render(self.camera, MODES.index(self.mode.get()), self.wire.get(), self.specular.get(), width, height, self.model)  # Рендеринг сцены.
            size = (max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))  # Размер холста.
            self.photo = ImageTk.PhotoImage(image.resize(size, Image.Resampling.BILINEAR))  # Изображение Tk.
            self.canvas.itemconfigure(self.item, image=self.photo)  # Замена кадра.
            p, f, _, _ = self.renderer.mesh  # Параметры модели.
            x, y, z = np.degrees(self.model.angles)  # Углы модели.
            elapsed = time.perf_counter() - now  # Время рендеринга.
            fps = 1 / (now - self.frame_time) if self.frame_time is not None and now - self.frame_time < 0.5 else 0  # Частота обновлений.
            self.frame_time = now  # Начало текущего кадра.
            self.status.config(text=f'{fps:4.0f} FPS    •    {elapsed * 1000:.1f} мс    •    {self.mode.get()}    |    {len(p)} вершин / {len(f)} треугольников    |    Углы ({x:.0f}°; {y:.0f}°; {z:.0f}°)  ×{self.model.scale:.2f}')  # Состояние сцены.
            self.dirty = False  # Обновление завершено.
        delay = max(1, round(1000 / 60 - (time.perf_counter() - now) * 1000))  # Учёт времени отрисовки.
        self.root.after(delay, self.tick)  # Следующий шаг.

# Точка входа.
def main():  # Окно или сохранение кадров.
    parser = argparse.ArgumentParser(description='Поверхность вращения и программная закраска')  # Параметры запуска.
    parser.add_argument('--render', type=Path, help='Папка для контрольных изображений')  # Режим без окна.
    args = parser.parse_args()  # Чтение аргументов.
    if args.render:  # Сохранение результатов.
        args.render.mkdir(parents=True, exist_ok=True)  # Каталог изображений.
        renderer, camera = Renderer(), Camera()  # Сцена и камера.
        for mode, name in enumerate(('flat', 'gouraud', 'phong')):  # Три режима.
            renderer.render(camera, mode).save(args.render / f'{name}.png')  # Изображение режима.
        renderer.render(camera, 0, True).save(args.render / 'mesh.png')  # Триангуляция.
        model = Model()  # Независимые преобразования объекта.
        model.angles = np.radians([35, 25, 15])  # Контрольный поворот.
        renderer.render(camera, 2, model=model).save(args.render / 'rotation.png')  # Вращение при неподвижной камере.
    else:  # Интерактивный запуск.
        root = tk.Tk()  # Создание окна.
        Application(root)  # Настройка приложения.
        root.mainloop()  # Обработка событий.

# Запуск самостоятельного файла.
if __name__ == '__main__':  # Проверка точки входа.
    main()  # Выполнение программы.
