# Подключение вычислений.
import math  # Скалярные функции.
import numpy as np  # Буферы кадра.
from numba import njit  # Компиляция горячих циклов.

# Освещение одной точки.
@njit(cache=True, inline='always')  # Встраивание расчёта.
def shade(px, py, pz, nx, ny, nz, eye, light, specular):  # Интенсивность точки.
    inverse = 1.0 / max(math.sqrt(nx * nx + ny * ny + nz * nz), 1e-12)  # Длина нормали.
    nx, ny, nz = nx * inverse, ny * inverse, nz * inverse  # Единичная нормаль.
    lx, ly, lz = light[0] - px, light[1] - py, light[2] - pz  # Направление к свету.
    inverse = 1.0 / max(math.sqrt(lx * lx + ly * ly + lz * lz), 1e-12)  # Обратная длина.
    lx, ly, lz = lx * inverse, ly * inverse, lz * inverse  # Единичный луч.
    diffuse = max(nx * lx + ny * ly + nz * lz, 0.0)  # Диффузная компонента.
    intensity = 0.16 + 0.68 * diffuse  # Фон и рассеяние.
    if specular and diffuse > 0:  # Зеркальное отражение.
        vx, vy, vz = eye[0] - px, eye[1] - py, eye[2] - pz  # Вектор к камере.
        inverse = 1.0 / max(math.sqrt(vx * vx + vy * vy + vz * vz), 1e-12)  # Обратная длина.
        vx, vy, vz = vx * inverse, vy * inverse, vz * inverse  # Единичный взгляд.
        cosine = (2 * diffuse * nx - lx) * vx + (2 * diffuse * ny - ly) * vy + (2 * diffuse * nz - lz) * vz  # Отражённый луч.
        intensity += 0.28 * max(cosine, 0.0) ** 32  # Блик.
    return min(max(intensity, 0.0), 1.0)  # Допустимая яркость.

# Растеризация одного треугольника.
@njit(cache=True)  # Машинный код цикла пикселей.
def raster(tri, screen, rgb, depth, mode, flat, eye, light, specular):  # Заполнение видимой части.
    height, width = depth.shape  # Размер буфера.
    x0 = max(0, int(math.floor(min(screen[0, 0], screen[1, 0], screen[2, 0]))))  # Левая граница.
    x1 = min(width - 1, int(math.ceil(max(screen[0, 0], screen[1, 0], screen[2, 0]))))  # Правая граница.
    y0 = max(0, int(math.floor(min(screen[0, 1], screen[1, 1], screen[2, 1]))))  # Верхняя граница.
    y1 = min(height - 1, int(math.ceil(max(screen[0, 1], screen[1, 1], screen[2, 1]))))  # Нижняя граница.
    ax, ay = screen[0, 0], screen[0, 1]  # Первая вершина.
    bx, by = screen[1, 0], screen[1, 1]  # Вторая вершина.
    cx, cy = screen[2, 0], screen[2, 1]  # Третья вершина.
    denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)  # Удвоенная площадь.
    if abs(denominator) < 1e-9 or x0 > x1 or y0 > y1:  # Невидимая проекция.
        return False  # Пропуск грани.
    inverse = 1.0 / denominator  # Общий множитель.
    za, zb, zc = 1.0 / tri[0, 2], 1.0 / tri[1, 2], 1.0 / tri[2, 2]  # Обратные глубины.
    written = False  # Признак видимости.
    for y in range(y0, y1 + 1):  # Строки изображения.
        for x in range(x0, x1 + 1):  # Пиксели строки.
            u = ((by - cy) * (x + 0.5 - cx) + (cx - bx) * (y + 0.5 - cy)) * inverse  # Первый вес.
            v = ((cy - ay) * (x + 0.5 - cx) + (ax - cx) * (y + 0.5 - cy)) * inverse  # Второй вес.
            w = 1.0 - u - v  # Третий вес.
            if u < -1e-9 or v < -1e-9 or w < -1e-9:  # Вне треугольника.
                continue  # Следующий пиксель.
            a, b, c = u * za, v * zb, w * zc  # Поправка перспективы.
            z = 1.0 / max(a + b + c, 1e-12)  # Глубина поверхности.
            if z >= depth[y, x]:  # Скрытая поверхность.
                continue  # Освещение не требуется.
            value = flat  # Постоянный цвет грани.
            if mode == 1:  # Закраска Гуро.
                value = (a * tri[0, 9] + b * tri[1, 9] + c * tri[2, 9]) * z  # Интерполяция яркости.
            elif mode == 2:  # Закраска Фонга.
                a, b, c = a * z, b * z, c * z  # Нормированные веса.
                px = a * tri[0, 3] + b * tri[1, 3] + c * tri[2, 3]  # Мировой X.
                py = a * tri[0, 4] + b * tri[1, 4] + c * tri[2, 4]  # Мировой Y.
                pz = a * tri[0, 5] + b * tri[1, 5] + c * tri[2, 5]  # Мировой Z.
                nx = a * tri[0, 6] + b * tri[1, 6] + c * tri[2, 6]  # Нормаль по X.
                ny = a * tri[0, 7] + b * tri[1, 7] + c * tri[2, 7]  # Нормаль по Y.
                nz = a * tri[0, 8] + b * tri[1, 8] + c * tri[2, 8]  # Нормаль по Z.
                value = shade(px, py, pz, nx, ny, nz, eye, light, specular)  # Освещение пикселя.
            gray = int(np.rint(255 * value))  # Уровень серого.
            rgb[y, x, 0] = gray  # Красный канал.
            rgb[y, x, 1] = gray  # Зелёный канал.
            rgb[y, x, 2] = gray  # Синий канал.
            depth[y, x] = z  # Запись глубины.
            written = True  # Грань видима.
    return written  # Результат заполнения.

# Построение кадра поверхности.
@njit(cache=True, nogil=True)  # Компиляция обхода сетки.
def render_surface(p, faces, normals, face_normals, view, eye, light, mode, specular, wire, scale, near, rgb, depth):  # Общий конвейер.
    height, width = depth.shape  # Размер изображения.
    packed = np.empty((3, 10))  # Атрибуты исходной грани.
    clipped = np.empty((4, 10))  # Многоугольник отсечения.
    tri = np.empty((3, 10))  # Текущий треугольник.
    screen = np.empty((3, 3))  # Экранные координаты.
    edges = np.empty((len(faces) * 6 if wire else 0, 2, 3))  # Рёбра сетки.
    edge_count = 0  # Число сохранённых рёбер.
    for face_index in range(len(faces)):  # Обход всех граней.
        face = faces[face_index]  # Индексы вершин.
        px = (p[face[0], 0] + p[face[1], 0] + p[face[2], 0]) / 3  # Центр по X.
        py = (p[face[0], 1] + p[face[1], 1] + p[face[2], 1]) / 3  # Центр по Y.
        pz = (p[face[0], 2] + p[face[1], 2] + p[face[2], 2]) / 3  # Центр по Z.
        normal = face_normals[face_index]  # Нормаль грани.
        facing = normal[0] * (eye[0] - px) + normal[1] * (eye[1] - py) + normal[2] * (eye[2] - pz)  # Ориентация стороны.
        sign = 1.0 if facing > 0 else -1.0  # Двустороннее освещение.
        flat = 0.0  # Начальная интенсивность.
        if mode == 0:  # Расчёт только для плоского режима.
            flat = shade(px, py, pz, normal[0] * sign, normal[1] * sign, normal[2] * sign, eye, light, specular)  # Яркость центра.
        for corner in range(3):  # Подготовка вершин.
            index = face[corner]  # Индекс вершины.
            for component in range(3):  # Координаты и нормали.
                packed[corner, component] = view[index, component]  # Видовое положение.
                packed[corner, component + 3] = p[index, component]  # Мировое положение.
                packed[corner, component + 6] = normals[index, component] * sign  # Нормаль стороны.
            packed[corner, 9] = 0.0  # Начальная яркость вершины.
            if mode == 1:  # Освещение вершин только для Гуро.
                packed[corner, 9] = shade(packed[corner, 3], packed[corner, 4], packed[corner, 5], packed[corner, 6], packed[corner, 7], packed[corner, 8], eye, light, specular)  # Яркость вершины.
        count = 0  # Число вершин после отсечения.
        for corner in range(3):  # Рёбра исходной грани.
            previous = (corner + 2) % 3  # Предыдущая вершина.
            inside = packed[corner, 2] >= near  # Видимость конца.
            before = packed[previous, 2] >= near  # Видимость начала.
            if inside != before:  # Пересечение плоскости.
                t = (near - packed[previous, 2]) / (packed[corner, 2] - packed[previous, 2])  # Параметр пересечения.
                for component in range(10):  # Интерполяция всех атрибутов.
                    clipped[count, component] = packed[previous, component] + t * (packed[corner, component] - packed[previous, component])  # Атрибут пересечения.
                count += 1  # Добавление точки.
            if inside:  # Видимый конец ребра.
                clipped[count] = packed[corner]  # Копирование вершины.
                count += 1  # Новый размер.
        for part in range(1, count - 1):  # Триангуляция отсечения.
            tri[0], tri[1], tri[2] = clipped[0], clipped[part], clipped[part + 1]  # Вершины веера.
            for corner in range(3):  # Проекции вершин.
                screen[corner, 0] = width / 2 + scale * tri[corner, 0] / tri[corner, 2]  # Экранный X.
                screen[corner, 1] = height / 2 - scale * tri[corner, 1] / tri[corner, 2]  # Экранный Y.
                screen[corner, 2] = tri[corner, 2]  # Глубина.
            visible = raster(tri, screen, rgb, depth, mode, flat, eye, light, specular)  # Заполнение грани.
            if visible and wire:  # Видимая сетка.
                for corner in range(3):  # Три ребра.
                    edges[edge_count, 0] = screen[corner]  # Начало ребра.
                    edges[edge_count, 1] = screen[(corner + 1) % 3]  # Конец ребра.
                    edge_count += 1  # Новое ребро.
    return edges[:edge_count]  # Рёбра для наложения.

# Быстрое наложение рёбер.
@njit(cache=True, nogil=True)  # Компиляция линейных проходов.
def draw_edges(edges, rgb, depth):  # Сетка с проверкой глубины.
    height, width = depth.shape  # Размер изображения.
    for edge in edges:  # Все видимые рёбра.
        a, b = edge[0], edge[1]  # Концы отрезка.
        dx, dy = b[0] - a[0], b[1] - a[1]  # Экранное направление.
        low, high = 0.0, 1.0  # Видимый интервал.
        for component in range(2):  # Границы окна.
            delta = dx if component == 0 else dy  # Текущая координата.
            limit = width - 1 if component == 0 else height - 1  # Размер полосы.
            if abs(delta) < 1e-12:  # Параллельный отрезок.
                if a[component] < 0 or a[component] > limit:  # За границей.
                    high = -1.0  # Пустое пересечение.
            else:  # Пересечение границ.
                first, second = -a[component] / delta, (limit - a[component]) / delta  # Вход и выход.
                low = max(low, min(first, second))  # Начало интервала.
                high = min(high, max(first, second))  # Конец интервала.
        if low > high:  # Невидимый отрезок.
            continue  # Следующее ребро.
        count = int(max(abs(dx), abs(dy)) * (high - low)) + 2  # Число отсчётов.
        for step in range(count):  # Пиксели отрезка.
            t = low + (high - low) * step / (count - 1)  # Параметр точки.
            x = min(width - 1, max(0, int(np.rint(a[0] + t * dx))))  # Экранный X.
            y = min(height - 1, max(0, int(np.rint(a[1] + t * dy))))  # Экранный Y.
            z = 1.0 / ((1 - t) / a[2] + t / b[2])  # Глубина точки.
            if z <= depth[y, x] + 0.018:  # Видимость ребра.
                rgb[y, x, 0], rgb[y, x, 1], rgb[y, x, 2] = 100, 105, 125  # Цвет сетки.
