# Подключение проверок.
import unittest  # Стандартный набор тестов.
import numpy as np  # Численные сравнения.
from main import Camera, Renderer, build_mesh, clip_near, lighting, NEAR, Model, model_rotation  # Проверяемые компоненты.
from fast_render import shade  # Ускоренная модель освещения.

# Проверки геометрии и визуализации.
class SceneTests(unittest.TestCase):  # Независимые сценарии.
    # Совпадение вычислительных вариантов.
    def test_shader_equivalence(self):  # Проверка ускоренного освещения.
        from main import LIGHT  # Положение источника.
        rng = np.random.default_rng(42)  # Воспроизводимые точки.
        eye = np.array([4.5, 2.6, 6.5])  # Положение камеры.
        for specular in (False, True):  # Два состава освещения.
            for _ in range(40):  # Разные положения и нормали.
                p, n = rng.normal(size=(2, 3))  # Контрольные векторы.
                expected = float(lighting(p, n, eye, specular))  # Модель NumPy.
                actual = shade(*p, *n, eye, LIGHT, specular)  # Скомпилированная модель.
                self.assertAlmostEqual(actual, expected, places=12)  # Численное совпадение.

    def test_mesh(self):  # Топология поверхности.
        p, f, n, fn = build_mesh(12, 20)  # Контрольная сетка.
        self.assertEqual(p.shape, (260, 3))  # Число вершин.
        self.assertEqual(f.shape, (480, 3))  # Число треугольников.
        self.assertTrue(np.all(np.linalg.norm(np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]]), axis=1) > 0))  # Нет вырождения.
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1)  # Единичные нормали.
        centers = p[f].mean(axis=1)  # Центры граней.
        self.assertTrue(np.all(centers[:, 0] * fn[:, 0] + centers[:, 2] * fn[:, 2] > 0))  # Внешний обход.
        edges = {}  # Кратности рёбер.
        for face in f:  # Все грани.
            for a, b in zip(face, np.roll(face, -1)):  # Соседние индексы.
                key = tuple(sorted((a, b)))  # Неориентированное ребро.
                edges[key] = edges.get(key, 0) + 1  # Число смежных граней.
        self.assertEqual(sum(v == 1 for v in edges.values()), 40)  # Только два открытых торца.
        self.assertTrue(all(v in (1, 2) for v in edges.values()))  # Нет разрывов шва.

    def test_light(self):  # Контроль интенсивности.
        point = np.zeros(3)  # Начало координат.
        normal = np.array([0, -1, 0])  # Теневая сторона.
        self.assertAlmostEqual(float(lighting(point, normal, np.array([0, 0, 5]))), 0.16)  # Только фон.
        values = lighting(np.zeros((3, 3)), np.eye(3), np.array([0, 0, 5]))  # Разные нормали.
        self.assertTrue(np.all((values >= 0) & (values <= 1)))  # Диапазон палитры.

    # Неподвижность камеры при вращении.
    def test_camera(self):  # Базис и преобразования объекта.
        camera, model = Camera(), Model()  # Камера и модель.
        origin, basis = camera.eye.copy(), camera.basis()  # Исходные параметры.
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)  # Ортонормальность.
        model.rotate(120, 80)  # Мышиное вращение.
        model.update({'e', 'd'}, 0.4)  # Клавиатурное вращение.
        model.zoom(2)  # Изменение масштаба.
        np.testing.assert_array_equal(camera.eye, origin)  # Камера без переноса.
        np.testing.assert_array_equal(camera.basis(), basis)  # Направление неизменно.
        rotation = model_rotation(model.angles)  # Матрица модели.
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)  # Сохранение углов.
        renderer = Renderer(8, 12)  # Контрольная геометрия.
        original = renderer.mesh[0].copy()  # Исходные вершины.
        before = np.asarray(renderer.render(camera, width=320, height=240))  # Начальный кадр.
        after = np.asarray(renderer.render(camera, width=320, height=240, model=model))  # Поворот модели.
        self.assertTrue(np.any(before != after))  # Изображение изменено.
        np.testing.assert_array_equal(renderer.mesh[0], original)  # Базовая сетка неизменна.
        np.testing.assert_array_equal(renderer.basis, basis)  # Базис камеры неизменен.
        np.testing.assert_allclose(renderer.axis_rotation, rotation, atol=1e-12)  # Оси следуют модели.
        model.reset()  # Сброс объекта.
        np.testing.assert_array_equal(model.angles, np.zeros(3))  # Нулевые углы.
        self.assertEqual(model.scale, 1)  # Единичный масштаб.

    def test_clip(self):  # Пересечение ближней плоскости.
        polygon = [np.array([0., 0., -1., 0.]), np.array([1., 0., 1., 1.]), np.array([0., 1., 1., 1.])]  # Треугольник.
        result = clip_near(polygon)  # Отсечённый четырёхугольник.
        self.assertEqual(len(result), 4)  # Число вершин.
        self.assertTrue(all(v[2] >= NEAR - 1e-12 for v in result))  # Предел глубины.
        for v in result:  # Интерполированные атрибуты.
            self.assertAlmostEqual(v[3], (v[2] + 1) / 2)  # Согласованная интерполяция.

    def test_render(self):  # Режимы и сложные ракурсы.
        renderer, camera = Renderer(8, 12), Camera()  # Компактная сцена.
        frames = [np.asarray(renderer.render(camera, mode, width=320, height=240)) for mode in range(3)]  # Три закраски.
        self.assertTrue(np.any(frames[0] != frames[1]))  # Гуро отличается от плоской.
        self.assertTrue(np.any(frames[1] != frames[2]))  # Фонг отличается от Гуро.
        for point in ([0, 0, 0], [0.8, 0, 0], [0, 1.7, 0], [0, 0, 100]):  # Внутри и снаружи.
            camera.eye = np.array(point, dtype=float)  # Положение наблюдателя.
            image = renderer.render(camera, 2, True, width=320, height=240)  # Рендеринг с отсечением.
            self.assertEqual(image.size, (320, 240))  # Корректный кадр.

# Самостоятельный запуск проверок.
if __name__ == '__main__':  # Точка входа.
    unittest.main(verbosity=2)  # Выполнение набора.
