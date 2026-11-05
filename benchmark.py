# Подключение измерений.
import statistics  # Медиана результатов.
import time  # Точный таймер.
from main import Camera, Renderer, Model, MODES  # Компоненты сцены.

# Измерение установившейся скорости.
def main():  # Проверка всех режимов.
    renderer, camera, model = Renderer(), Camera(), Model()  # Одинаковая геометрия.
    print('Кадр 900x650; сетка 24x40; первый прогрев исключён.')  # Условия измерения.
    for mode, name in enumerate(MODES):  # Три варианта закраски.
        for wire in (False, True):  # Без сетки и с сеткой.
            renderer.render(camera, mode, wire, model=model)  # Прогрев компиляции.
            samples = []  # Длительности кадров.
            for _ in range(40):  # Серия измерений.
                model.rotate(0.3, 0)  # Небольшой поворот модели.
                start = time.perf_counter()  # Начало расчёта.
                renderer.render(camera, mode, wire, model=model)  # Полный кадр.
                samples.append((time.perf_counter() - start) * 1000)  # Время в миллисекундах.
            median = statistics.median(samples)  # Устойчивая оценка.
            p95 = sorted(samples)[int(len(samples) * 0.95) - 1]  # Верхняя граница 95 процентов.
            print(f'{name:8} | сетка {str(wire):5} | медиана {median:6.2f} мс | P95 {p95:6.2f} мс')  # Результат без вывода в окно.

# Самостоятельный запуск.
if __name__ == '__main__':  # Проверка точки входа.
    main()  # Выполнение измерений.
