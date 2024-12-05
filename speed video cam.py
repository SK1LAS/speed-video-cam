import cv2
import threading
import time
import numpy as np


class CameraSpeedCalculator:
    def __init__(self, distance_to_object, fps, display_interval):
        self.distance_to_object = distance_to_object  # расстояние до объекта в метрах
        self.fps = fps  # частота кадров (FPS)
        self.display_interval = display_interval  # частота вывода скорости в секундах
        self.last_frame = None
        self.speed_history = []  # Для хранения предыдущих значений скорости
        self.speed = 0.0
        self.speed_smoothing_factor = 0.8  # Фактор сглаживания скорости

        # Открываем видеопоток с камеры
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Ошибка при подключении к камере")

        # Устанавливаем разрешение 1920x1080
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.running = True

        # Запуск потоков
        self.display_thread = threading.Thread(target=self.display_speed)
        self.display_thread.start()

    def calculate_speed(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.last_frame is not None:
                # Вычисление оптического потока
                flow = cv2.calcOpticalFlowFarneback(self.last_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                avg_magnitude = np.mean(magnitude)

                # Пример расчета скорости (в км/ч)
                new_speed = (avg_magnitude / self.distance_to_object) * self.fps * 3.6

                # Сглаживание значений скорости
                if self.speed_history:
                    new_speed = (self.speed_smoothing_factor * self.speed_history[-1] + (
                                1 - self.speed_smoothing_factor) * new_speed)
                self.speed_history.append(new_speed)

                # Среднее значение за последние несколько измерений
                if len(self.speed_history) > 5:
                    self.speed = np.mean(self.speed_history[-5:])
                else:
                    self.speed = new_speed

            self.last_frame = gray_frame

            time.sleep(1 / self.fps)

    def display_speed(self):
        while self.running:
            time.sleep(self.display_interval)  # Пауза перед выводом
            print(f"Текущая скорость: {self.speed:.2f} км/ч")

    def run(self):
        try:
            self.calculate_speed()
        finally:
            self.running = False
            self.display_thread.join()
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    distance_to_object = float(input("Введите расстояние до объекта в метрах: "))
    fps = int(input("Введите частоту кадров (FPS): "))
    display_interval = float(input("Введите частоту вывода скорости в секундах: "))

    speed_calculator = CameraSpeedCalculator(distance_to_object, fps, display_interval)
    speed_calculator.run()
