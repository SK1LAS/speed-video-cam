import cv2
import numpy as np
import threading
import time


class VideoProcessor:
    def __init__(self, camera_index=0, threshold_value=25):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Ошибка: Не удалось открыть камеру.")
            exit()

        self.previous_frame = None
        self.threshold_value = threshold_value
        self.movement_sum = 0
        self.total_pixels = 1920 * 1080  # Полное разрешение для 1080p
        self.speed_lock = threading.Lock()
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.daemon = True  # Устанавливаем поток как демонический
        self.processing_thread.start()

    def process_video(self):
        while True:
            ret, current_frame = self.cap.read()
            if not ret:
                print("Ошибка: Не удалось получить кадр.")
                break

            gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

            # Применение детектора границ Canny
            edge_frame = cv2.Canny(blurred_frame, 50, 150)

            if self.previous_frame is None:
                self.previous_frame = blurred_frame
                continue

            frame_diff = cv2.absdiff(self.previous_frame, blurred_frame)
            _, threshold_image = cv2.threshold(frame_diff, self.threshold_value, 255, cv2.THRESH_BINARY)

            with self.speed_lock:
                # Подсчет измененных пикселей
                self.movement_sum = np.sum(threshold_image > 0)  # Количество пикселей, изменившихся

            self.previous_frame = blurred_frame
            cv2.imshow('Frame', current_frame)
            cv2.imshow('Edges', edge_frame)  # Показываем детализированное изображение

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def get_movement_data(self):
        with self.speed_lock:
            return self.movement_sum, self.total_pixels


class SpeedTracker:
    def __init__(self, video_processor, scale=100):
        self.video_processor = video_processor
        self.scale = scale
        self.current_speed = 0.1
        self.distance = 5.0  # начальное расстояние в метрах
        self.update_frequency = 1  # частота обновления
        self.timer_thread = threading.Thread(target=self.update_speed)
        self.timer_thread.daemon = True  # Устанавливаем поток как демонический
        self.timer_thread.start()

    def update_speed(self):
        while True:
            movement_sum, total_pixels = self.video_processor.get_movement_data()
            if total_pixels > 0:
                with self.video_processor.speed_lock:
                    if movement_sum > 0:
                        # Применяем формулу для расчета скорости
                        self.current_speed = (movement_sum / total_pixels) * self.scale * self.distance  # Настройте scale и distance для точности
            time.sleep(self.update_frequency)

    def set_distance(self, distance):
        with self.video_processor.speed_lock:
            self.distance = distance  # Обновляем расстояние для расчета скорости
    def set_update_frequency(self, frequency):
        self.update_frequency = frequency  # Обновляем частоту обновления


def print_speed(speed_tracker):
    while True:
        time.sleep(speed_tracker.update_frequency)
        with speed_tracker.video_processor.speed_lock:
            print(f'Current Speed: {speed_tracker.current_speed:.2f} m/s')


def main():
    print("Перед началом обработки введите необходимые параметры.")

    video_source = input("Введите путь к видеофайлу (или '0' для использования камеры): ")
    if video_source == '0':
        video_source = 0  # Указываем, что нужно использовать камеру

    # Ввод параметров
    distance = float(input("Введите расстояние до объекта (в метрах): "))
    fps = int(input("Введите частоту кадров (FPS) для обработки изображения: "))
    update_frequency = float(input("Введите частоту вывода скорости (в секундах): "))

    vp = VideoProcessor()
    st = SpeedTracker(vp, scale=0.1)  # Настройте масштаб для скорости
    st.set_distance(distance)
    st.set_update_frequency(update_frequency)

    # Запуск потока для вывода скорости
    speed_thread = threading.Thread(target=print_speed, args=(st,))
    speed_thread.daemon = True
    speed_thread.start()

    # Основной цикл программы для обработки кадров
    vp.process_video()


if __name__ == "__main__":
    main()