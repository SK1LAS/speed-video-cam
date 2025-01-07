import cv2
import numpy as np
import threading
import time
import os
from typing import Union, Tuple, Optional, List


class CustomTimer:
    def __init__(self, interval: float, event: threading.Event):
        """
        Инициализирует таймер.

        Args:
            interval: Интервал между срабатываниями таймера в секундах.
            event: Событие, которое устанавливается таймером.
        """
        self.interval = interval
        self.event = event
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """
        Запускает таймер.

        Returns:
            True если таймер был успешно запущен.
        """
        if self._thread is None:
            self._thread = threading.Thread(target=self._run)
            self._thread.daemon = True
            self._thread.start()
            return True
        return False

    def _run(self):
        """
        Основной цикл таймера.
        """
        while not self._stop_event.is_set():
            self.event.set()
            time.sleep(self.interval)
            self.event.clear()

    def stop(self):
        """
        Останавливает таймер.
        """
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._thread = None


class VideoProcessor:
    def __init__(self, video_source: Union[int, str] = 0, threshold_value: int = 25, blur_size: int = 5,
                 canny_low: int = 50, canny_high: int = 150, update_frequency: float = 1.0):
        """
        Инициализирует процессор видео.

        Args:
            video_source: Источник видео (индекс камеры или путь к файлу).
            threshold_value: Пороговое значение для определения движения.
            blur_size: Размер размытия по Гауссу.
            canny_low: Нижний порог для детектора Canny.
            canny_high: Верхний порог для детектора Canny.
            update_frequency: Частота обработки кадров.
        """
        self.video_source = video_source
        self.cap = self._get_video_capture()
        if not self.cap or not self.cap.isOpened():
            print(f"Ошибка: Не удалось открыть источник видео: {self.video_source}")
            exit()

        self.previous_frame: Optional[np.ndarray] = None
        self.threshold_value = threshold_value
        self.blur_size = blur_size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.movement_sum = 0
        self.total_pixels = 0
        self.speed_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self.timer = CustomTimer(update_frequency, self._frame_ready)
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        if not self.timer.start():
            print("Ошибка запуска таймера. Завершение.")
            self.cap.release()
            exit()
        self.is_running = True
        self.last_time: Optional[float] = None

    def _get_video_capture(self) -> Optional[cv2.VideoCapture]:
        """
        Создает объект захвата видео.

        Returns:
            Объект cv2.VideoCapture.
        """
        if isinstance(self.video_source, str):
            return cv2.VideoCapture(self.video_source)
        else:
            return cv2.VideoCapture(int(self.video_source))

    def process_video(self):
        """
        Основной цикл обработки видео.
        """
        while self.is_running:
            ret, current_frame = self.cap.read()
            if not ret:
                print("Ошибка: Не удалось получить кадр.")
                break

            current_time = time.time()
            if self.last_time is not None:
                time_elapsed = current_time - self.last_time
            else:
                time_elapsed = 0.0
            self.last_time = current_time

            gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.GaussianBlur(gray_frame, (self.blur_size, self.blur_size), 0)
            edge_frame = cv2.Canny(blurred_frame, self.canny_low, self.canny_high)
            if self.previous_frame is None:
                self.previous_frame = blurred_frame
                self.total_pixels = gray_frame.size
                continue

            frame_diff = cv2.absdiff(self.previous_frame, blurred_frame)
            _, threshold_image = cv2.threshold(frame_diff, self.threshold_value, 255, cv2.THRESH_BINARY)
            with self.speed_lock:
                self.movement_sum = self._calculate_movement(threshold_image)
                self.time_elapsed = time_elapsed

            self.previous_frame = blurred_frame
            cv2.imshow('Frame', current_frame)
            cv2.imshow('Edges', edge_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break

    def _calculate_movement(self, threshold_image: np.ndarray) -> int:
        """
        Вычисляет количество переместившихся пикселей.

        Args:
            threshold_image: Бинаризованное изображение после сравнения кадров.

        Returns:
            Количество пикселей, где произошло движение.
        """
        return np.sum(threshold_image > 0)

    def get_movement_data(self) -> Tuple[int, int, float]:
        """
        Получает данные о движении.

        Returns:
            Кортеж (movement_sum, total_pixels, time_elapsed).
        """
        with self.speed_lock:
            return self.movement_sum, self.total_pixels, self.time_elapsed

    def get_frames_from_buffer(self) -> threading.Event:
        """
        Возвращает Event объект.
        """
        return self._frame_ready

    def stop(self):
        """
        Останавливает обработку видео и освобождает ресурсы.
        """
        self.is_running = False
        self.timer.stop()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


class SpeedTracker:
    def __init__(self, video_processor: VideoProcessor, scale: float = 100.0, update_frequency: float = 1.0,
                 pixel_distance: float = 100.0, smoothing_window: int = 5):
        """
        Инициализирует трекер скорости.

        Args:
            video_processor: Объект VideoProcessor.
            scale: Коэффициент масштабирования скорости.
            update_frequency: Частота обновления скорости.
            pixel_distance: Количество пикселей, соответствующих известному расстоянию.
            smoothing_window: Размер окна для скользящего среднего.
        """
        self.video_processor = video_processor
        self.scale = scale
        self._current_speed = 0.0
        self.distance = 1.0
        self.pixel_distance = pixel_distance
        self.update_frequency = update_frequency
        self.speed_lock = threading.Lock()
        self.speed_update_event = threading.Event()
        self.timer = CustomTimer(update_frequency, self.speed_update_event)
        self.speed_thread = threading.Thread(target=self.update_speed)
        self.speed_thread.daemon = True
        self.speed_thread.start()
        self.timer.start()
        self.smoothing_window = smoothing_window
        self._speed_history: List[float] = []

    def update_speed(self):
        """
        Обновляет значение скорости.
        """
        while True:
            self.speed_update_event.wait()
            self.speed_update_event.clear()
            movement_sum, total_pixels, time_elapsed = self.video_processor.get_movement_data()

            if total_pixels > 0 and time_elapsed > 0:
                if movement_sum > 0:
                    try:
                        # Вычисление скорости в пикселях/секунду
                        pixel_speed = movement_sum / time_elapsed
                        # Преобразование в метры/секунду
                        speed = (pixel_speed / self.pixel_distance) * self.scale * self.distance
                        self._add_to_history(speed)
                        smoothed_speed = self._get_smoothed_speed()
                        self.set_speed(smoothed_speed)


                    except ZeroDivisionError:
                        print("Ошибка: Деление на ноль при расчете скорости.")
                        self.set_speed(0.0)
                else:
                    self.set_speed(0.0)
            else:
                self.set_speed(0.0)

    def _add_to_history(self, speed: float):
        """Добавляет скорость в историю и ограничивает размер истории."""
        self._speed_history.append(speed)
        if len(self._speed_history) > self.smoothing_window:
            self._speed_history.pop(0)

    def _get_smoothed_speed(self) -> float:
        """Возвращает скользящее среднее скорости."""
        if self._speed_history:
            return sum(self._speed_history) / len(self._speed_history)
        return 0.0

    def get_speed(self) -> float:
        """
        Возвращает текущую скорость.
        """
        with self.speed_lock:
            return self._current_speed

    def set_speed(self, speed: float):
        """
        Устанавливает текущую скорость.
        """
        with self.speed_lock:
            self._current_speed = speed

    def set_distance(self, distance: float):
        """
        Устанавливает расстояние до объекта.
        """
        with self.speed_lock:
            self.distance = distance

    def set_update_frequency(self, frequency: float):
        """
        Устанавливает частоту обновления скорости.
        """
        self.update_frequency = frequency
        self.timer.stop()
        self.timer = CustomTimer(frequency, self.speed_update_event)
        self.timer.start()


def print_speed(speed_tracker: SpeedTracker):
    """
    Выводит текущую скорость в консоль.
    """
    while True:
        speed_tracker.video_processor.get_frames_from_buffer().wait()
        speed_tracker.speed_update_event.set()
        current_speed = speed_tracker.get_speed()
        print(f'Текущая скорость: {current_speed:.2f} м/с')
        speed_tracker.video_processor.get_frames_from_buffer().clear()
        speed_tracker.speed_update_event.clear()


def main():
    """
    Основная функция программы.
    """
    print("Перед началом обработки введите необходимые параметры.")

    video_source = input("Введите путь к видеофайлу (или '0' для использования камеры): ")
    if video_source == '0':
        video_source = 0

    # Ввод параметров
    distance = float(input("Введите расстояние до объекта (в метрах): "))
    update_frequency = float(input("Введите частоту вывода скорости (в секундах): "))
    threshold_value = int(input("Введите пороговое значение для определения движения (0-255): "))
    blur_size = int(input("Введите размер размытия по Гауссу (нечетное число, например, 5): "))
    canny_low = int(input("Введите нижний порог для детектора Canny: "))
    canny_high = int(input("Введите верхний порог для детектора Canny: "))
    scale = float(input("Введите коэффициент масштабирования скорости: "))
    pixel_distance = float(input("Введите количество пикселей, которые объект проходит за 1 метр:"))
    smoothing_window = int(input("Введите размер окна для скользящего среднего:"))

    vp = VideoProcessor(video_source, threshold_value=threshold_value, blur_size=blur_size, canny_low=canny_low,
                        canny_high=canny_high, update_frequency=update_frequency)
    st = SpeedTracker(vp, scale=scale, update_frequency=update_frequency, pixel_distance=pixel_distance,
                      smoothing_window=smoothing_window)
    st.set_distance(distance)

    # Запуск потока для вывода скорости
    speed_thread = threading.Thread(target=print_speed, args=(st,))
    speed_thread.daemon = True
    speed_thread.start()

    try:
        while vp.is_running:
            pass
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        vp.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Произошла ошибка: {e}")