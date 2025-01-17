# Speed Tracker - Программа для отслеживания скорости объектов на видео

Этот репозиторий содержит программу на Python для отслеживания скорости движения объектов на видео в реальном времени или на предварительно записанном видео.

## Описание

Программа использует библиотеку OpenCV для обработки видео и вычисляет скорость движения объектов, сравнивая изменения между кадрами. Она предоставляет следующие возможности:

*   **Обработка видео в реальном времени:** Поддерживает работу с веб-камерами.
*   **Обработка видеофайлов:** Позволяет анализировать заранее записанные видеофайлы.
*   **Настройка параметров:** Предоставляет возможность настроить различные параметры, такие как пороговое значение для определения движения, размер размытия и т.д.
*   **Вывод скорости в реальном времени:** Выводит скорость движения объекта в консоль с заданной частотой.
*   **Калибровка:** Вы можете откалибровать систему, задав количество пикселей, которые объект проходит за известное расстояние.
*   **Сглаживание:** Скорость сглаживается с помощью скользящего среднего.

## Предварительные требования

Прежде чем начать работу, убедитесь, что на вашем компьютере установлены следующие программы и библиотеки:

*   **Python 3.7 или новее:**
    *   Вы можете скачать Python с официального сайта: [https://www.python.org/downloads/](https://www.python.org/downloads/)
*   **pip:** Пакетный менеджер для Python (обычно входит в состав Python).
*   **Необходимые библиотеки Python:** Вы можете установить их с помощью pip:

    ```bash
    pip install opencv-python numpy
    ```

    **Рекомендуется использовать virtual environment:**

    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # В Windows
    source .venv/bin/activate # В MacOS и Linux

    pip install opencv-python numpy
    ```

## Начало работы

1.  **Клонируйте репозиторий:**

    ```bash
    git clone <адрес_вашего_репозитория>
    cd speed-video-cam
    ```
    
    https://github.com/SK1LAS/speed-video-cam

2.  **Установите необходимые зависимости:**

    ```bash
    # Внутри папки проекта
    pip install -r requirements.txt
    ```
    Если у вас нет файла `requirements.txt`, вы можете создать его и добавить `opencv-python` и `numpy`.
3. **Запустите программу:**

    ```bash
    python speed_tracker.py
    ```

4.  **Следуйте инструкциям в консоли:**
   После запуска программа запросит у вас следующие параметры:

    *   **`Введите путь к видеофайлу (или '0' для использования камеры):`**
         * Укажите путь к видеофайлу, если вы хотите проанализировать существующее видео.
         * Введите `0`, если хотите использовать камеру по умолчанию.
    *   **`Введите расстояние до объекта (в метрах):`** Укажите расстояние от камеры до объекта, скорость которого вы хотите измерить.
    *   **`Введите частоту вывода скорости (в секундах):`** Укажите, как часто вы хотите, чтобы скорость выводилась на консоль.
    *   **`Введите пороговое значение для определения движения (0-255):`** Это значение определяет, насколько сильно должны измениться пиксели между кадрами, чтобы считаться движением. Начните со значения 25 и настраивайте по мере необходимости.
    *   **`Введите размер размытия по Гауссу (нечетное число, например, 5):`** Значение для размытия кадра для устранения шума. Начните со значения 5.
    *   **`Введите нижний порог для детектора Canny:`**  Нижний порог для определения границ. Начните со значения 50.
    *   **`Введите верхний порог для детектора Canny:`** Верхний порог для определения границ. Начните со значения 150.
    *   **`Введите коэффициент масштабирования скорости:`**  Используется для преобразования скорости из "пикселей в секунду" в "метры в секунду".
    *  **`Введите количество пикселей, которые объект проходит за 1 метр:`** Используется для калибровки системы.
    *  **`Введите размер окна для скользящего среднего:`**  Используется для сглаживания значения скорости. Начните со значения 5.

**Примечание:**
* Если вы используете видеофайл, убедитесь, что он находится в том же каталоге, что и программа, или укажите полный путь к файлу.
*   Параметры настройки могут варьироваться в зависимости от ваших целей.

## Устранение неполадок

*   **Не удалось открыть видеопоток:** Проверьте, правильно ли указан путь к видеофайлу или индекс камеры.
*   **Ошибка с `numpy` и `cv2`:** Убедитесь, что ваши версии `numpy` и `opencv-python` совместимы. Для этого переустановите их как было сказано в шаге "Предварительные требования".
*   **Неправильные показания скорости:** Убедитесь, что правильно откалибровали параметр `pixel_distance`.

## Лицензия

Этот проект распространяется под лицензией MIT.

## Автор

SK1LAS