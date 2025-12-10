# src/main.py
import cv2
import numpy as np
import json
import os
from datetime import datetime


class WallDetector:
    def __init__(self, min_wall_length=50, merge_distance=15, min_thickness=0.5):
        """
        Инициализация детектора стен

        Args:
            min_wall_length: минимальная длина стены в пикселях
            merge_distance: максимальное расстояние для объединения стен
            min_thickness: минимальная толщина для коротких стен (0-1)
        """
        self.min_wall_length = min_wall_length
        self.merge_distance = merge_distance
        self.min_thickness = min_thickness
        self.walls = []

    def load_image(self, image_path):
        """Загрузка изображения"""
        print(f"Загрузка изображения: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        self.original = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Улучшаем контраст
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def preprocess_image(self, image):
        """Предобработка изображения"""

        # 1. Бинаризация
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 2. Удаление шума
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # 3. Утолщение линий
        kernel_line = np.ones((3, 3), np.uint8)
        thickened = cv2.dilate(cleaned, kernel_line, iterations=1)

        # 4. Скелетонизация
        skeleton = cv2.erode(thickened, kernel_line, iterations=1)

        return skeleton

    def estimate_line_thickness(self, x1, y1, x2, y2, binary_img):
        """Оценка толщины линии (возвращает одно число от 0 до 1)"""

        # Создаем маску шириной 5 пикселей
        mask = np.zeros_like(binary_img)
        cv2.line(mask, (x1, y1), (x2, y2), 255, 5)

        # Находим пересечение
        intersection = cv2.bitwise_and(mask, binary_img)

        # Считаем пиксели
        mask_pixels = cv2.countNonZero(mask)
        inter_pixels = cv2.countNonZero(intersection)

        if mask_pixels > 0:
            return inter_pixels / mask_pixels
        return 0

    def detect_walls(self, processed_img):
        """Детекция стен"""

        binary_for_thickness = processed_img.copy()

        # Детекция линий
        lines = cv2.HoughLinesP(
            processed_img,
            rho=1,
            theta=np.pi / 180,
            threshold=40,  # Увеличен порог
            minLineLength=self.min_wall_length,
            maxLineGap=20
        )

        walls = []

        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]

                # Рассчитываем длину
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Оцениваем толщину
                thickness_score = self.estimate_line_thickness(x1, y1, x2, y2, binary_for_thickness)

                # ПРАВИЛО 1: Фильтрация по длине
                if length < self.min_wall_length:
                    # Короткие линии принимаем только если очень толстые
                    if length >= 30 and thickness_score > 0.8:
                        pass  # Принимаем
                    else:
                        continue

                # ПРАВИЛО 2: Фильтрация тонких линий
                if length > 100 and thickness_score < 0.5:
                    continue  # Длинные и тонкие - не стены

                if 50 <= length <= 100 and thickness_score < 0.4:
                    continue  # Средние и тонкие

                # ПРАВИЛО 3: Очень тонкие линии всегда отфильтровываем
                if thickness_score < 0.3:
                    continue

                # Рассчитываем угол
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                if angle > 90:
                    angle -= 180

                # Сохраняем стену
                wall = {
                    'id': f'w{len(walls) + 1}',
                    'points': [[int(x1), int(y1)], [int(x2), int(y2)]],
                    'length': float(length),
                    'angle': float(angle),
                    'thickness': float(thickness_score)
                }
                walls.append(wall)

        # Объединяем стены
        merged_walls = self._merge_walls(walls)

        # Финальная фильтрация
        filtered_walls = []
        for wall in merged_walls:
            length = wall['length']
            thickness = wall['thickness']

            # Адаптивные правила по длине
            if length < 50 and thickness < 0.7:
                continue
            elif 50 <= length < 100 and thickness < 0.5:
                continue
            elif length >= 100 and thickness < 0.4:
                continue

            filtered_walls.append(wall)

        return filtered_walls

    def _merge_walls(self, walls):
        """Объединение близких стен"""
        if not walls:
            return []

        merged = []
        used = set()

        for i, wall1 in enumerate(walls):
            if i in used:
                continue

            points1 = wall1['points']
            angle1 = wall1['angle']

            to_merge = [i]

            for j, wall2 in enumerate(walls[i + 1:], i + 1):
                if j in used:
                    continue

                points2 = wall2['points']
                angle2 = wall2['angle']

                # Проверяем коллинеарность
                angle_diff = abs(angle1 - angle2)
                if angle_diff > 10 and angle_diff < 170:
                    continue

                # Проверяем близость концов
                x11, y11 = points1[0]
                x12, y12 = points1[1]
                x21, y21 = points2[0]
                x22, y22 = points2[1]

                dist1 = np.sqrt((x11 - x21) ** 2 + (y11 - y21) ** 2)
                dist2 = np.sqrt((x11 - x22) ** 2 + (y11 - y22) ** 2)
                dist3 = np.sqrt((x12 - x21) ** 2 + (y12 - y21) ** 2)
                dist4 = np.sqrt((x12 - x22) ** 2 + (y12 - y22) ** 2)

                if min(dist1, dist2, dist3, dist4) < self.merge_distance:
                    to_merge.append(j)

            # Объединяем
            if len(to_merge) == 1:
                merged.append(walls[i])
            else:
                all_points = []
                all_thickness = []
                for idx in to_merge:
                    all_points.extend(walls[idx]['points'])
                    all_thickness.append(walls[idx]['thickness'])

                all_points = np.array(all_points)
                x_coords = all_points[:, 0]
                y_coords = all_points[:, 1]

                if abs(angle1) < 45 or abs(angle1) > 135:
                    x1, x2 = np.min(x_coords), np.max(x_coords)
                    y_avg = np.mean(y_coords)
                    merged_points = [[int(x1), int(y_avg)], [int(x2), int(y_avg)]]
                else:
                    y1, y2 = np.min(y_coords), np.max(y_coords)
                    x_avg = np.mean(x_coords)
                    merged_points = [[int(x_avg), int(y1)], [int(x_avg), int(y2)]]

                avg_thickness = np.mean(all_thickness)

                merged_wall = {
                    'id': f'w{len(merged) + 1}',
                    'points': merged_points,
                    'length': float(np.sqrt((merged_points[1][0] - merged_points[0][0]) ** 2 +
                                            (merged_points[1][1] - merged_points[0][1]) ** 2)),
                    'angle': float(angle1),
                    'thickness': float(avg_thickness)
                }
                merged.append(merged_wall)

            used.update(to_merge)

        return merged

    def visualize_walls(self, image_path, walls, output_path):
        """Визуализация стен - все стены синим цветом"""

        img = cv2.imread(image_path)

        # Один цвет для всех стен - синий
        wall_color = (255, 0, 0)  # BGR: синий цвет

        for i, wall in enumerate(walls):
            points = wall['points']

            # Толщина линии зависит от длины стены
            length = wall['length']
            if length > 150:
                line_thickness = 4
            elif length > 80:
                line_thickness = 3
            else:
                line_thickness = 2

            # Рисуем стену синим цветом
            cv2.line(img,
                     tuple(points[0]),
                     tuple(points[1]),
                     wall_color, line_thickness, cv2.LINE_AA)

            # Можно раскомментировать для отображения ID стен
            # mid_x = (points[0][0] + points[1][0]) // 2
            # mid_y = (points[0][1] + points[1][1]) // 2
            # cv2.putText(img, wall['id'], (mid_x - 10, mid_y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Статистика
        stats = f"Найдено стен: {len(walls)}"
        cv2.putText(img, stats, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(img, stats, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite(output_path, img)
        print(f"Визуализация сохранена: {output_path}")

        return img

    def process_image(self, image_path):
        """Основной пайплайн"""

        print(f"\n" + "=" * 50)
        print(f"Обработка: {os.path.basename(image_path)}")
        print("=" * 50)

        # 1. Загрузка
        img = self.load_image(image_path)

        # 2. Предобработка
        processed = self.preprocess_image(img)

        # 3. Детекция
        walls = self.detect_walls(processed)

        # 4. Результат
        result = {
            "meta": {
                "source": os.path.basename(image_path),
                "processing_date": datetime.now().isoformat(),
                "image_size": {
                    "width": int(self.original.shape[1]),
                    "height": int(self.original.shape[0])
                }
            },
            "detection_params": {
                "min_wall_length": self.min_wall_length,
                "merge_distance": self.merge_distance,
                "min_thickness": self.min_thickness
            },
            "statistics": {
                "total_walls": len(walls),
                "avg_wall_length": float(np.mean([w['length'] for w in walls])) if walls else 0,
                "avg_thickness": float(np.mean([w['thickness'] for w in walls])) if walls else 0
            },
            "walls": [
                {
                    "id": wall['id'],
                    "points": wall['points'],
                    "length": wall['length'],
                    "thickness": wall['thickness']
                }
                for wall in walls
            ]
        }

        return result, walls

    def save_to_json(self, data, output_path):
        """Сохранение в JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"JSON сохранен: {output_path}")


def main():
    """Основная функция"""

    # НАСТРОЙКИ (можно менять здесь):
    detector = WallDetector(
        min_wall_length=50,  # Минимальная длина стены
        merge_distance=15,  # Расстояние для объединения
        min_thickness=0.5  # Минимальная толщина
    )

    # Пути
    input_dir = "data/input"
    output_dir = "data/output"

    os.makedirs(output_dir, exist_ok=True)

    # Обработка
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                input_path = os.path.join(input_dir, filename)
                base_name = os.path.splitext(filename)[0]

                result, walls = detector.process_image(input_path)

                json_path = os.path.join(output_dir, f"{base_name}.json")
                detector.save_to_json(result, json_path)

                viz_path = os.path.join(output_dir, f"{base_name}_detected.png")
                detector.visualize_walls(input_path, walls, viz_path)

                print(f"Обработка завершена: {filename}")
                print(f"  Найдено стен: {len(walls)}")
                if walls:
                    print(f"  Средняя толщина: {result['statistics']['avg_thickness']:.2f}")
                print(f"  JSON: {os.path.basename(json_path)}")
                print(f"  Визуализация: {os.path.basename(viz_path)}")
                print()

            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")


if __name__ == "__main__":
    main()