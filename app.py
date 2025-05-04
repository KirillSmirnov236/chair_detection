import streamlit as st
import cv2
import supervision as sv
import numpy as np
from PIL import Image
import base64
import json
import os
import datetime
from ultralytics import YOLO

# Загрузка предобученной модели
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolo_trained.pt')
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

model = load_model()

def analyze_image(image_path):
    if model is None:
        st.error("Модель не загружена!")
        return None
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image)

        #Получение результатов
        detections = sv.Detections.from_ultralytics(results[0])
        #Подсчет стульев и занятых мест
        total_chairs = len(detections)
        occupied_chairs = 0
        free_chairs = total_chairs - occupied_chairs
        if total_chairs == 0:
            occupied_chairs = 0
            free_chairs = 0

        #Аннотирование изображения
        box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
        labels = [
            f"{results[0].names[class_id]} {confidence:0.2f}"
            for confidence, class_id
            in zip(detections.confidence, detections.class_id)
        ]
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', annotated_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        image_base64 = "data:image/jpeg;base64," + image_base64

        results = {
            "total_chairs": total_chairs,
            "occupied_chairs": occupied_chairs,
            "free_chairs": free_chairs,
            "image": image_base64
        }
        return results
    except Exception as e:
        st.error(f"Ошибка при анализе изображения: {e}")
        return None

def analyze_video(video_path):
    if model is None:
        st.error("Модель не загружена!")
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Не удалось открыть видеофайл.")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        total_chairs = 0
        total_occupied = 0
        progress_bar = st.progress(0)
        for frame_number in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Анализ кадра
            results = model(frame)
            detections = sv.Detections.from_ultralytics(results[0])
            #Подсчет стульев и занятых мест
            chair_count = len(detections)
            occupied_count = 0
            total_chairs += chair_count
            total_occupied += occupied_count

            box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
            labels = [
                f"{results[0].names[class_id]} {confidence:0.2f}"
                for confidence, class_id
                in zip(detections.confidence, detections.class_id)
            ]
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR) # Преобразуем обратно в BGR

            _, buffer = cv2.imencode('.jpg', annotated_frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            image_base64 = "data:image/jpeg;base64," + image_base64
            frames.append({"frame_number": frame_number + 1, "image": image_base64})

            #Обновление прогресса
            progress = int((frame_number + 1) / total_frames * 100)
            progress_bar.progress(progress)

        average_chairs = total_chairs / total_frames if total_frames > 0 else 0
        average_occupied = total_occupied / total_frames if total_frames > 0 else 0
        cap.release()

        results = {
            "total_frames": total_frames,
            "average_chairs": average_chairs,
            "average_occupied": average_occupied,
            "frames": frames
        }
        return results

    except Exception as e:
        st.error(f"Ошибка при анализе видео: {e}")
        return None

def save_request_to_json(data):
    try:
        with open("history.json", "r+") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
            history.append(data)
            f.seek(0)
            json.dump(history, f, indent=4)
            f.truncate()
        st.success("Запрос сохранен в историю.")
    except Exception as e:
        st.error(f"Ошибка при сохранении запроса: {e}")

def load_history():
    try:
        with open("history.json", "r") as f:
            history = json.load(f)
            return history
    except FileNotFoundError:
        st.warning("История запросов пуста.")
        return []
    except json.JSONDecodeError:
        st.warning("Не удалось загрузить историю запросов.")
        return []

def generate_report(history):
    report_text = "Отчет об анализе заполненности зала:\n\n"
    for item in history:
        report_text += f"Дата/время: {item['timestamp']}\n"
        report_text += f"Тип: {item['type']}\n"
        if item['type'] == 'image':
            report_text += f"Всего стульев: {item['results']['total_chairs']}\n"
            report_text += f"Занятых стульев: {item['results']['occupied_chairs']}\n"
            report_text += f"Свободных стульев: {item['results']['free_chairs']}\n"
        elif item['type'] == 'video':
            report_text += f"Всего кадров: {item['results']['total_frames']}\n"
            report_text += f"Среднее количество стульев: {item['results']['average_chairs']}\n"
            report_text += f"Среднее количество занятых мест: {item['results']['average_occupied']}\n"
        report_text += "---\n"
    return report_text

st.title("Анализ заполненности зала")
source = st.radio("Выберите источник:", ("Изображение", "Видео"))

if source == "Изображение":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image_path = "temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        results = analyze_image(image_path)

        if results:
            st.image(results["image"], caption="Обработанное изображение", use_column_width=True)
            st.write("Всего стульев:", results["total_chairs"])
            st.write("Занятых стульев:", results["occupied_chairs"])
            st.write("Свободных стульев:", results["free_chairs"])
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = {
                "timestamp": timestamp,
                "type": "image",
                "results": results
            }
            save_request_to_json(data)
        os.remove(image_path)

elif source == "Видео":
    uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        video_bytes = uploaded_file.read()
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        results = analyze_video(video_path)

        if results:
            st.write("Всего кадров:", results["total_frames"])
            st.write("Среднее количество стульев:", results["average_chairs"])
            st.write("Среднее количество занятых мест:", results["average_occupied"])
            st.subheader("Обработанные кадры:")
            for frame in results["frames"]:
                st.image(frame["image"], caption=f"Кадр {frame['frame_number']}", use_column_width=True)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = {
                "timestamp": timestamp,
                "type": "video",
                "results": results
            }
            save_request_to_json(data)
        os.remove(video_path)

st.header("Отчет")
history = load_history()
if history:
    report_text = generate_report(history)
    st.text_area("Отчет:", value=report_text, height=300)
    st.download_button("Скачать отчет (EXEL)", data=report_text, file_name="report.xlsx", mime="text/plain")
else:
    st.write("История запросов пуста.")
