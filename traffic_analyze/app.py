from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import psycopg2
from dotenv import load_dotenv
import os
from celery import Celery
from video_processor import VideoProcessor
import shutil
from fastapi import Form

load_dotenv()

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешите фронтенд
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Конфигурация Celery
celery = Celery('tasks', broker=os.getenv('REDIS_URL'), backend=os.getenv('REDIS_URL'))

# Конфигурация БД
db_config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Модель для создания дрона
class DroneCreate(BaseModel):
    drone_name: str
    description: str

# Celery задача для обработки видео
# @celery.task
# def process_video_task(source_path: str, target_path: str, drone_id: int):
#     processor = VideoProcessor(
#         source_weights_path="model/traffic_analysis.pt",
#         source_video_path=source_path,
#         target_video_path=target_path,
#         drone_id=drone_id,
#         db_config=db_config
#     )
#     processor.process_video()

@app.get("/drones")
async def get_drones():
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT drone_id, drone_name, description FROM drones")
    drones = [{"drone_id": row[0], "drone_name": row[1], "description": row[2]} for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return drones

@app.post("/drones")
async def create_drone(drone_name: str, description: str):
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO drones (drone_name, description) VALUES (%s, %s) RETURNING drone_id",
            (drone_name, description)
        )
        drone_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        return {"drone_id": drone_id}
    except psycopg2.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
    except psycopg2.Error as e:
        raise HTTPException(status_code=400, detail=f"Database error: {str(e)}")

# @app.post("/upload")
# async def upload_video(drone_id: int = Form(...), video: UploadFile = File(...)):
#     if not video.filename.endswith(('.mp4', '.mov')):
#         raise HTTPException(status_code=400, detail="Only .mp4 or .mov files are allowed")
    

#     print("видос есть")
#     os.makedirs("uploads", exist_ok=True)
#     os.makedirs("results", exist_ok=True)
#     source_path = f"uploads/{video.filename}"
#     target_path = f"results/processed_{video.filename}"
    
#     with open(source_path, "wb") as f:
#         shutil.copyfileobj(video.file, f)
    
#     # Запуск асинхронной обработки
#     process_video_task.delay(source_path, target_path, drone_id)
#     return {"message": "Video uploaded and processing started", "output_path": target_path}

@app.post("/upload")
async def upload_video(
    drone_id: int = Form(...),
    file: UploadFile = File(...)
):
    try:
        # conn = psycopg2.connect(**db_config)
        # cursor = conn.cursor()
        
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        print("Файл получен:", file_path)
        
        # Обязательно формируем путь к выходному видео заранее
        processed_path = f"results/processed_{file.filename}"

        processor = VideoProcessor(
            source_weights_path="../model/traffic_analysis.pt",
            source_video_path=file_path,
            target_video_path=processed_path,
            drone_id=drone_id,
            db_config=db_config
        )
        mission_id = processor.process_video()  # здесь не важно, что он возвращает, мы путь уже знаем
        print("Обработан файл:", processed_path)
        
        # ⬇️ Сохраняем путь к обработанному файлу
        # cursor.execute(
        #     "INSERT INTO missions (drone_id, video_path, fps) VALUES (%s, %s, %s) RETURNING mission_id",
        #     (drone_id, processed_path, 29)
        # )
        # mission_id = cursor.fetchone()[0]
        # conn.commit()
        # cursor.close()
        # conn.close()
        
        return {
            "message": "Video processed",
            "output_path": processed_path,
            "mission_id": mission_id
        }
    except psycopg2.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


import subprocess



@app.get("/processed-video/{mission_id}")
async def get_processed_video(mission_id: int):
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT video_path FROM missions WHERE mission_id = %s", (mission_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Mission not found")
        
        video_path = result[0]
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Processed video not found")
        
        result_path = f"recoded_videos/{os.path.basename(video_path)}"

        # перекодировка для браузера
        subprocess.run([
            "ffmpeg", "-y",  # -y чтобы перезаписать, если уже существует
            "-i", video_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-movflags", "faststart",
            result_path
        ], check=True)
        print(f"🟡 Trying to serve file: {result_path}")
        print("🟡 Absolute path:", os.path.abspath(result_path))
        print("🟡 Exists?", os.path.exists(result_path))
        print("Отдаем:", result_path)
   
        # second = f"fixed/trim.mp4"
        # fix_video_format(video_path, second)
        return FileResponse(result_path, media_type="video/mp4", filename=os.path.basename(result_path))
    except psycopg2.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
# import subprocess

# def fix_video_format(input_path: str, output_path: str):
#     command = [
#         'ffmpeg',
#         '-i', input_path,
#         '-c:v', 'libx264',
#         '-c:a', 'aac',
#         '-movflags', '+faststart',
#         output_path
#     ]
#     subprocess.run(command, check=True)


@app.get("/db")
async def get_db_data(table: str, limit: int = 100):
    valid_tables = ["drones", "missions", "tracked_objects"]
    if table not in valid_tables:
        raise HTTPException(status_code=400, detail="Invalid table name")
    
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table} LIMIT %s", (limit,))
    columns = [desc[0] for desc in cursor.description]
    data = [dict(zip(columns, row)) for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return data

@app.get("/download/{filename}")
async def download_result(filename: str):
    file_path = f"results/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)