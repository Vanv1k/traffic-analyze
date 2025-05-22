from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import psycopg2
from dotenv import load_dotenv
import os
from celery import Celery
from video_processor import VideoProcessor
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Path, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional

load_dotenv()

app = FastAPI(
    title="Traffic Analysis API",
    description="API для анализа трафика с дронов",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

class HTTPErrorResponse(BaseModel):
    detail: str

# Документация моделей
class DroneResponse(BaseModel):
    drone_id: int
    drone_name: str
    description: str

class MissionResponse(BaseModel):
    mission_id: int
    drone_id: int
    video_path: str
    fps: int

class TrackedObjectResponse(BaseModel):
    object_id: int
    mission_id: int
    class_name: str
    confidence: float
    frame_number: int
    x: int
    y: int
    width: int
    height: int

class DroneCreate(BaseModel):
    drone_name: str
    description: str

@app.get("/drones", 
    response_model=List[DroneResponse],
    summary="Получить список всех дронов",
    responses={
        200: {"description": "Успешный запрос"},
        500: {"model": HTTPErrorResponse, "description": "Ошибка сервера"}
    })
async def get_drones():
    """Возвращает список всех зарегистрированных дронов"""
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT drone_id, drone_name, description FROM drones")
    drones = [{"drone_id": row[0], "drone_name": row[1], "description": row[2]} for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return drones

class DroneCreateRequest(BaseModel):
    drone_name: str
    description: str

@app.post("/drones", 
    response_model=Dict[str, int],
    summary="Добавить новый дрон",
    responses={
        200: {"description": "Дрон успешно создан"},
        400: {"model": HTTPErrorResponse, "description": "Неверные данные"},
        500: {"model": HTTPErrorResponse, "description": "Ошибка сервера"}
    })
async def create_drone(drone: DroneCreateRequest):
    """Создает новую запись дрона в базе данных"""
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO drones (drone_name, description) VALUES (%s, %s) RETURNING drone_id",
            (drone.drone_name, drone.description)
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

@app.post("/upload", 
    summary="Загрузить видео для обработки",
    responses={
        200: {
            "description": "Видео успешно загружено и обрабатывается",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Video processed",
                        "output_path": "results/processed_video.mp4",
                        "mission_id": 1
                    }
                }
            }
        },
        400: {"model": HTTPErrorResponse, "description": "Неверный формат файла"},
        500: {"model": HTTPErrorResponse, "description": "Ошибка обработки"}
    })
async def upload_video(
    drone_id: int = Form(..., description="ID дрона", example=1),
    file: UploadFile = File(..., description="Видеофайл (MP4, MOV)")
):
    """Загружает видео для последующего анализа трафика"""
    try:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        print("Файл получен:", file_path)
        
        processed_path = f"results/processed_{file.filename}"

        processor = VideoProcessor(
            source_weights_path="../model/traffic_analysis.pt",
            source_video_path=file_path,
            target_video_path=processed_path,
            drone_id=drone_id,
            db_config=db_config
        )
        mission_id = processor.process_video()  
        print("Обработан файл:", processed_path)
        
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



@app.api_route("/processed-video/{mission_id}", methods=["GET", "HEAD"], 
    summary="Получить обработанное видео",
    responses={
        200: {"description": "Видеофайл"},
        404: {"model": HTTPErrorResponse, "description": "Видео не найдено"},
        500: {"model": HTTPErrorResponse, "description": "Ошибка сервера"}
    })
async def get_processed_video(mission_id: int = Path(..., description="ID миссии", example=1)):
    """Возвращает обработанное видео по ID миссии"""
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
   

        return FileResponse(result_path, media_type="video/mp4", filename=os.path.basename(result_path))
    except psycopg2.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/db", 
    summary="Получить данные из таблицы",
    responses={
        200: {"description": "Данные из таблицы"},
        400: {"model": HTTPErrorResponse, "description": "Неверное имя таблицы"},
        500: {"model": HTTPErrorResponse, "description": "Ошибка сервера"}
    })
async def get_db_data(
    table: str = Query(..., description="Имя таблицы", example="missions"),
    limit: int = Query(100, description="Лимит записей", ge=1, le=1000)
):
    """Возвращает данные из указанной таблицы с ограничением количества записей"""
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

@app.get("/filters/class_names",
    response_model=List[str],
    summary="Получить уникальные классы объектов",
    description="Возвращает список всех уникальных классов обнаруженных объектов",
    tags=["Filters"],
    responses={
        200: {
            "description": "Список классов объектов",
            "content": {
                "application/json": {
                    "example": ["car", "truck", "person", "bicycle"]
                }
            }
        },
        500: {"model": HTTPErrorResponse, "description": "Ошибка сервера"}
    })
def get_class_names():
    """Возвращает список уникальных классов объектов из таблицы tracked_objects"""
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT class_name FROM tracked_objects")
    results = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return results

@app.get("/filters/mission_ids",
    response_model=List[int],
    summary="Получить ID всех миссий",
    description="Возвращает список всех ID миссий, отсортированных по возрастанию",
    tags=["Filters"],
    responses={
        200: {
            "description": "Список ID миссий",
            "content": {
                "application/json": {
                    "example": [1, 2, 3, 4, 5]
                }
            }
        },
        500: {"model": HTTPErrorResponse, "description": "Ошибка сервера"}
    })
def get_mission_ids():
    """Возвращает список всех ID миссий из таблицы missions"""
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT mission_id FROM missions ORDER BY mission_id")
    results = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return results

@app.get("/filters/drone_ids",
    response_model=List[Dict[str, Union[int, str]]],
    summary="Получить список дронов",
    description="Возвращает список всех дронов с их ID и названиями",
    tags=["Filters"],
    responses={
        200: {
            "description": "Список дронов",
            "content": {
                "application/json": {
                    "example": [
                        {"drone_id": 1, "drone_name": "DJI Mavic 3"},
                        {"drone_id": 2, "drone_name": "Autel EVO II"}
                    ]
                }
            }
        },
        500: {"model": HTTPErrorResponse, "description": "Ошибка сервера"}
    })
def get_drone_ids():
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT drone_id, drone_name FROM drones ORDER BY drone_id")
    results = [{"drone_id": row[0], "drone_name": row[1]} for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return results

@app.get("/download/{filename}",
    summary="Скачать результат обработки",
    description="Позволяет скачать обработанный видеофайл по имени",
    tags=["Download"],
    responses={
        200: {
            "description": "Видеофайл",
            "content": {"video/mp4": {}}
        },
        404: {
            "model": HTTPErrorResponse,
            "description": "Файл не найден",
            "content": {
                "application/json": {
                    "example": {"detail": "File not found"}
                }
            }
        },
        500: {"model": HTTPErrorResponse, "description": "Ошибка сервера"}
    })
async def download_result(
    filename: str = Path(..., description="Имя файла для скачивания", example="processed_video.mp4")
):
    """Возвращает файл по имени из директории results"""
    file_path = f"results/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)