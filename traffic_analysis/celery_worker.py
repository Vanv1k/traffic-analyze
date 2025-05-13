from celery import Celery
from dotenv import load_dotenv
import os

load_dotenv()

app = Celery('tasks', broker=os.getenv('redis://localhost:6379/0'), backend=os.getenv('redis://localhost:6379/0'))
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)