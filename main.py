from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Depends, Query, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import uuid
import time
import logging
import json
import shutil
import redis
import threading
import concurrent.futures
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Импорт из локальной копии ppee_analyzer
from ppee_analyzer.vector_store import QdrantManager, OllamaEmbeddings, BGEReranker
from ppee_analyzer.document_processor import DoclingPDFConverter, PPEEDocumentSplitter
from ppee_analyzer.checklist import ChecklistAnalyzer
from qdrant_client.http import models

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ppee-api.log")
    ]
)
logger = logging.getLogger("ppee-api")

# Инициализация FastAPI
app = FastAPI(
    title="PPEE Analyzer API",
    description="API для анализа документов ППЭЭ с использованием семантического поиска и LLM",
    version="1.0.0"
)

# Добавление CORS middleware для возможности обращения из веб-интерфейса
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене лучше указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация
UPLOAD_DIR = os.environ.get('UPLOAD_DIR', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Инициализация Redis
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', None)
TASK_KEY_PREFIX = "ppee:task:"
TASK_TTL = 60 * 60 * 24 * 7  # 7 дней

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True  # Автоматическое декодирование ответов
    )
    # Проверка подключения
    redis_client.ping()
    logger.info(f"Успешное подключение к Redis: {REDIS_HOST}:{REDIS_PORT}, DB: {REDIS_DB}")
except Exception as e:
    logger.error(f"Ошибка подключения к Redis: {str(e)}")
    redis_client = None

# Создаем пул потоков для выполнения блокирующих операций
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)


# Модели данных
class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, progress, complete, error
    progress: Optional[int] = 0
    message: Optional[str] = None
    stage: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SearchRequest(BaseModel):
    application_id: str
    query: str
    limit: Optional[int] = Field(5, ge=1, le=50)
    use_reranker: Optional[bool] = False
    use_smart_search: Optional[bool] = False
    vector_weight: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    text_weight: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    hybrid_threshold: Optional[int] = Field(10, ge=1, le=100)
    rerank_limit: Optional[int] = None


class AnalyzeRequest(BaseModel):
    application_id: str
    checklist_path: str
    limit: Optional[int] = Field(3, ge=1, le=20)


# Функции для работы с задачами
def update_task_status(
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        stage: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
) -> Dict[str, Any]:
    """Обновляет статус задачи в Redis"""
    # Создаем ключ Redis
    task_key = f"{TASK_KEY_PREFIX}{task_id}"

    # Получаем текущий статус
    task_info = {}

    if redis_client:
        task_json = redis_client.get(task_key)
        if task_json:
            try:
                task_info = json.loads(task_json)
            except json.JSONDecodeError:
                logger.warning(f"Невозможно декодировать JSON для задачи {task_id}")

    # Создаем базовую структуру, если задача новая
    if not task_info:
        task_info = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "Задача создана",
            "stage": None,
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat()
        }

    # Обновляем информацию о задаче
    task_info["updated_at"] = datetime.now().isoformat()

    if status is not None:
        task_info["status"] = status
    if progress is not None:
        task_info["progress"] = progress
    if message is not None:
        task_info["message"] = message
    if stage is not None:
        task_info["stage"] = stage
    if result is not None:
        task_info["result"] = result
    if error is not None:
        task_info["error"] = error

    # Сохраняем в Redis
    if redis_client:
        try:
            redis_client.set(task_key, json.dumps(task_info))
            # Устанавливаем TTL для автоматической очистки
            redis_client.expire(task_key, TASK_TTL)
        except Exception as e:
            logger.error(f"Ошибка при сохранении статуса задачи в Redis: {str(e)}")

    return task_info


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Возвращает текущий статус задачи из Redis"""
    task_key = f"{TASK_KEY_PREFIX}{task_id}"

    # Если Redis не доступен, выбрасываем исключение
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis недоступен, невозможно получить статус задачи")

    task_json = redis_client.get(task_key)

    if not task_json:
        raise HTTPException(status_code=404, detail=f"Задача с ID {task_id} не найдена")

    try:
        task_info = json.loads(task_json)
        return task_info
    except json.JSONDecodeError:
        logger.error(f"Невозможно декодировать JSON для задачи {task_id}")
        raise HTTPException(status_code=500, detail=f"Ошибка при чтении статуса задачи")


# Зависимости
def get_qdrant_manager():
    """Создает и возвращает экземпляр QdrantManager"""
    try:
        return QdrantManager(
            host=os.environ.get('QDRANT_HOST', 'localhost'),
            port=int(os.environ.get('QDRANT_PORT', 6333)),
            collection_name=os.environ.get('QDRANT_COLLECTION', 'ppee_applications'),
            embeddings_type="ollama",
            model_name="bge-m3",
            ollama_url=os.environ.get('OLLAMA_URL', 'http://localhost:11434')
        )
    except Exception as e:
        logger.error(f"Ошибка при создании QdrantManager: {str(e)}")
        raise


def verify_api_key(x_api_key: str = Header(None)):
    """Проверяет API ключ, если он установлен в конфигурации"""
    api_key = os.environ.get('API_KEY')

    # Если API ключ не настроен, пропускаем проверку
    if not api_key:
        return True

    # Если ключ настроен, но не предоставлен в запросе
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Отсутствует API ключ (X-API-Key в заголовке)"
        )

    # Проверяем ключ
    if x_api_key != api_key:
        raise HTTPException(
            status_code=403,
            detail="Недействительный API ключ"
        )

    return True


# Базовые эндпоинты
@app.get("/")
async def root():
    """Проверка работоспособности API"""
    return {
        "message": "PPEE Analyzer API работает",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/health")
async def health_check():
    """Расширенная проверка работоспособности API и связанных компонентов"""
    health_status = {
        "status": "ok",
        "api_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }

    # Проверка Redis
    try:
        if redis_client:
            redis_client.ping()
            health_status["components"]["redis"] = {
                "status": "ok",
                "host": f"{REDIS_HOST}:{REDIS_PORT}",
                "db": REDIS_DB
            }
        else:
            health_status["status"] = "degraded"
            health_status["components"]["redis"] = {
                "status": "error",
                "error": "Redis клиент не инициализирован"
            }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["redis"] = {
            "status": "error",
            "error": str(e)
        }

    # Проверка Qdrant
    try:
        qdrant_manager = get_qdrant_manager()
        # Проверяем доступность коллекции
        collections = qdrant_manager.client.get_collections().collections
        health_status["components"]["qdrant"] = {
            "status": "ok",
            "collections": [c.name for c in collections]
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["qdrant"] = {
            "status": "error",
            "error": str(e)
        }

    # Проверка Ollama (опционально)
    try:
        from ppee_analyzer.vector_store.ollama_embeddings import OllamaEmbeddings
        ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
        ollama = OllamaEmbeddings(
            model_name="bge-m3",
            base_url=ollama_url,
            check_availability=True
        )
        health_status["components"]["ollama"] = {
            "status": "ok",
            "url": ollama_url
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["ollama"] = {
            "status": "error",
            "error": str(e)
        }

    return health_status


# Реализация функции для синхронной обработки индексации
def process_document_indexing_sync(
        task_id: str,
        application_id: str,
        document_path: str,
        delete_existing: bool
):
    """Обрабатывает индексацию документа в синхронном режиме"""
    start_time = time.time()

    try:
        # Обновляем статус - начало обработки
        update_task_status(
            task_id=task_id,
            status="progress",
            progress=10,
            message="Начало индексации документа",
            stage="prepare"
        )

        # Проверяем существование файла
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Файл не найден: {document_path}")

        # Проверяем права доступа
        if not os.access(document_path, os.R_OK):
            raise PermissionError(f"Нет прав на чтение файла: {document_path}")

        # Определяем тип файла
        is_pdf = document_path.lower().endswith('.pdf')
        processing_path = document_path

        # Проверяем размер файла для предупреждения о больших PDF
        file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
        if is_pdf and file_size_mb > 50:  # Если файл больше 50 МБ
            logger.warning(f"Большой PDF ({file_size_mb:.1f} МБ). Семантическое разделение может занять много времени.")

            # Обновляем статус с предупреждением
            update_task_status(
                task_id=task_id,
                message=f"Предупреждение: большой PDF ({file_size_mb:.1f} МБ). Обработка может занять продолжительное время."
            )

        # Конвертация PDF в Markdown
        if is_pdf:
            update_task_status(
                task_id=task_id,
                status="progress",
                progress=20,
                message="Конвертация PDF документа в Markdown",
                stage="convert"
            )

            try:
                converter = DoclingPDFConverter()
                md_path = os.path.splitext(document_path)[0] + ".md"
                converter.convert_pdf_to_markdown(document_path, md_path)

                if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
                    processing_path = md_path
                    logger.info(f"PDF успешно конвертирован в Markdown: {md_path}")
                else:
                    logger.warning(f"Конвертация PDF не удалась, используем исходный файл: {document_path}")
            except Exception as e:
                logger.error(f"Ошибка при конвертации PDF: {str(e)}")
                # Записываем предупреждение, но не используем стандартный сплиттер
                update_task_status(
                    task_id=task_id,
                    message=f"Предупреждение: не удалось конвертировать PDF, будет обработан исходный файл. Ошибка: {str(e)}",
                    # Не меняем статус и stage
                )

        # Обновляем статус - разделение документа
        update_task_status(
            task_id=task_id,
            status="progress",
            progress=40,
            message="Разделение документа на смысловые фрагменты",
            stage="split"
        )

        # Создаем семантический сплиттер
        try:
            # Импортируем семантический сплиттер
            from ppee_analyzer.semantic_chunker import SemanticDocumentSplitter

            # Определяем, использовать ли GPU
            use_gpu = os.environ.get('USE_GPU_FOR_CHUNKING', '1') == '1'

            # Получаем настройки чанков из переменных окружения
            chunk_size = int(os.environ.get('CHUNK_SIZE', 1500))
            chunk_overlap = int(os.environ.get('CHUNK_OVERLAP', 150))

            # Создаем экземпляр семантического сплиттера с заданными параметрами
            splitter = SemanticDocumentSplitter(
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            logger.info(
                f"Семантический сплиттер инициализирован (GPU: {use_gpu}, chunk_size: {chunk_size}, overlap: {chunk_overlap})")

            # Разделяем документ на фрагменты
            chunks = splitter.load_and_process_file(processing_path, application_id)

            # Логирование распределения типов контента
            content_types_distribution = {}
            for chunk in chunks:
                content_type = chunk.metadata.get("content_type", "unknown")
                content_types_distribution[content_type] = content_types_distribution.get(content_type, 0) + 1

            logger.info(f"Документ разделен на {len(chunks)} семантических фрагментов")
            logger.info(f"Распределение типов контента: {content_types_distribution}")

        except ImportError as e:
            logger.error(f"Критическая ошибка: модуль semantic_chunker не установлен: {str(e)}")
            raise RuntimeError(f"Модуль semantic_chunker не установлен. Невозможно продолжить индексацию: {str(e)}")
        except Exception as e:
            logger.error(f"Критическая ошибка при семантическом разделении документа: {str(e)}")
            raise RuntimeError(f"Ошибка при семантическом разделении документа: {str(e)}")

        # Обновляем статус - индексация
        update_task_status(
            task_id=task_id,
            status="progress",
            progress=60,
            message=f"Индексация {len(chunks)} фрагментов",
            stage="index"
        )

        # Получаем экземпляр QdrantManager
        qdrant_manager = get_qdrant_manager()

        # Если нужно удалить существующие данные
        if delete_existing:
            try:
                deleted_count = qdrant_manager.delete_application(application_id)
                logger.info(f"Удалено {deleted_count} существующих документов для заявки {application_id}")

                if deleted_count > 0:
                    # Только если что-то было удалено, обновляем сообщение
                    update_task_status(
                        task_id=task_id,
                        message=f"Удалено {deleted_count} существующих фрагментов для заявки {application_id}"
                    )
            except Exception as e:
                logger.warning(f"Предупреждение: не удалось удалить существующие данные: {str(e)}")
                # Не обновляем статус, продолжаем работу

        # Собираем статистику по типам фрагментов
        content_types = {}
        for chunk in chunks:
            content_type = chunk.metadata.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1

        # Индексируем фрагменты
        batch_size = 32
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            batch = chunks[i:end_idx]

            # Добавляем фрагменты в индекс
            try:
                qdrant_manager.add_documents(batch)

                # Обновляем прогресс
                progress = 60 + int(35 * (end_idx / total_chunks))
                update_task_status(
                    task_id=task_id,
                    status="progress",
                    progress=progress,
                    message=f"Проиндексировано {end_idx}/{total_chunks} фрагментов",
                    stage="index"
                )
            except Exception as e:
                logger.error(f"Ошибка при индексации батча {i}-{end_idx}: {str(e)}")
                # Обновляем статус, но продолжаем с следующим батчем
                update_task_status(
                    task_id=task_id,
                    message=f"Предупреждение: ошибка при индексации батча {i}-{end_idx}: {str(e)}"
                )

            # Небольшая пауза для снижения нагрузки
            time.sleep(0.1)

        # Вычисляем время выполнения
        execution_time = time.time() - start_time

        # Обновляем статус - завершение
        update_task_status(
            task_id=task_id,
            status="complete",
            progress=100,
            message="Индексация завершена успешно",
            stage="complete",
            result={
                "application_id": application_id,
                "document_path": document_path,
                "processing_path": processing_path,
                "total_chunks": total_chunks,
                "content_types": content_types,
                "execution_time": f"{execution_time:.2f} сек",
                "status": "success"
            }
        )

        logger.info(f"Индексация для заявки {application_id} успешно завершена за {execution_time:.2f} сек")

    except Exception as e:
        execution_time = time.time() - start_time
        logger.exception(f"Ошибка при индексации документа: {str(e)}")

        # Добавляем детали ошибки
        error_details = {
            "error_type": type(e).__name__,
            "document_path": document_path,
            "application_id": application_id,
            "execution_time": f"{execution_time:.2f} сек",
            "timestamp": datetime.now().isoformat()
        }

        # Обновляем статус с ошибкой
        update_task_status(
            task_id=task_id,
            status="error",
            progress=0,
            message=f"Ошибка при индексации: {str(e)}",
            stage="error",
            error=str(e),
            result={
                "error_details": error_details,
                "status": "error"
            }
        )


# Эндпоинт для индексации документа - теперь использует пул потоков
@app.post("/api/index-document", response_model=TaskStatus, dependencies=[Depends(verify_api_key)])
async def index_document(
        file: UploadFile = File(...),
        application_id: str = Form(...),
        delete_existing: bool = Form(False)
):
    """
    Индексирует документ в векторной базе данных.

    - **file**: Загружаемый файл (PDF, DOCX, MD, TXT)
    - **application_id**: ID заявки
    - **delete_existing**: Удалить ли существующие данные заявки
    """
    # Генерируем уникальный ID задачи
    task_id = str(uuid.uuid4())

    # Создаем директорию для этой задачи
    task_dir = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    try:
        # Сохраняем загруженный файл
        file_path = os.path.join(task_dir, file.filename)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        # Устанавливаем права доступа
        os.chmod(file_path, 0o644)  # Права на чтение для всех

        logger.info(f"Файл {file.filename} для заявки {application_id} сохранен как {file_path}")

        # Создаем запись о задаче
        task_info = update_task_status(
            task_id=task_id,
            status="pending",
            progress=0,
            message="Задача индексации создана",
            stage="prepare"
        )

        # Запускаем задачу в отдельном потоке через ThreadPoolExecutor
        thread_pool.submit(
            process_document_indexing_sync,
            task_id=task_id,
            application_id=application_id,
            document_path=file_path,
            delete_existing=delete_existing
        )

        return TaskStatus(**task_info)

    except Exception as e:
        # Если произошла ошибка при создании задачи, очищаем
        logger.error(f"Ошибка при создании задачи индексации: {str(e)}")

        # Пытаемся удалить созданную директорию
        try:
            if os.path.exists(task_dir):
                shutil.rmtree(task_dir)
        except Exception as cleanup_error:
            logger.error(f"Ошибка при очистке директории задачи: {str(cleanup_error)}")

        # Возвращаем ошибку
        raise HTTPException(status_code=500, detail=f"Ошибка при создании задачи индексации: {str(e)}")


# Функция для синхронной обработки поиска
def process_search_sync(
        task_id: str,
        application_id: str,
        query: str,
        limit: int = 5,
        use_reranker: bool = False,
        use_smart_search: bool = False,
        vector_weight: float = 0.5,
        text_weight: float = 0.5,
        hybrid_threshold: int = 10,
        rerank_limit: Optional[int] = None
):
    """Выполняет семантический поиск в синхронном режиме"""
    start_time = time.time()

    try:
        # Обновляем статус
        update_task_status(
            task_id=task_id,
            status="progress",
            progress=30,
            message="Инициализация поиска",
            stage="initializing"
        )

        # Получаем экземпляр QdrantManager с настройкой ререйтинга
        qdrant_manager = QdrantManager(
            host=os.environ.get('QDRANT_HOST', 'localhost'),
            port=int(os.environ.get('QDRANT_PORT', 6333)),
            collection_name=os.environ.get('QDRANT_COLLECTION', 'ppee_applications'),
            embeddings_type="ollama",
            model_name="bge-m3",
            ollama_url=os.environ.get('OLLAMA_URL', 'http://localhost:11434'),
            use_reranker=use_reranker,
            reranker_model="BAAI/bge-reranker-v2-m3"
        )

        # Определяем метод поиска и обновляем статус
        search_method = "vector"

        if use_smart_search:
            if len(query) < hybrid_threshold:
                search_method = "hybrid"
                update_task_status(
                    task_id=task_id,
                    status="progress",
                    progress=40,
                    message="Выполнение гибридного поиска",
                    stage="hybrid_search"
                )
            else:
                update_task_status(
                    task_id=task_id,
                    status="progress",
                    progress=40,
                    message="Выполнение векторного поиска",
                    stage="vector_search"
                )
        else:
            update_task_status(
                task_id=task_id,
                status="progress",
                progress=40,
                message="Выполнение векторного поиска",
                stage="vector_search"
            )

        # Выполняем поиск в зависимости от выбранного метода
        if use_smart_search:
            # Умный поиск определит гибридный или векторный метод
            results = qdrant_manager.smart_search(
                application_id=application_id,
                query=query,
                limit=limit,
                use_reranker=use_reranker,
                rerank_limit=rerank_limit,
                vector_weight=vector_weight,
                text_weight=text_weight,
                hybrid_threshold=hybrid_threshold
            )
        elif search_method == "hybrid":
            # Явно выбран гибридный поиск
            results = qdrant_manager.hybrid_search(
                application_id=application_id,
                query=query,
                limit=limit,
                vector_weight=vector_weight,
                text_weight=text_weight,
                use_reranker=use_reranker
            )
        else:
            # Обычный векторный поиск
            results = qdrant_manager.search(
                application_id=application_id,
                query=query,
                limit=limit,
                use_reranker=use_reranker,
                rerank_limit=rerank_limit
            )

        # Если использовался ререйтинг, обновляем статус
        if use_reranker:
            update_task_status(
                task_id=task_id,
                status="progress",
                progress=70,
                message="Выполнение ререйтинга завершено",
                stage="reranking"
            )

        # Форматируем результаты для возврата
        formatted_results = []
        for i, result in enumerate(results):
            formatted_result = {
                'position': i + 1,
                'text': result.get('text', ''),
                'section': result.get('metadata', {}).get('section', 'Неизвестно'),
                'content_type': result.get('metadata', {}).get('content_type', 'Неизвестно'),
                'score': round(float(result.get('score', 0.0)), 4),
                'search_type': result.get('search_type', search_method)
            }

            # Добавляем оценку ререйтинга, если она есть
            if use_reranker and 'rerank_score' in result:
                formatted_result['rerank_score'] = round(float(result.get('rerank_score', 0.0)), 4)

            formatted_results.append(formatted_result)

        # Вычисляем время выполнения
        execution_time = time.time() - start_time

        # Завершаем задачу
        update_task_status(
            task_id=task_id,
            status="complete",
            progress=100,
            message="Поиск завершен успешно",
            stage="complete",
            result={
                'status': 'success',
                'count': len(formatted_results),
                'use_reranker': use_reranker,
                'use_smart_search': use_smart_search,
                'search_method': search_method,
                'execution_time': round(execution_time, 2),
                'results': formatted_results
            }
        )

        # Освобождаем ресурсы, если использовался ререйтинг
        if use_reranker:
            try:
                qdrant_manager.cleanup()
            except Exception as e:
                logger.warning(f"Ошибка при освобождении ресурсов ререйтинга: {str(e)}")

    except Exception as e:
        execution_time = time.time() - start_time
        logger.exception(f"Ошибка при выполнении поиска: {str(e)}")

        # Обновляем статус с ошибкой
        update_task_status(
            task_id=task_id,
            status="error",
            progress=0,
            message=f"Ошибка при выполнении поиска: {str(e)}",
            stage="error",
            error=str(e),
            result={
                "status": "error",
                "error_details": {
                    "error_type": type(e).__name__,
                    "application_id": application_id,
                    "query": query,
                    "execution_time": f"{execution_time:.2f} сек",
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

        # Освобождаем ресурсы даже в случае ошибки
        if use_reranker:
            try:
                qdrant_manager.cleanup()
            except:
                pass


# Эндпоинт для семантического поиска - использует пул потоков
@app.post("/api/search", response_model=Dict[str, Any], dependencies=[Depends(verify_api_key)])
async def search(
        request: SearchRequest
):
    """
    Выполняет семантический поиск в векторной базе данных.

    - **application_id**: ID заявки
    - **query**: Поисковый запрос
    - **limit**: Максимальное количество результатов
    - **use_reranker**: Использовать ли ререйтинг
    - **use_smart_search**: Использовать ли умный выбор метода поиска
    - **vector_weight**: Вес векторного поиска (для гибридного)
    - **text_weight**: Вес текстового поиска (для гибридного)
    - **hybrid_threshold**: Порог длины запроса для гибридного поиска
    - **rerank_limit**: Количество документов для ререйтинга
    """
    # Генерируем ID задачи
    task_id = str(uuid.uuid4())

    # Создаем запись о задаче
    task_info = update_task_status(
        task_id=task_id,
        status="pending",
        progress=0,
        message="Задача поиска создана",
        stage="initializing"
    )

    # Запускаем поиск в отдельном потоке через ThreadPoolExecutor
    thread_pool.submit(
        process_search_sync,
        task_id=task_id,
        application_id=request.application_id,
        query=request.query,
        limit=request.limit,
        use_reranker=request.use_reranker,
        use_smart_search=request.use_smart_search,
        vector_weight=request.vector_weight,
        text_weight=request.text_weight,
        hybrid_threshold=request.hybrid_threshold,
        rerank_limit=request.rerank_limit
    )

    return {"task_id": task_id, "status": "pending"}


# Функция для синхронной обработки анализа чек-листа
def process_checklist_analysis_sync(
        task_id: str,
        application_id: str,
        checklist_path: str,
        limit: int = 3
):
    """Выполняет анализ чек-листа в синхронном режиме"""
    start_time = time.time()

    try:
        # Обновляем статус
        update_task_status(
            task_id=task_id,
            status="progress",
            progress=10,
            message="Инициализация анализа",
            stage="prepare"
        )

        # Проверяем существование файла чек-листа
        if not os.path.exists(checklist_path):
            raise FileNotFoundError(f"Файл чек-листа не найден: {checklist_path}")

        # Получаем экземпляр QdrantManager
        qdrant_manager = get_qdrant_manager()

        # Создаем экземпляр ChecklistAnalyzer
        analyzer = ChecklistAnalyzer(qdrant_manager)

        # Обновляем статус
        update_task_status(
            task_id=task_id,
            status="progress",
            progress=20,
            message="Парсинг чек-листа",
            stage="analyze"
        )

        # Парсим чек-лист
        checklist_items = analyzer.parse_checklist(checklist_path)

        if not checklist_items:
            raise ValueError("Не удалось найти пункты чек-листа")

        # Обновляем статус
        update_task_status(
            task_id=task_id,
            status="progress",
            progress=40,
            message=f"Выполнение поиска по {len(checklist_items)} пунктам",
            stage="analyze"
        )

        # Выполняем поиск по пунктам чек-листа
        search_results = analyzer.search_checklist_items(
            checklist_items=checklist_items,
            application_id=application_id,
            limit=limit
        )

        # Обновляем статус
        update_task_status(
            task_id=task_id,
            status="progress",
            progress=80,
            message="Извлечение данных из результатов",
            stage="analyze"
        )

        # Извлекаем данные из результатов
        extracted_data = analyzer.extract_data_from_results(search_results)

        # Вычисляем время выполнения
        execution_time = time.time() - start_time

        # Формируем результаты
        results = {
            "application_id": application_id,
            "checklist_path": checklist_path,
            "timestamp": datetime.now().isoformat(),
            "execution_time": f"{execution_time:.2f} сек",
            "checklist_items": {item['id']: item['content'] for item in checklist_items},
            "extracted_data": extracted_data
        }

        # Обновляем статус
        update_task_status(
            task_id=task_id,
            status="complete",
            progress=100,
            message="Анализ завершен успешно",
            stage="complete",
            result=results
        )

        logger.info(f"Анализ по чек-листу для заявки {application_id} успешно завершен за {execution_time:.2f} сек")

    except Exception as e:
        execution_time = time.time() - start_time
        logger.exception(f"Ошибка при анализе чек-листа: {str(e)}")

        # Обновляем статус с ошибкой
        update_task_status(
            task_id=task_id,
            status="error",
            progress=0,
            message=f"Ошибка при анализе чек-листа: {str(e)}",
            stage="error",
            error=str(e),
            result={
                "status": "error",
                "error_details": {
                    "error_type": type(e).__name__,
                    "application_id": application_id,
                    "checklist_path": checklist_path,
                    "execution_time": f"{execution_time:.2f} сек",
                    "timestamp": datetime.now().isoformat()
                }
            }
        )


# Эндпоинт для анализа по чек-листу
@app.post("/api/analyze-checklist", response_model=Dict[str, Any], dependencies=[Depends(verify_api_key)])
async def analyze_checklist(
        request: AnalyzeRequest
):
    """
    Анализирует заявку по чек-листу.

    - **application_id**: ID заявки
    - **checklist_path**: Путь к файлу чек-листа
    - **limit**: Максимальное количество результатов для каждого запроса
    """
    # Генерируем ID задачи
    task_id = str(uuid.uuid4())

    # Создаем запись о задаче
    task_info = update_task_status(
        task_id=task_id,
        status="pending",
        progress=0,
        message="Задача анализа создана",
        stage="prepare"
    )

    # Запускаем анализ в отдельном потоке
    thread_pool.submit(
        process_checklist_analysis_sync,
        task_id=task_id,
        application_id=request.application_id,
        checklist_path=request.checklist_path,
        limit=request.limit
    )

    return {"task_id": task_id, "status": "pending"}


# Эндпоинт для проверки статуса задачи
@app.get("/api/status/{task_id}", response_model=TaskStatus, dependencies=[Depends(verify_api_key)])
async def check_task_status(task_id: str):
    """
    Возвращает текущий статус задачи по ID.

    - **task_id**: ID задачи
    """
    try:
        task_info = get_task_status(task_id)

        # Преобразуем в TaskStatus модель
        return TaskStatus(
            task_id=task_id,
            status=task_info.get("status", "unknown"),
            progress=task_info.get("progress", 0),
            message=task_info.get("message", "Статус не определен"),
            stage=task_info.get("stage", None),
            result=task_info.get("result", None),
            error=task_info.get("error", None),
            created_at=task_info.get("created_at"),
            updated_at=task_info.get("updated_at")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении статуса задачи {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении статуса задачи: {str(e)}")


# Эндпоинт для получения списка задач
@app.get("/api/tasks", response_model=Dict[str, Any], dependencies=[Depends(verify_api_key)])
async def list_tasks(status: Optional[str] = None, limit: int = 100, offset: int = 0):
    """
    Возвращает список всех задач с возможностью фильтрации по статусу.

    - **status**: Фильтр по статусу (pending, progress, complete, error)
    - **limit**: Максимальное количество результатов
    - **offset**: Смещение для пагинации
    """
    try:
        # Если Redis не доступен, выбрасываем исключение
        if not redis_client:
            raise HTTPException(status_code=503, detail="Redis недоступен, невозможно получить список задач")

        # Получаем все ключи задач
        pattern = f"{TASK_KEY_PREFIX}*"
        all_keys = redis_client.keys(pattern)

        # Сортируем ключи для консистентной пагинации
        all_keys.sort(reverse=True)  # Самые новые первыми

        tasks = []
        filtered_count = 0

        # Обрабатываем каждый ключ
        for key in all_keys:
            # Извлекаем ID задачи из ключа
            task_id = key.replace(TASK_KEY_PREFIX, "")

            # Получаем информацию о задаче
            task_json = redis_client.get(key)
            if not task_json:
                continue

            try:
                task_info = json.loads(task_json)

                # Применяем фильтр по статусу
                if status and task_info.get("status") != status:
                    continue

                filtered_count += 1

                # Применяем пагинацию
                if filtered_count > offset and len(tasks) < limit:
                    # Добавляем только необходимые поля для списка
                    tasks.append({
                        "task_id": task_id,
                        "status": task_info.get("status"),
                        "progress": task_info.get("progress"),
                        "message": task_info.get("message"),
                        "stage": task_info.get("stage"),
                        "created_at": task_info.get("created_at"),
                        "updated_at": task_info.get("updated_at")
                    })
            except json.JSONDecodeError:
                logger.warning(f"Невозможно декодировать JSON для задачи {task_id}")

        return {
            "total": len(all_keys),
            "filtered": filtered_count,
            "offset": offset,
            "limit": limit,
            "tasks": tasks
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении списка задач: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении списка задач: {str(e)}")


# Эндпоинт для очистки старых задач
@app.get("/api/maintenance/cleanup", response_model=Dict[str, Any], dependencies=[Depends(verify_api_key)])
async def cleanup_tasks(
        age_days: int = Query(7, ge=1, le=30, description="Возраст задач для очистки в днях"),
        status_filter: str = Query("complete,error", description="Статусы для очистки (через запятую)")
):
    """
    Очищает старые задачи и связанные файлы.

    - **age_days**: Возраст задач для очистки в днях
    - **status_filter**: Статусы для очистки (через запятую)
    """
    try:
        # Если Redis не доступен, выбрасываем исключение
        if not redis_client:
            raise HTTPException(status_code=503, detail="Redis недоступен, невозможно выполнить очистку")

        # Разбираем статусы для фильтрации
        allowed_statuses = [s.strip() for s in status_filter.split(",")]

        # Расчет минимальной даты обновления для очистки
        min_date = datetime.now() - timedelta(days=age_days)
        min_date_str = min_date.isoformat()

        # Получаем все ключи задач
        pattern = f"{TASK_KEY_PREFIX}*"
        task_keys = redis_client.keys(pattern)

        cleaned_count = 0
        files_cleaned = 0

        # Обрабатываем каждый ключ
        for key in task_keys:
            # Извлекаем ID задачи из ключа
            task_id = key.replace(TASK_KEY_PREFIX, "")

            # Получаем информацию о задаче
            task_json = redis_client.get(key)
            if not task_json:
                continue

            try:
                task_info = json.loads(task_json)

                # Проверяем статус и время последнего обновления
                status = task_info.get("status")
                updated_at = task_info.get("updated_at")

                # Проверяем условия очистки
                if (status in allowed_statuses and updated_at and
                        updated_at < min_date_str):
                    # Удаляем файлы
                    task_dir = os.path.join(UPLOAD_DIR, task_id)
                    if os.path.exists(task_dir):
                        try:
                            shutil.rmtree(task_dir)
                            files_cleaned += 1
                        except Exception as e:
                            logger.warning(f"Не удалось удалить директорию {task_dir}: {str(e)}")

                    # Удаляем запись задачи
                    redis_client.delete(key)
                    cleaned_count += 1
            except json.JSONDecodeError:
                logger.warning(f"Невозможно декодировать JSON для задачи {task_id}")

        return {
            "status": "success",
            "cleaned_tasks": cleaned_count,
            "files_cleaned": files_cleaned,
            "total_tasks_scanned": len(task_keys),
            "age_days": age_days,
            "allowed_statuses": allowed_statuses
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при очистке задач: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при очистке задач: {str(e)}")


# Вспомогательные эндпоинты
@app.get("/api/applications", response_model=List[str], dependencies=[Depends(verify_api_key)])
async def get_applications():
    """Возвращает список ID заявок в базе данных"""
    try:
        qdrant_manager = get_qdrant_manager()
        application_ids = qdrant_manager.get_application_ids()
        return application_ids
    except Exception as e:
        logger.error(f"Ошибка при получении списка заявок: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении списка заявок: {str(e)}")


@app.delete("/api/applications/{application_id}", response_model=Dict[str, Any], dependencies=[Depends(verify_api_key)])
async def delete_application(application_id: str):
    """Удаляет данные заявки из векторной базы данных"""
    try:
        qdrant_manager = get_qdrant_manager()
        deleted = qdrant_manager.delete_application(application_id)
        return {
            "status": "success" if deleted else "error",
            "application_id": application_id,
            "message": f"Удалены данные заявки {application_id}" if deleted else "Не удалось удалить данные"
        }
    except Exception as e:
        logger.error(f"Ошибка при удалении данных заявки {application_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при удалении данных: {str(e)}")


# Новый эндпоинт: статус заявки
@app.get("/api/applications/{application_id}/stats", response_model=Dict[str, Any],
         dependencies=[Depends(verify_api_key)])
async def get_application_stats(application_id: str):
    """
    Возвращает статус и статистику заявки.

    - **application_id**: ID заявки
    """
    try:
        # Получаем экземпляр QdrantManager
        qdrant_manager = get_qdrant_manager()

        # Получаем статистику по заявке
        stats = qdrant_manager.get_stats(application_id)

        # Проверяем наличие данных
        has_data = stats.get("total_points", 0) > 0

        return {
            "application_id": application_id,
            "status": "indexed" if has_data else "not_indexed",
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Ошибка при получении статистики заявки {application_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении статистики: {str(e)}")


# Эндпоинт для просмотра чанков заявки
@app.get("/api/applications/{application_id}/chunks", response_model=Dict[str, Any],
         dependencies=[Depends(verify_api_key)])
async def get_application_chunks(
        application_id: str,
        limit: int = Query(500, ge=1, le=1000),
        cache: bool = Query(True, description="Использовать кэш Redis")
):
    """
    Возвращает чанки документов заявки.

    - **application_id**: ID заявки
    - **limit**: Максимальное количество возвращаемых чанков
    - **cache**: Использовать кэш Redis для ускорения
    """
    try:
        # Проверяем кэш, если включено кэширование
        if cache and redis_client:
            cache_key = f"ppee:chunks:{application_id}:{limit}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                try:
                    # Возвращаем кэшированные данные
                    return json.loads(cached_data)
                except json.JSONDecodeError:
                    logger.warning(f"Невозможно декодировать JSON из кэша для заявки {application_id}")
                    # Продолжаем с получением новых данных

        # Получаем экземпляр QdrantManager
        qdrant_manager = get_qdrant_manager()

        # Получаем статистику по заявке
        stats = qdrant_manager.get_stats(application_id)

        # Получаем все чанки заявки (с ограничением)
        response = qdrant_manager.client.scroll(
            collection_name=qdrant_manager.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.application_id",
                        match=models.MatchValue(value=application_id)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        # Преобразуем результаты в более удобный формат
        chunks = []
        page_numbers = set()
        sections = set()

        for point in response[0]:
            if "payload" in point.__dict__:
                # Получаем текст
                text = ""
                if "page_content" in point.payload:
                    text = point.payload["page_content"]

                # Получаем метаданные
                metadata = {}
                if "metadata" in point.payload:
                    metadata = point.payload["metadata"]

                    # Собираем статистику
                    if "page_number" in metadata and metadata["page_number"]:
                        page_numbers.add(metadata["page_number"])
                    if "section" in metadata and metadata["section"]:
                        sections.add(metadata["section"])

                # Добавляем в список
                chunk = {
                    "id": str(point.id),
                    "text": text,
                    "metadata": metadata
                }
                chunks.append(chunk)

        # Сортируем чанки по порядку (если есть chunk_index в метаданных)
        chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0) if x["metadata"] else 0)

        # Дополняем статистику
        if "pages" not in stats:
            stats["pages"] = sorted(list(page_numbers))
        if "sections" not in stats:
            stats["sections"] = sorted(list(sections))
            stats["sections_count"] = len(sections)

        result = {
            "application_id": application_id,
            "chunks_count": len(chunks),
            "stats": stats,
            "chunks": chunks,
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }

        # Сохраняем в кэш на 1 час, если включено кэширование
        if cache and redis_client:
            try:
                redis_client.set(cache_key, json.dumps(result), ex=3600)
            except Exception as e:
                logger.warning(f"Не удалось сохранить результаты в кэш: {str(e)}")

        return result

    except Exception as e:
        logger.error(f"Ошибка при получении чанков заявки {application_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении чанков: {str(e)}")


# Обработка ошибок
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception(f"Необработанная ошибка: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": f"Внутренняя ошибка сервера: {str(exc)}"}
    )


# Запуск приложения
if __name__ == "__main__":
    import uvicorn

    # Проверка наличия Redis при запуске
    if not redis_client:
        logger.warning("Redis недоступен! Некоторые функции будут работать некорректно.")

    uvicorn.run(app, host="0.0.0.0", port=8000)