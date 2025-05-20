from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import time
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Импорты из локальной копии ppee_analyzer
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

# Глобальное хранилище для статусов задач
tasks_store = {}


# Модели данных
class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, progress, complete, error
    progress: Optional[int] = 0
    message: Optional[str] = None
    stage: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SearchRequest(BaseModel):
    application_id: str
    query: str
    limit: Optional[int] = 5
    use_reranker: Optional[bool] = False
    use_smart_search: Optional[bool] = False
    vector_weight: Optional[float] = 0.5
    text_weight: Optional[float] = 0.5
    hybrid_threshold: Optional[int] = 10
    rerank_limit: Optional[int] = None


class AnalyzeRequest(BaseModel):
    application_id: str
    checklist_path: str
    limit: Optional[int] = 3


# Функции для работы с задачами
def update_task_status(
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        stage: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
):
    """Обновляет статус задачи"""
    if task_id not in tasks_store:
        tasks_store[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "Задача создана",
            "stage": None,
            "result": None,
            "error": None
        }

    if status is not None:
        tasks_store[task_id]["status"] = status
    if progress is not None:
        tasks_store[task_id]["progress"] = progress
    if message is not None:
        tasks_store[task_id]["message"] = message
    if stage is not None:
        tasks_store[task_id]["stage"] = stage
    if result is not None:
        tasks_store[task_id]["result"] = result
    if error is not None:
        tasks_store[task_id]["error"] = error


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Возвращает текущий статус задачи"""
    if task_id not in tasks_store:
        raise HTTPException(status_code=404, detail=f"Задача с ID {task_id} не найдена")
    return tasks_store[task_id]


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


# Эндпоинт для индексации документа
@app.post("/api/index-document", response_model=TaskStatus)
async def index_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        application_id: str = Form(...),
        delete_existing: bool = Form(False)
):
    """
    Индексирует документ в векторной базе данных.

    - **file**: Загружаемый файл (PDF, DOCX, MD, TXT)
    - **application_id**: ID заявки
    - **delete_existing**: Удалять ли существующие данные заявки
    """
    # Генерируем уникальный ID задачи
    task_id = str(uuid.uuid4())

    # Сохраняем загруженный файл
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    logger.info(f"Файл {file.filename} для заявки {application_id} сохранен как {file_path}")

    # Создаем запись о задаче
    update_task_status(
        task_id=task_id,
        status="pending",
        progress=0,
        message="Задача индексации создана",
        stage="prepare"
    )

    # Запускаем задачу в фоновом режиме
    background_tasks.add_task(
        process_document_indexing,
        task_id=task_id,
        application_id=application_id,
        document_path=file_path,
        delete_existing=delete_existing
    )

    return TaskStatus(**tasks_store[task_id])


# Функция для фоновой обработки индексации документа
async def process_document_indexing(
        task_id: str,
        application_id: str,
        document_path: str,
        delete_existing: bool
):
    """Обрабатывает индексацию документа в фоновом режиме"""
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

        # Определяем тип файла
        is_pdf = document_path.lower().endswith('.pdf')
        processing_path = document_path

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
                # В случае ошибки используем исходный файл

        # Обновляем статус - разделение документа
        update_task_status(
            task_id=task_id,
            status="progress",
            progress=40,
            message="Разделение документа на смысловые фрагменты",
            stage="split"
        )

        # Создаем сплиттер документов
        try:
            # Пробуем использовать семантический сплиттер
            from ppee_analyzer.semantic_chunker import SemanticDocumentSplitter
            splitter = SemanticDocumentSplitter(use_gpu=True)
            logger.info("Используем SemanticDocumentSplitter")
        except (ImportError, Exception) as e:
            logger.warning(f"Не удалось создать SemanticDocumentSplitter: {str(e)}")
            logger.info("Используем стандартный PPEEDocumentSplitter")
            splitter = PPEEDocumentSplitter()

        # Разделяем документ на фрагменты
        chunks = splitter.load_and_process_file(processing_path, application_id)
        logger.info(f"Документ разделен на {len(chunks)} фрагментов")

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
            deleted_count = qdrant_manager.delete_application(application_id)
            logger.info(f"Удалено {deleted_count} существующих документов для заявки {application_id}")

        # Индексируем фрагменты
        batch_size = 32
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            batch = chunks[i:end_idx]

            # Добавляем фрагменты в индекс
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

            # Небольшая пауза для снижения нагрузки
            await asyncio.sleep(0.1)

        # Собираем статистику по типам фрагментов
        content_types = {}
        for chunk in chunks:
            content_type = chunk.metadata.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1

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
                "status": "success"
            }
        )

        logger.info(f"Индексация для заявки {application_id} успешно завершена")

    except Exception as e:
        logger.exception(f"Ошибка при индексации документа: {str(e)}")

        # Обновляем статус с ошибкой
        update_task_status(
            task_id=task_id,
            status="error",
            progress=0,
            message=f"Ошибка при индексации: {str(e)}",
            stage="error",
            error=str(e)
        )


# Эндпоинт для проверки статуса задачи
@app.get("/api/status/{task_id}", response_model=TaskStatus)
async def check_task_status(task_id: str):
    """
    Возвращает текущий статус задачи по ID.

    - **task_id**: ID задачи
    """
    try:
        return TaskStatus(**get_task_status(task_id))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении статуса задачи {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении статуса задачи: {str(e)}")


# Эндпоинт для семантического поиска
@app.post("/api/search", response_model=Dict[str, Any])
async def search(
        background_tasks: BackgroundTasks,
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
    update_task_status(
        task_id=task_id,
        status="pending",
        progress=0,
        message="Задача поиска создана",
        stage="initializing"
    )

    # Запускаем поиск в фоновом режиме
    background_tasks.add_task(
        process_search,
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


# Функция для выполнения поиска в фоновом режиме
async def process_search(
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
    """Выполняет семантический поиск в фоновом режиме"""
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
                'execution_time': round(time.time() - time.time(), 2),
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
        logger.exception(f"Ошибка при выполнении поиска: {str(e)}")

        # Обновляем статус с ошибкой
        update_task_status(
            task_id=task_id,
            status="error",
            progress=0,
            message=f"Ошибка при выполнении поиска: {str(e)}",
            stage="error",
            error=str(e)
        )

        # Освобождаем ресурсы даже в случае ошибки
        if use_reranker:
            try:
                qdrant_manager.cleanup()
            except:
                pass


# Эндпоинт для анализа по чек-листу
@app.post("/api/analyze-checklist", response_model=Dict[str, Any])
async def analyze_checklist(
        background_tasks: BackgroundTasks,
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
    update_task_status(
        task_id=task_id,
        status="pending",
        progress=0,
        message="Задача анализа создана",
        stage="prepare"
    )

    # Запускаем анализ в фоновом режиме
    background_tasks.add_task(
        process_checklist_analysis,
        task_id=task_id,
        application_id=request.application_id,
        checklist_path=request.checklist_path,
        limit=request.limit
    )

    return {"task_id": task_id, "status": "pending"}


# Функция для анализа чек-листа в фоновом режиме
async def process_checklist_analysis(
        task_id: str,
        application_id: str,
        checklist_path: str,
        limit: int = 3
):
    """Выполняет анализ чек-листа в фоновом режиме"""
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

        # Формируем результаты
        results = {
            "application_id": application_id,
            "checklist_path": checklist_path,
            "timestamp": datetime.now().isoformat(),
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

        logger.info(f"Анализ по чек-листу для заявки {application_id} успешно завершен")

    except Exception as e:
        logger.exception(f"Ошибка при анализе чек-листа: {str(e)}")

        # Обновляем статус с ошибкой
        update_task_status(
            task_id=task_id,
            status="error",
            progress=0,
            message=f"Ошибка при анализе чек-листа: {str(e)}",
            stage="error",
            error=str(e)
        )


# Вспомогательные эндпоинты
@app.get("/api/applications", response_model=List[str])
async def get_applications():
    """Возвращает список ID заявок в базе данных"""
    try:
        qdrant_manager = get_qdrant_manager()
        application_ids = qdrant_manager.get_application_ids()
        return application_ids
    except Exception as e:
        logger.error(f"Ошибка при получении списка заявок: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении списка заявок: {str(e)}")


@app.delete("/api/applications/{application_id}", response_model=Dict[str, Any])
async def delete_application(application_id: str):
    """Удаляет данные заявки из векторной базы данных"""
    try:
        qdrant_manager = get_qdrant_manager()
        deleted = qdrant_manager.delete_application_data(application_id)
        return {
            "status": "success" if deleted else "error",
            "application_id": application_id,
            "message": f"Удалены данные заявки {application_id}" if deleted else "Не удалось удалить данные"
        }
    except Exception as e:
        logger.error(f"Ошибка при удалении данных заявки {application_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при удалении данных: {str(e)}")


# Эндпоинт для просмотра чанков заявки
@app.get("/api/applications/{application_id}/chunks", response_model=Dict[str, Any])
async def get_application_chunks(application_id: str, limit: int = Query(500, ge=1, le=1000)):
    """
    Возвращает чанки документов заявки.

    - **application_id**: ID заявки
    - **limit**: Максимальное количество возвращаемых чанков
    """
    try:
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

                # Добавляем в список
                chunk = {
                    "id": str(point.id),
                    "text": text,
                    "metadata": metadata
                }
                chunks.append(chunk)

        # Сортируем чанки по порядку (если есть chunk_index в метаданных)
        chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0) if x["metadata"] else 0)

        return {
            "application_id": application_id,
            "chunks_count": len(chunks),
            "stats": stats,
            "chunks": chunks
        }

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

    uvicorn.run(app, host="0.0.0.0", port=8000)