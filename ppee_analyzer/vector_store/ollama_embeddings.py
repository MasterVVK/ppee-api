"""
Класс для работы с эмбеддингами Ollama
"""

import requests
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from langchain_core.embeddings import Embeddings

# Настройка логирования
logger = logging.getLogger(__name__)


class OllamaEmbeddings(Embeddings):
    """Класс для работы с эмбеддингами через локальный сервер Ollama"""

    # Статические настройки по умолчанию
    DEFAULT_OPTIONS = {
        "num_ctx": 8192,      # Максимальный размер контекста
        "num_keep": 8192,     # Сколько токенов сохранять
        "num_thread": 2      # Для многопоточной обработки
        #"use_gpu": True       # Использовать GPU если доступно
    }

    def __init__(
            self,
            model_name: str = "nomic-embed-text",
            base_url: str = "http://localhost:11434",
            embed_batch_size: int = 10,
            timeout: int = 60,
            normalize_embeddings: bool = True,
            check_availability: bool = True,
            options: Dict[str, Any] = None  # Переопределение настроек
    ):
        """
        Инициализирует класс для работы с эмбеддингами Ollama.

        Args:
            model_name: Название модели для эмбеддингов в Ollama
            base_url: Базовый URL для API Ollama
            embed_batch_size: Размер пакета для обработки текстов
            timeout: Таймаут запроса в секундах
            normalize_embeddings: Нормализовать ли векторы эмбеддингов
            check_availability: Проверять ли доступность модели при инициализации
            options: Дополнительные опции для запросов к Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.embed_batch_size = embed_batch_size
        self.timeout = timeout
        self.normalize_embeddings = normalize_embeddings

        # Создаем копию настроек по умолчанию и обновляем их переданными параметрами
        self.options = self.DEFAULT_OPTIONS.copy()
        if options:
            self.options.update(options)

        logger.info(f"Инициализация OllamaEmbeddings с опциями: {self.options}")

        # Проверка доступности сервера и модели только если это требуется
        if check_availability:
            self._check_model_availability()

    @classmethod
    def get_default_options(cls) -> Dict[str, Any]:
        """
        Возвращает копию настроек по умолчанию.

        Returns:
            Dict[str, Any]: Копия настроек по умолчанию
        """
        return cls.DEFAULT_OPTIONS.copy()

    def _check_model_availability(self) -> None:
        """
        Проверяет доступность сервера Ollama и указанной модели.
        Выполняет загрузку модели, если она еще не загружена.
        """
        try:
            # Проверяем доступность сервера
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            response.raise_for_status()

            models = response.json().get("models", [])
            model_exists = any(model.get("name") == self.model_name for model in models)

            if not model_exists:
                logger.info(f"Модель {self.model_name} не найдена. Попытка загрузки...")
                self._pull_model()
            else:
                logger.info(f"Модель {self.model_name} доступна для использования")

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при проверке доступности Ollama: {str(e)}")
            raise ConnectionError(f"Не удалось подключиться к серверу Ollama: {str(e)}")

    def _pull_model(self) -> None:
        """
        Загружает модель, если она еще не загружена.
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={
                    "name": self.model_name,
                    "options": self.options  # Используем единые опции
                },
                timeout=None  # Убираем таймаут для загрузки модели
            )
            response.raise_for_status()
            logger.info(f"Модель {self.model_name} успешно загружена")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при загрузке модели {self.model_name}: {str(e)}")
            raise RuntimeError(f"Не удалось загрузить модель {self.model_name}: {str(e)}")

    def _get_embedding(self, text: str) -> List[float]:
        """
        Получает эмбеддинг для одного текста.

        Args:
            text: Текст для эмбеддинга

        Returns:
            List[float]: Вектор эмбеддинга
        """
        # Добавляем префикс "query:" для моделей BGE
        if "bge" in self.model_name.lower():
            text = f"query: {text}"

        # Формируем запрос к API Ollama
        url = f"{self.base_url}/api/embed"  # Используем /api/embed вместо /api/embeddings

        payload = {
            "model": self.model_name,
            "input": text,  # Для /api/embed используется "input" вместо "prompt"
            "options": self.options  # Используем единые опции
        }

        logger.info(f"Отправка запроса для эмбеддинга с опциями: {self.options}")

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Получаем эмбеддинг
            result = response.json()
            embeddings = result.get("embeddings", [])

            if embeddings and len(embeddings) > 0:
                embedding = embeddings[0]  # Берем первый эмбеддинг
                logger.info(f"Получен эмбеддинг длиной {len(embedding)}")
            else:
                logger.error("Ollama API не вернул эмбеддинг")
                return [0.0] * 1024  # Возвращаем нулевой вектор для bge-m3

            # Нормализация вектора, если требуется
            if self.normalize_embeddings and embedding:
                embedding = self._normalize_vector(embedding)

            return embedding

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при получении эмбеддинга: {str(e)}")
            # В случае ошибки возвращаем нулевой вектор для bge-m3
            return [0.0] * 1024

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Нормализует вектор до единичной длины.

        Args:
            vector: Вектор для нормализации

        Returns:
            List[float]: Нормализованный вектор
        """
        array = np.array(vector)
        norm = np.linalg.norm(array)
        if norm > 0:
            normalized = array / norm
            return normalized.tolist()
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Получает эмбеддинги для списка текстов.

        Args:
            texts: Список текстов для эмбеддинга

        Returns:
            List[List[float]]: Список векторов эмбеддингов
        """
        embeddings = []

        # Обрабатываем тексты пакетами для оптимизации
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i + self.embed_batch_size]
            batch_embeddings = [self._get_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
            logger.debug(
                f"Обработан пакет {i // self.embed_batch_size + 1} из {(len(texts) - 1) // self.embed_batch_size + 1}")

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Получает эмбеддинг для запроса.

        Args:
            text: Текст запроса

        Returns:
            List[float]: Вектор эмбеддинга
        """
        # Используем тот же метод _get_embedding
        return self._get_embedding(text)