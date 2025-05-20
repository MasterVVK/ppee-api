"""
Класс для ре-ранкинга результатов поиска с использованием bge-reranker
"""

import logging
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

class BGEReranker:
    """Класс для ре-ранкинга результатов с использованием BGE Reranker"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 4096,
        min_vram_mb: int = 500  # Минимальное количество свободной VRAM в МБ
    ):
        """
        Инициализирует ре-ранкер на базе BGE.

        Args:
            model_name: Название модели для ре-ранкинга
            device: Устройство для вычислений (cuda/cpu)
            batch_size: Размер батча для обработки
            max_length: Максимальная длина входного текста
            min_vram_mb: Минимальное количество свободной VRAM в МБ для использования GPU
        """
        self.model_name = model_name
        self.requested_device = device  # Сохраняем изначально запрошенное устройство
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_vram_mb = min_vram_mb

        # Определяем устройство с учетом доступной VRAM
        if device == "cuda" and torch.cuda.is_available():
            if self._check_vram_availability(min_vram_mb):
                self.device = "cuda"
                logger.info(f"Достаточно VRAM для использования GPU")
            else:
                self.device = "cpu"
                logger.warning(f"Недостаточно VRAM для использования GPU. Используем CPU.")
        else:
            self.device = "cpu"
            logger.info("GPU недоступен или не запрошен. Используем CPU.")

        logger.info(f"Инициализация BGE Reranker с моделью {model_name} на устройстве {self.device}")

        try:
            # Загружаем токенизатор и модель
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Переводим модель в режим оценки (не обучения)
            logger.info(f"Модель {model_name} успешно загружена на {self.device}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise RuntimeError(f"Не удалось загрузить модель {model_name}: {str(e)}")

    def _check_vram_availability(self, min_free_mb: int = 500) -> bool:
        """
        Проверяет доступность VRAM для работы ререйтинга.

        Args:
            min_free_mb: Минимальное количество свободной памяти в МБ

        Returns:
            bool: Достаточно ли памяти
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA недоступен, проверка VRAM невозможна")
            return False

        try:
            # Очищаем кэш CUDA перед проверкой
            torch.cuda.empty_cache()

            # Данные от nvidia-smi (более точные для общего состояния)
            nvidia_smi_info = None
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    check=True
                )

                memory_values = result.stdout.strip().split(',')
                nvidia_smi_info = {
                    "total_mb": float(memory_values[0].strip()),
                    "used_mb": float(memory_values[1].strip()),
                    "free_mb": float(memory_values[2].strip())
                }
                logger.info(f"VRAM (nvidia-smi): "
                           f"Всего: {nvidia_smi_info['total_mb']:.1f} МБ, "
                           f"Используется: {nvidia_smi_info['used_mb']:.1f} МБ, "
                           f"Свободно: {nvidia_smi_info['free_mb']:.1f} МБ")
            except Exception as e:
                logger.warning(f"Не удалось получить информацию через nvidia-smi: {str(e)}")

            # Данные от torch.cuda (более точные для текущего процесса)
            torch_info = None
            try:
                device = torch.cuda.current_device()
                device_props = torch.cuda.get_device_properties(device)
                total_mem = device_props.total_memory
                allocated_mem = torch.cuda.memory_allocated(device)
                reserved_mem = torch.cuda.memory_reserved(device)

                torch_info = {
                    "total_mb": total_mem / (1024 * 1024),
                    "allocated_mb": allocated_mem / (1024 * 1024),
                    "reserved_mb": reserved_mem / (1024 * 1024),
                    "free_mb": (total_mem - allocated_mem) / (1024 * 1024)
                }
                logger.info(f"VRAM (torch): "
                           f"Всего: {torch_info['total_mb']:.1f} МБ, "
                           f"Аллоцировано: {torch_info['allocated_mb']:.1f} МБ, "
                           f"Зарезервировано: {torch_info['reserved_mb']:.1f} МБ, "
                           f"Свободно: {torch_info['free_mb']:.1f} МБ")
            except Exception as e:
                logger.warning(f"Не удалось получить информацию через torch.cuda: {str(e)}")

            # Определяем доступную память, используя наиболее консервативную оценку
            if nvidia_smi_info and torch_info:
                # Учитываем как данные nvidia-smi, так и torch.cuda
                # nvidia-smi дает информацию о свободной памяти на уровне системы
                # torch.cuda дает информацию о памяти, которую может использовать текущий процесс

                # Берем минимум из двух значений для консервативной оценки
                free_mem_mb = min(
                    nvidia_smi_info["free_mb"],
                    torch_info["free_mb"]
                )

                # Учитываем уже зарезервированную PyTorch память
                # Если зарезервированная память больше, чем аллоцированная, вычитаем разницу
                if torch_info["reserved_mb"] > torch_info["allocated_mb"]:
                    free_mem_mb -= (torch_info["reserved_mb"] - torch_info["allocated_mb"])

                logger.info(f"Консервативная оценка свободной VRAM: {free_mem_mb:.1f} МБ")
            elif nvidia_smi_info:
                free_mem_mb = nvidia_smi_info["free_mb"]
                logger.info(f"Используем данные nvidia-smi: {free_mem_mb:.1f} МБ")
            elif torch_info:
                free_mem_mb = torch_info["free_mb"]
                logger.info(f"Используем данные torch.cuda: {free_mem_mb:.1f} МБ")
            else:
                logger.error("Не удалось получить информацию о памяти GPU")
                return False

            # Оцениваем необходимую память для модели
            estimated_mem = self._estimate_memory_requirements()
            logger.info(f"Оценочные требования памяти для модели: {estimated_mem:.1f} МБ")

            # Добавляем буфер безопасности (20% от оценки)
            safety_buffer = estimated_mem * 0.2
            total_required = estimated_mem + safety_buffer

            # Принимаем решение на основе свободной памяти
            if free_mem_mb >= min_free_mb and free_mem_mb >= total_required:
                logger.info(f"Проверка VRAM: ОК (свободно {free_mem_mb:.1f} МБ > требуется {total_required:.1f} МБ)")
                return True
            else:
                logger.warning(f"Проверка VRAM: НЕ ОК (свободно {free_mem_mb:.1f} МБ < требуется {total_required:.1f} МБ)")
                return False
        except Exception as e:
            logger.error(f"Ошибка при проверке VRAM: {str(e)}")
            return False

    def _estimate_memory_requirements(self) -> float:
        """
        Оценивает потребление памяти для ререйтера.

        Returns:
            float: Оценка потребления VRAM в МБ
        """
        # Примерные размеры моделей в миллионах параметров и их требования к памяти
        model_sizes = {
            "bge-reranker-base": {"params": 110, "multiplier": 4.2},  # ~110M параметров
            "bge-reranker-v2-m3": {"params": 350, "multiplier": 4.5},  # ~350M параметров
            "bge-reranker-large": {"params": 870, "multiplier": 4.8},  # ~870M параметров
        }

        # Находим ближайшую модель из известных
        model_info = model_sizes.get("bge-reranker-v2-m3")  # По умолчанию для m3
        for known_model, info in model_sizes.items():
            if known_model.lower() in self.model_name.lower():
                model_info = info
                break

        # Расчет памяти (в МБ)
        model_memory_mb = model_info["params"] * model_info["multiplier"]

        # Размер входных данных (запрос и документ)
        # Учитываем размер батча и максимальную длину последовательности
        batch_overhead = self.batch_size * (self.max_length * 4) / (1024 * 1024)  # в МБ

        # Промежуточные буферы (может значительно меняться)
        activation_memory = model_memory_mb * 0.3

        # Служебная память PyTorch и прочие буферы
        system_overhead = 300

        # Общая оценка
        total_memory = model_memory_mb + batch_overhead + activation_memory + system_overhead

        logger.info(f"Оценка памяти для модели {self.model_name}:")
        logger.info(f"  - Параметры модели: ~{model_info['params']}M ({model_memory_mb:.1f} МБ)")
        logger.info(f"  - Батч (размер={self.batch_size}, max_length={self.max_length}): ~{batch_overhead:.1f} МБ")
        logger.info(f"  - Активации: ~{activation_memory:.1f} МБ")
        logger.info(f"  - Системные нужды: ~{system_overhead:.1f} МБ")
        logger.info(f"  - ВСЕГО: ~{total_memory:.1f} МБ")

        return total_memory

    def _fallback_to_cpu(self) -> None:
        """
        Переключает модель на CPU при проблемах с VRAM.
        Освобождает ресурсы GPU и перезагружает модель при необходимости.
        """
        if self.device != "cpu":
            logger.warning(f"Переключение модели {self.model_name} с {self.device} на CPU")
            self.device = "cpu"

            try:
                # Фиксируем имя модели перед удалением объектов
                model_name = self.model_name

                # Отключаем модель от GPU и освобождаем ресурсы
                if hasattr(self, 'model'):
                    self.model.cpu()  # Сначала переносим на CPU
                    del self.model
                    self.model = None

                if hasattr(self, 'tokenizer'):
                    del self.tokenizer
                    self.tokenizer = None

                # Явно собираем мусор
                import gc
                gc.collect()

                # Очищаем кэш CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Ждем небольшое время для освобождения ресурсов
                import time
                time.sleep(1)

                # Заново инициализируем модель и токенизатор на CPU
                logger.info("Перезагрузка модели на CPU...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.eval()  # Режим оценки (не обучения)
                logger.info(f"Модель {model_name} успешно перезагружена на CPU")

            except Exception as e:
                logger.error(f"Критическая ошибка при переключении на CPU: {str(e)}")
                raise RuntimeError(f"Не удалось переключить модель на CPU: {str(e)}")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None,
        text_key: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Выполняет ре-ранкинг списка документов.

        Args:
            query: Поисковый запрос
            documents: Список документов (словарей с ключом 'text' или указанным в text_key)
            top_k: Количество документов в возвращаемом результате (None - все)
            text_key: Ключ, по которому извлекается текстовое содержимое документа

        Returns:
            List[Dict[str, Any]]: Отсортированный список документов с добавленной оценкой rerank_score
        """
        if not documents:
            return []

        if top_k is None:
            top_k = len(documents)

        logger.info(f"Выполнение ре-ранкинга для {len(documents)} документов на устройстве {self.device}")

        # Если использование GPU, проверяем доступность VRAM
        if self.device == "cuda" and not self._check_vram_availability(self.min_vram_mb):
            logger.warning(f"Недостаточно VRAM перед ре-ранкингом. Переключаемся на CPU")
            self._fallback_to_cpu()

        # Получаем тексты документов
        texts = [doc.get(text_key, "") for doc in documents]

        try:
            # Вычисляем оценки релевантности
            scores = self._compute_scores(query, texts)
        except RuntimeError as e:
            # Если ошибка связана с CUDA и мы используем GPU
            if "CUDA out of memory" in str(e) and self.device != "cpu":
                logger.warning(f"Ошибка CUDA при ре-ранкинге: {str(e)}")

                # Переключаемся на CPU и повторяем попытку
                self._fallback_to_cpu()

                # Повторно вычисляем оценки
                scores = self._compute_scores(query, texts)
            else:
                # Если ошибка не связана с CUDA или мы уже на CPU, пробрасываем её дальше
                raise

        # Добавляем оценки ре-ранкинга к документам
        for i, score in enumerate(scores):
            documents[i]["rerank_score"] = float(score)

        # Сортируем документы по убыванию оценки ре-ранкинга
        reranked_documents = sorted(documents, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        # Возвращаем top_k документов
        return reranked_documents[:top_k]

    def _compute_scores(self, query: str, texts: List[str]) -> List[float]:
        """
        Вычисляет оценки релевантности между запросом и текстами.

        Args:
            query: Поисковый запрос
            texts: Список текстов документов

        Returns:
            List[float]: Список оценок релевантности
        """
        scores = []

        # Обрабатываем тексты батчами
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            try:
                # Подготавливаем входные данные для модели
                features = self.tokenizer(
                    [query] * len(batch_texts),
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                # Отключаем вычисление градиентов
                with torch.no_grad():
                    outputs = self.model(**features)
                    batch_scores = outputs.logits.squeeze(-1)

                # Добавляем оценки батча к общему списку
                scores.extend(batch_scores.cpu().tolist())

            except RuntimeError as e:
                # Если ошибка CUDA и мы на GPU, переключаемся на CPU
                if "CUDA out of memory" in str(e) and self.device != "cpu":
                    logger.warning(f"Ошибка CUDA при обработке батча {i}: {str(e)}")

                    # Переключаемся на CPU
                    self._fallback_to_cpu()

                    # Подготавливаем входные данные заново для CPU
                    features = self.tokenizer(
                        [query] * len(batch_texts),
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(self.device)  # теперь self.device - это "cpu"

                    # Вычисляем оценки
                    with torch.no_grad():
                        outputs = self.model(**features)
                        batch_scores = outputs.logits.squeeze(-1)

                    # Добавляем оценки батча к общему списку
                    scores.extend(batch_scores.cpu().tolist())
                else:
                    # Другой тип ошибки - пробрасываем дальше
                    raise

        return scores

    def cleanup(self):
        """
        Освобождает ресурсы, занимаемые моделью.
        Вызывается для очистки VRAM после использования.
        """
        logger.info("Освобождение ресурсов модели ререйтинга...")

        # Выгружаем модель из памяти GPU
        if hasattr(self, 'model'):
            try:
                # Переносим модель на CPU, если она была на GPU
                if self.device == "cuda":
                    self.model = self.model.to('cpu')

                # Удаляем ссылки на модель и токенизатор
                del self.model
                if hasattr(self, 'tokenizer'):
                    del self.tokenizer
                self.model = None
                self.tokenizer = None

                # Явно вызываем сборщик мусора
                import gc
                gc.collect()

                # Очищаем кэш CUDA если доступно
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("VRAM очищена после использования ререйтинга")
            except Exception as e:
                logger.error(f"Ошибка при очистке ресурсов ререйтинга: {str(e)}")