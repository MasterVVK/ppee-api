"""
Класс для семантического разделения документов ППЭЭ с использованием docling.
"""

import os
import re
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Импорты из текущего модуля
from .models import SemanticChunk, DocumentAnalysisResult
from .utils import (
    detect_docling_availability,
    detect_gpu_availability,
    is_likely_table_continuation,
    identify_content_type,
    extract_section_info,
    generate_unique_id
)

# Импорты для интеграции с ppee_analyzer
from langchain_core.documents import Document
from ..document_processor.splitter import PPEEDocumentSplitter

# Настройка логирования
logger = logging.getLogger(__name__)


class SemanticChunker:
    """Класс для семантического разделения документов с использованием docling"""

    def __init__(self, use_gpu: bool = None, threads: int = 8):
        """
        Инициализирует чанкер для семантического разделения документов.

        Args:
            use_gpu: Использовать ли GPU (None - автоопределение)
            threads: Количество потоков
        """
        # Проверяем доступность docling
        self.docling_available = detect_docling_availability()
        if not self.docling_available:
            raise ImportError("Библиотека docling не установлена. Установите её для работы с SemanticChunker.")

        # Импортируем docling только если он доступен
        import docling
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorDevice, AcceleratorOptions

        # Проверяем доступность GPU
        if use_gpu is None:
            use_gpu = detect_gpu_availability()

        # Настраиваем опции ускорителя
        accelerator_options = AcceleratorOptions(
            num_threads=threads,
            device=AcceleratorDevice.CUDA if use_gpu else AcceleratorDevice.CPU
        )

        # Настраиваем опции обработки PDF
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        # Если используем GPU, включаем Flash Attention 2 для лучшей производительности
        if use_gpu:
            pipeline_options.accelerator_options.cuda_use_flash_attention2 = True

        # Настраиваем конвертер Docling
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        logger.info(f"SemanticChunker инициализирован (GPU: {use_gpu}, потоков: {threads})")

    def extract_chunks(self, pdf_path: str) -> List[Dict]:
        """
        Извлекает и структурирует документ по смысловым блокам.

        Args:
            pdf_path: Путь к PDF файлу

        Returns:
            List[Dict]: Список смысловых блоков документа
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")

        logger.info(f"Начало обработки документа: {pdf_path}")

        # Конвертируем PDF с помощью Docling
        result = self.converter.convert(pdf_path)
        document = result.document

        chunks = []
        current_chunk = {
            "content": "",
            "type": None,
            "page": None,
            "heading": None,
            "table_id": None
        }

        current_table = None
        last_caption = None  # Для хранения последнего заголовка таблицы

        # Словарь для отслеживания статистики по страницам
        pages_encountered = set()

        # Проходим по элементам документа
        for i, (element, level) in enumerate(document.iterate_items()):
            # Определяем страницу
            current_page = None
            if hasattr(element, 'prov') and element.prov and len(element.prov) > 0:
                current_page = element.prov[0].page_no
                pages_encountered.add(current_page)

            # Проверяем, есть ли у элемента атрибут label
            if not hasattr(element, 'label'):
                # Если нет label, но есть текст, добавляем как неопределенный тип
                if hasattr(element, 'text') and element.text.strip():
                    if current_chunk["content"]:
                        chunks.append(current_chunk.copy())

                    current_chunk = {
                        "content": element.text,
                        "type": "unknown",
                        "page": current_page,
                        "heading": None,
                        "table_id": None
                    }
                continue

            # Определяем тип элемента
            if element.label == "caption" or (
                    element.label == "text" and hasattr(element, 'text') and re.match(r'^Таблица\s*\d+[.:]',
                                                                                      element.text, re.IGNORECASE)):
                # Это заголовок таблицы
                if current_chunk["content"] and current_chunk["type"] != "table":
                    chunks.append(current_chunk.copy())

                last_caption = element.text if hasattr(element, 'text') else str(element)
                current_chunk = {
                    "content": "",
                    "type": None,
                    "page": current_page,
                    "heading": None,
                    "table_id": None
                }

            elif element.label == "table":
                # Обработка таблиц
                table_id = element.self_ref if hasattr(element, 'self_ref') else str(uuid.uuid4())

                # Получаем контент таблицы
                table_content = ""
                try:
                    # Пробуем экспортировать в markdown с передачей документа
                    table_content = element.export_to_markdown(doc=document)
                except:
                    try:
                        # Если не получилось, пробуем DataFrame
                        df = element.export_to_dataframe()
                        table_content = df.to_string()
                    except:
                        # В крайнем случае используем строковое представление data
                        table_content = str(element.data) if hasattr(element, 'data') else str(element)

                # Если есть caption, добавляем его
                if hasattr(element, 'caption_text'):
                    try:
                        caption = element.caption_text(document)
                        if caption and not last_caption:
                            last_caption = caption
                    except:
                        pass

                # Всегда создаем новый чанк для таблицы
                if current_chunk["content"]:
                    chunks.append(current_chunk.copy())

                # Создаем чанк для таблицы
                current_chunk = {
                    "content": table_content,
                    "type": "table",
                    "page": current_page,
                    "heading": last_caption,  # Привязываем заголовок к таблице
                    "table_id": table_id,
                    "pages": [current_page] if current_page else []
                }

                # Добавляем этот чанк таблицы
                chunks.append(current_chunk.copy())

                # Сбрасываем current_chunk и table-related переменные
                current_chunk = {
                    "content": "",
                    "type": None,
                    "page": None,
                    "heading": None,
                    "table_id": None
                }
                current_table = None
                last_caption = None

            elif element.label == "heading" or element.label == "section_header":
                # Если это заголовок раздела, начинаем новый чанк
                if current_chunk["content"]:
                    chunks.append(current_chunk.copy())

                current_table = None  # Сбрасываем идентификатор таблицы
                last_caption = None  # Сбрасываем заголовок таблицы
                current_chunk = {
                    "content": element.text if hasattr(element, 'text') else str(element),
                    "type": "heading",
                    "page": current_page,
                    "heading": element.text if hasattr(element, 'text') else str(element),
                    "level": level,
                    "table_id": None
                }

            elif element.label == "document_index":
                # Обработка оглавления как отдельного типа
                if current_chunk["content"]:
                    chunks.append(current_chunk.copy())

                # Пробуем получить контент оглавления
                content = ""
                if hasattr(element, 'text'):
                    content = element.text
                elif hasattr(element, 'export_to_markdown'):
                    try:
                        content = element.export_to_markdown(doc=document)
                    except:
                        content = str(element)
                else:
                    content = str(element)

                current_chunk = {
                    "content": content,
                    "type": "document_index",
                    "page": current_page,
                    "heading": "Оглавление",
                    "table_id": None
                }
                chunks.append(current_chunk.copy())

                # Сбрасываем current_chunk
                current_chunk = {
                    "content": "",
                    "type": None,
                    "page": None,
                    "heading": None,
                    "table_id": None
                }

            elif element.label == "text" or element.label == "paragraph" or element.label == "list-item":
                # Проверяем, не является ли текст подписью к таблице
                if hasattr(element, 'text') and re.match(r'^Таблица\s*\d+[.:]\s*', element.text, re.IGNORECASE):
                    # Это заголовок таблицы
                    if current_chunk["content"]:
                        chunks.append(current_chunk.copy())

                    last_caption = element.text
                    continue

                # Обычный текст или параграф
                current_table = None  # Сбрасываем идентификатор таблицы
                text_content = element.text if hasattr(element, 'text') else str(element)

                if current_chunk["type"] == "heading":
                    # Если предыдущий элемент был заголовком, добавляем текст к нему
                    current_chunk["content"] += "\n\n" + text_content
                    current_chunk["type"] = "section"
                elif current_chunk["type"] == "section":
                    # Если уже идет секция, продолжаем добавлять текст
                    current_chunk["content"] += "\n\n" + text_content
                else:
                    # Начинаем новый текстовый блок
                    if current_chunk["content"]:
                        chunks.append(current_chunk.copy())

                    current_chunk = {
                        "content": text_content,
                        "type": "paragraph" if element.label == "paragraph" else element.label,
                        "page": current_page,
                        "heading": None,
                        "table_id": None
                    }

            else:
                # Для всех остальных типов элементов
                if hasattr(element, 'text') and element.text.strip():
                    if current_chunk["content"]:
                        chunks.append(current_chunk.copy())

                    current_chunk = {
                        "content": element.text,
                        "type": element.label,
                        "page": current_page,
                        "heading": None,
                        "table_id": None
                    }

        # Добавляем последний чанк
        if current_chunk["content"]:
            chunks.append(current_chunk)

        logger.info(f"Документ разделен на {len(chunks)} смысловых блоков")
        logger.info(f"Обработано страниц: {sorted(list(pages_encountered))}")

        return chunks

    def post_process_tables(self, chunks: List[Dict]) -> List[Dict]:
        """
        Постобработка таблиц для объединения разорванных на страницах.

        Args:
            chunks: Список чанков документа

        Returns:
            List[Dict]: Обработанные чанки с объединенными таблицами
        """
        processed_chunks = []
        current_table = None

        # Шаг 1: Создание отображения страниц и анализ нумерации
        page_elements = {}
        element_numbers = {}

        for i, chunk in enumerate(chunks):
            page = chunk.get("page")

            if page is not None:
                if page not in page_elements:
                    page_elements[page] = []
                page_elements[page].append(i)

                # Извлекаем любые числовые последовательности, похожие на нумерацию
                import re
                content = chunk.get("content", "")
                # Ищем паттерны нумерации (число с точкой или число с точкой и подпунктом)
                standard_numbers = re.findall(r'\b(\d+)\.\s', content)
                hierarchy_numbers = re.findall(r'\b(\d+)\.(\d+)\.?\s', content)

                if standard_numbers:
                    element_numbers[i] = [int(n) for n in standard_numbers]

        # Шаг 2: Обработка чанков
        for i, chunk in enumerate(chunks):
            chunk_type = chunk.get("type", "")

            # Обработка явных таблиц
            if chunk_type == "table":
                # Определяем, является ли это продолжением предыдущей таблицы
                is_continuation = False

                if current_table is not None:
                    # Проверка 1: Последовательные страницы
                    prev_pages = current_table.get("pages", [current_table.get("page")])
                    if not isinstance(prev_pages, list):
                        prev_pages = [prev_pages] if prev_pages else []

                    curr_page = chunk.get("page")

                    if prev_pages and curr_page:
                        max_prev_page = max(prev_pages)
                        if curr_page == max_prev_page + 1 or curr_page == max_prev_page:
                            # Таблицы с одинаковым или отсутствующим заголовком вероятно связаны
                            if not chunk.get("heading") or chunk.get("heading") == current_table.get("heading"):
                                is_continuation = True

                    # Проверка 2: Структурное сходство таблиц
                    if not is_continuation:
                        # Анализируем структуру таблиц
                        curr_content = chunk.get("content", "")
                        prev_content = current_table.get("content", "")

                        # Для таблиц с разделителями (|)
                        if "|" in prev_content and "|" in curr_content:
                            # Посчитаем среднее количество столбцов
                            prev_lines = [line.count("|") for line in prev_content.split("\n") if "|" in line][:5]
                            curr_lines = [line.count("|") for line in curr_content.split("\n") if "|" in line][:5]

                            if prev_lines and curr_lines:
                                prev_avg = sum(prev_lines) / len(prev_lines)
                                curr_avg = sum(curr_lines) / len(curr_lines)

                                # Если структура таблиц схожа
                                if abs(prev_avg - curr_avg) <= 2:  # Допустимо небольшое различие
                                    is_continuation = True

                # Если это продолжение предыдущей таблицы, объединяем
                if is_continuation:
                    current_table["content"] += "\n\n" + chunk["content"]

                    # Обновляем страницы
                    existing_pages = current_table.get("pages", [])
                    if not isinstance(existing_pages, list):
                        existing_pages = [existing_pages] if existing_pages else []

                    curr_page = chunk.get("page")
                    if curr_page and curr_page not in existing_pages:
                        existing_pages.append(curr_page)
                        current_table["pages"] = sorted(existing_pages)
                else:
                    # Добавляем предыдущую таблицу и начинаем новую
                    if current_table:
                        processed_chunks.append(current_table)

                    current_table = chunk.copy()

            else:
                # Обработка нетабличных элементов, которые могут быть продолжением таблицы
                if current_table:
                    curr_page = chunk.get("page")
                    prev_page = current_table.get("page")
                    content = chunk.get("content", "")

                    # Проверка на потенциальное продолжение таблицы
                    table_continuation = False

                    # Проверка 1: Элемент находится на следующей странице после таблицы
                    if curr_page and prev_page and curr_page - prev_page <= 2:  # Допускаем разрыв в 1-2 страницы
                        # Проверка на нумерацию, характерную для таблиц
                        import re

                        # Находим любые числовые пункты (например, "29.")
                        number_points = re.findall(r'^\s*(\d+)\.\s', content, re.MULTILINE)

                        if number_points:
                            # Извлекаем номера пунктов из текущей таблицы
                            table_numbers = []
                            table_content = current_table.get("content", "")
                            table_number_points = re.findall(r'^\s*(\d+)\.\s', table_content, re.MULTILINE)

                            if table_number_points:
                                table_numbers = [int(n) for n in table_number_points]
                                current_numbers = [int(n) for n in number_points]

                                # Проверяем, продолжается ли нумерация
                                if table_numbers and current_numbers:
                                    max_table_num = max(table_numbers)
                                    min_current_num = min(current_numbers)

                                    # Если нумерация последовательна или близка к последовательной
                                    if min_current_num > max_table_num and min_current_num - max_table_num <= 5:
                                        table_continuation = True

                        # Проверка 2: Анализ первой строки элемента
                        if not table_continuation:
                            first_line = content.strip().split('\n')[0] if '\n' in content else content.strip()

                            # Проверка, является ли текст продолжением предложения
                            # 1. Нет заглавной буквы в начале (продолжение предложения)
                            # 2. Начинается с предлога или союза (во многих языках)
                            # 3. Нет знаков препинания в начале

                            # Простая эвристика: если первая буква строчная и нет знаков препинания в начале
                            if first_line and not first_line[0].isupper() and not first_line[0] in ',.;:!?':
                                table_continuation = True

                    # Если это продолжение таблицы
                    if table_continuation:
                        # Объединяем с текущей таблицей
                        current_table["content"] += "\n\n" + content

                        # Обновляем страницы
                        existing_pages = current_table.get("pages", [])
                        if not isinstance(existing_pages, list):
                            existing_pages = [existing_pages] if existing_pages else []

                        if curr_page and curr_page not in existing_pages:
                            existing_pages.append(curr_page)
                            current_table["pages"] = sorted(existing_pages)

                        # Пропускаем этот чанк в обработке
                        continue

                    # Если это не продолжение, завершаем таблицу
                    processed_chunks.append(current_table)
                    current_table = None

                # Добавляем обычный чанк
                processed_chunks.append(chunk)

        # Добавляем последнюю таблицу, если она осталась
        if current_table:
            processed_chunks.append(current_table)

        return processed_chunks

    def group_semantic_chunks(self, chunks: List[Dict], min_length: int = 200) -> List[Dict]:
        """
        Объединяет все чанки с одной страницы с учетом продолжения таблиц
        """
        grouped_chunks = []
        current_page_chunks = []
        current_page = None
        last_table_caption = None

        # Проверяем, является ли блок продолжением таблицы
        def is_likely_table_continuation(content: str) -> bool:
            # Признаки продолжения таблицы
            table_indicators = [
                r'\d+\.\s*\w+',  # Нумерация (29. Конструкция выпуска)
                r'^\d+',  # Начинается с числа
                r'Координаты:',  # Специфические слова
                r'^\s*[А-Яа-я\s\-]+$',  # Только текст (возможно заголовок колонки)
                r'соответствии',  # Признак продолжения текста
                r'^[А-Я][а-я]+\s+с',  # Начинается с заглавной буквы и предлога
                r'\|\s*$',  # Признак таблицы
            ]

            for indicator in table_indicators:
                if re.search(indicator, content.strip()[:100]):
                    return True
            return False

        for i, chunk in enumerate(chunks):
            chunk_page = chunk.get("page")

            # Проверяем, является ли текущий чанк заголовком таблицы
            if chunk["type"] == "text" or chunk["type"] == "paragraph":
                if re.match(r'^Таблица\s*\d+[:.]\s*', chunk["content"], re.IGNORECASE):
                    last_table_caption = chunk["content"]
                    continue  # Пропускаем этот чанк, сохраняя заголовок для следующей таблицы

            # Если это таблица
            if chunk["type"] == "table":
                # Добавляем заголовок к таблице, если он есть
                if last_table_caption and not chunk.get("heading"):
                    chunk["heading"] = last_table_caption
                last_table_caption = None

            # Проверяем, не является ли текущий блок продолжением таблицы
            if grouped_chunks and chunk["type"] in ["text", "paragraph", "merged_page"]:
                prev_chunk = grouped_chunks[-1]

                # Если предыдущий чанк - таблица, и текущий находится на следующей странице
                if (prev_chunk["type"] == "table" and
                        chunk_page == prev_chunk.get("page", 0) + 1 and
                        is_likely_table_continuation(chunk["content"])):

                    # Объединяем с предыдущей таблицей
                    prev_chunk["content"] += "\n\n" + chunk["content"]
                    if "pages" not in prev_chunk:
                        prev_chunk["pages"] = [prev_chunk.get("page")]
                    if chunk_page not in prev_chunk["pages"]:
                        prev_chunk["pages"].append(chunk_page)
                    prev_chunk["page"] = min(prev_chunk["pages"])  # Обновляем page до минимальной страницы
                    continue

            # Если страница изменилась и есть накопленные чанки
            if chunk_page != current_page and current_page_chunks:
                grouped_chunks.append(self._merge_page_chunks(current_page_chunks))
                current_page_chunks = [chunk]
                current_page = chunk_page

            # Добавляем чанк к текущей странице
            else:
                current_page_chunks.append(chunk)
                current_page = chunk_page

        # Объединяем последнюю страницу
        if current_page_chunks:
            grouped_chunks.append(self._merge_page_chunks(current_page_chunks))

        return grouped_chunks

    def _merge_page_chunks(self, chunks: List[Dict]) -> Dict:
        """
        Объединяет чанки с одной страницы
        """
        if not chunks:
            return {}

        if len(chunks) == 1:
            return chunks[0]

        # Берем базовую информацию из первого чанка
        merged_chunk = {
            "content": "",
            "type": "merged_page",
            "page": chunks[0].get("page"),
            "heading": None,
            "table_id": None
        }

        sections = []
        current_section = None

        for chunk in chunks:
            # Если это заголовок, начинаем новую секцию
            if chunk["type"] == "heading":
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "heading": chunk["content"],
                    "content": []
                }

            # Если уже есть секция, добавляем контент
            elif current_section:
                current_section["content"].append(chunk["content"])

            # Иначе добавляем как отдельный контент
            else:
                if chunk["content"].strip():
                    sections.append({
                        "heading": None,
                        "content": [chunk["content"]]
                    })

        # Добавляем последнюю секцию
        if current_section:
            sections.append(current_section)

        # Объединяем все секции
        content_parts = []
        for section in sections:
            if section["heading"]:
                content_parts.append(f"## {section['heading']}")
            content_parts.extend(section["content"])

        merged_chunk["content"] = "\n\n".join(content_parts)

        return merged_chunk

    def analyze_document(self, pdf_path: str) -> DocumentAnalysisResult:
        """
        Анализирует PDF документ и возвращает результаты в структурированном виде.

        Args:
            pdf_path: Путь к PDF файлу

        Returns:
            DocumentAnalysisResult: Результаты анализа документа
        """
        # Шаг 1: Извлекаем чанки
        chunks = self.extract_chunks(pdf_path)
        logger.info(f"Найдено {len(chunks)} начальных блоков")

        # Шаг 2: Обрабатываем таблицы
        processed_chunks = self.post_process_tables(chunks)
        logger.info(f"После обработки таблиц: {len(processed_chunks)} блоков")

        # Шаг 3: Группируем короткие блоки
        grouped_chunks = self.group_semantic_chunks(processed_chunks)
        logger.info(f"После группировки: {len(grouped_chunks)} финальных блоков")

        # Собираем статистику
        pages = set()
        content_types = {}

        for chunk in grouped_chunks:
            # Добавляем страницы
            if chunk.get("page"):
                pages.add(chunk["page"])
            elif chunk.get("pages"):
                pages.update(chunk["pages"])

            # Подсчитываем типы контента
            chunk_type = chunk.get("type", "unknown")
            content_types[chunk_type] = content_types.get(chunk_type, 0) + 1

        # Создаем объекты SemanticChunk
        semantic_chunks = []
        for chunk in grouped_chunks:
            semantic_chunks.append(SemanticChunk(
                content=chunk["content"],
                type=chunk["type"],
                page=chunk.get("page"),
                heading=chunk.get("heading"),
                table_id=chunk.get("table_id"),
                pages=chunk.get("pages"),
                section_path=None  # Можно добавить логику определения section_path
            ))

        # Формируем результат
        statistics = {
            "total_chunks": len(semantic_chunks),
            "pages": sorted(list(pages)),
            "total_pages": len(pages),
            "content_types": content_types
        }

        return DocumentAnalysisResult(
            chunks=semantic_chunks,
            document_path=pdf_path,
            statistics=statistics
        )


class SemanticDocumentSplitter(PPEEDocumentSplitter):
    """
    Расширение PPEEDocumentSplitter для использования семантического разделения.
    Интегрирует функциональность SemanticChunker в инфраструктуру ppee_analyzer.
    """

    def __init__(
            self,
            use_gpu: bool = None,
            threads: int = 8,
            chunk_size: int = 1500,
            chunk_overlap: int = 150
    ):
        """
        Инициализирует семантический разделитель документов.

        Args:
            use_gpu: Использовать ли GPU (None - автоопределение)
            threads: Количество потоков
            chunk_size: Размер фрагмента для обычного текста (используется при fallback)
            chunk_overlap: Перекрытие между фрагментами (используется при fallback)
        """
        # Инициализируем базовый класс
        super().__init__(
            text_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Инициализируем семантический чанкер
        try:
            self.semantic_chunker = SemanticChunker(use_gpu=use_gpu, threads=threads)
            self.use_semantic_chunking = True
            logger.info("Инициализирован семантический разделитель документов")
        except ImportError as e:
            logger.warning(f"Не удалось инициализировать семантический разделитель: {e}")
            logger.warning("Будет использован базовый разделитель")
            self.use_semantic_chunking = False

    def process_document(self, text: str, application_id: str, document_id: str, document_name: str) -> List[Document]:
        """
        Обрабатывает документ ППЭЭ и разделяет его на фрагменты с метаданными.
        Переопределяет метод базового класса для использования семантического разделения.

        Args:
            text: Текст документа
            application_id: ID заявки
            document_id: ID документа
            document_name: Название документа

        Returns:
            List[Document]: Список фрагментов с метаданными
        """
        # Проверяем, используем ли семантическое разделение
        if not self.use_semantic_chunking:
            logger.info("Используется базовый разделитель")
            return super().process_document(text, application_id, document_id, document_name)

        # Получаем путь к документу из метаданных (если это текст, преобразуем во временный файл)
        if document_name.lower().endswith('.pdf'):
            # Ищем исходный PDF файл по имени документа
            # Для этого нужно знать структуру директорий проекта
            possible_paths = [
                os.path.join('uploads', document_name),
                os.path.join('data', document_name),
                document_name  # Абсолютный путь
            ]

            pdf_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    pdf_path = path
                    break

            if not pdf_path:
                logger.warning(f"PDF файл не найден: {document_name}")
                logger.info("Используется базовый разделитель")
                return super().process_document(text, application_id, document_id, document_name)

            # Анализируем PDF файл с помощью семантического чанкера
            try:
                # Шаг 1: Извлекаем чанки
                chunks = self.semantic_chunker.extract_chunks(pdf_path)
                logger.info(f"Найдено {len(chunks)} начальных блоков")

                # Шаг 2: Обрабатываем таблицы
                processed_chunks = self.semantic_chunker.post_process_tables(chunks)
                logger.info(f"После обработки таблиц: {len(processed_chunks)} блоков")

                # Шаг 3: Группируем короткие блоки
                grouped_chunks = self.semantic_chunker.group_semantic_chunks(processed_chunks)
                logger.info(f"После группировки: {len(grouped_chunks)} финальных блоков")

                # Преобразуем чанки в формат langchain Document
                documents = []
                for i, chunk in enumerate(grouped_chunks):
                    # Создаем метаданные
                    metadata = {
                        "application_id": application_id,
                        "document_id": document_id,
                        "document_name": document_name,
                        "content_type": chunk.get("type", "unknown"),
                        "chunk_index": i,
                        "section": chunk.get("heading", "Не определено"),
                        "section_path": None,
                        "page_number": chunk.get("page")
                    }

                    # Добавляем информацию о таблице
                    if chunk.get("type") == "table":
                        metadata["table_id"] = chunk.get("table_id")
                        if chunk.get("pages"):
                            metadata["pages"] = chunk.get("pages")

                    # Создаем документ
                    documents.append(Document(
                        page_content=chunk.get("content", ""),
                        metadata=metadata
                    ))

                return documents

            except Exception as e:
                logger.error(f"Ошибка при семантическом разделении: {str(e)}")
                logger.info("Используется базовый разделитель")
                return super().process_document(text, application_id, document_id, document_name)
        else:
            # Для не-PDF документов используем базовый разделитель
            logger.info(f"Документ типа {document_name.split('.')[-1]} обрабатывается базовым разделителем")
            return super().process_document(text, application_id, document_id, document_name)