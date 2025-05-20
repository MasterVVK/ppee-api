"""
Вспомогательные функции для семантического разделения документов.
"""

import os
import re
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple

# Настройка логирования
logger = logging.getLogger(__name__)


def detect_docling_availability() -> bool:
    """
    Проверяет доступность библиотеки docling.

    Returns:
        bool: True если docling доступен, иначе False
    """
    try:
        import docling
        from docling.document_converter import DocumentConverter
        return True
    except ImportError:
        logger.warning("Библиотека docling не установлена. Некоторые функции будут недоступны.")
        return False


def detect_gpu_availability() -> bool:
    """
    Проверяет доступность CUDA для работы с GPU.

    Returns:
        bool: True если GPU доступен, иначе False
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            logger.info(f"CUDA доступна. Используется GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA недоступна. Используется CPU.")
        return has_cuda
    except ImportError:
        logger.info("PyTorch не установлен. Используется CPU.")
        return False


def is_likely_table_continuation(content: str) -> bool:
    """
    Определяет, является ли текст продолжением таблицы.

    Args:
        content: Текст для анализа

    Returns:
        bool: True если текст похож на продолжение таблицы, иначе False
    """
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


def identify_content_type(text: str) -> str:
    """
    Определяет тип содержимого текста.

    Args:
        text: Текст для анализа

    Returns:
        str: Тип содержимого ('table', 'heading', 'list', 'text')
    """
    if '|' in text and text.count('|') > 4:
        return 'table'
    elif any(line.startswith(('* ', '- ', '+ ')) for line in text.split('\n')):
        return 'list'
    elif text.strip().startswith('##'):
        return 'heading'
    else:
        return 'text'


def extract_section_info(text: str) -> Dict[str, str]:
    """
    Извлекает информацию о разделе и его структуре.

    Args:
        text: Текст для анализа

    Returns:
        Dict[str, str]: Информация о разделе (section, subsection, section_path)
    """
    section = "Не определено"
    subsection = ""
    section_path = ""

    # Поиск номера раздела (например, 4.г)
    section_number_match = re.search(r'(\d+(\.\w+)*)\.\s+', text)
    if section_number_match:
        section_path = section_number_match.group(1)

    # Поиск заголовка раздела
    section_match = re.search(r'##\s+([^\n]+)', text)
    if section_match:
        section = section_match.group(1).strip()

    # Поиск подзаголовка
    subsection_match = re.search(r'###\s+([^\n]+)', text)
    if subsection_match:
        subsection = subsection_match.group(1).strip()

    return {
        "section": section,
        "subsection": subsection,
        "section_path": section_path
    }


def generate_unique_id() -> str:
    """
    Генерирует уникальный идентификатор.

    Returns:
        str: Уникальный идентификатор
    """
    return str(uuid.uuid4())