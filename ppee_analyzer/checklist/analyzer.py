"""
Класс для анализа соответствия документов ППЭЭ чек-листу
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..vector_store.qdrant_manager import QdrantManager

# Настройка логирования
logger = logging.getLogger(__name__)


class ChecklistAnalyzer:
    """Класс для анализа соответствия документов ППЭЭ чек-листу"""

    def __init__(self, qdrant_manager: QdrantManager):
        """
        Инициализирует анализатор чек-листа.

        Args:
            qdrant_manager: Менеджер Qdrant для поиска информации
        """
        self.qdrant_manager = qdrant_manager

    def parse_checklist(self, checklist_path: str) -> List[Dict[str, Any]]:
        """
        Парсит файл чек-листа и извлекает пункты.

        Args:
            checklist_path: Путь к файлу чек-листа

        Returns:
            List[Dict[str, Any]]: Список пунктов чек-листа
        """
        # Проверяем существование файла
        if not os.path.exists(checklist_path):
            logger.error(f"Ошибка: Файл {checklist_path} не существует")
            return []

        # Читаем содержимое файла
        with open(checklist_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Разбиваем на строки и удаляем пустые строки
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        # Извлекаем пункты чек-листа
        checklist_items = []

        # Регулярное выражение для извлечения номера пункта и его содержания
        pattern = r'^(\d+(\.\d+)*)\.\s*(.+)$'

        for line in lines:
            # Пропускаем строки, которые не начинаются с цифры
            if not re.match(r'^\d', line):
                continue

            match = re.match(pattern, line)
            if match:
                number = match.group(1)
                content = match.group(3)

                # Определяем уровень вложенности по числу точек в номере
                level = number.count('.')

                # Формируем поисковый запрос на основе содержания пункта
                query = f"Информация о {content}"

                # Для некоторых конкретных пунктов чек-листа можно задать более точные запросы
                if "полное наименование" in content.lower():
                    query = "Полное наименование юридического лица"
                elif "сокращенное наименование" in content.lower():
                    query = "Краткое наименование юридического лица"
                elif "ОГРН" in content:
                    query = "ОГРН организации"
                elif "ИНН" in content:
                    query = "ИНН организации"
                elif "адрес" in content.lower():
                    query = "Юридический адрес местонахождение"
                elif "наименование ОНВОС" in content:
                    query = "Наименование объекта негативного воздействия"
                elif "местонахождение ОНВОС" in content:
                    query = "Место нахождения адрес объекта НВОС"
                elif "категория ОНВОС" in content:
                    query = "Категория объекта НВОС"
                elif "код ОНВОС" in content:
                    query = "Код объекта НВОС"

                # Создаем элемент чек-листа
                checklist_item = {
                    'id': number,
                    'content': content,
                    'level': level,
                    'query': query,
                    'parent': None
                }

                # Если это подпункт, связываем с родителем
                if level > 0:
                    parent_id = '.'.join(number.split('.')[:-1])
                    checklist_item['parent'] = parent_id

                checklist_items.append(checklist_item)

        logger.info(f"Найдено {len(checklist_items)} пунктов чек-листа")
        return checklist_items

    def search_checklist_items(
            self,
            checklist_items: List[Dict[str, Any]],
            application_id: str,
            limit: int = 3
    ) -> Dict[str, Any]:
        """
        Выполняет поиск по пунктам чек-листа в векторной базе данных.

        Args:
            checklist_items: Список пунктов чек-листа
            application_id: ID заявки
            limit: Максимальное количество результатов для каждого запроса

        Returns:
            Dict[str, Any]: Результаты поиска
        """
        results = {}

        # Общее количество пунктов чек-листа
        total_items = len(checklist_items)
        logger.info(f"Выполнение поиска по {total_items} пунктам чек-листа для заявки {application_id}...")

        # Для каждого пункта чек-листа выполняем поиск
        for i, item in enumerate(checklist_items):
            item_id = item['id']
            query = item['query']

            # Выводим прогресс
            if (i + 1) % 10 == 0 or i == 0 or i == total_items - 1:
                logger.info(f"Прогресс: {i + 1}/{total_items} ({(i + 1) / total_items * 100:.1f}%)")

            try:
                # Выполняем поиск с фильтрацией по заявке
                docs = self.qdrant_manager.search(
                    query=query,
                    filter_dict={"application_id": application_id},
                    k=limit
                )

                # Сохраняем результаты
                results[item_id] = {
                    'query': query,
                    'documents': [
                        {
                            'content': doc.page_content,
                            'metadata': doc.metadata
                        } for doc in docs
                    ]
                }
            except Exception as e:
                logger.error(f"Ошибка при поиске по запросу '{query}': {str(e)}")
                results[item_id] = {
                    'query': query,
                    'documents': [],
                    'error': str(e)
                }

        logger.info(f"Поиск завершен. Получены результаты для {len(results)} пунктов чек-листа")
        return results

    def extract_data_from_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлекает конкретные данные из результатов поиска.

        Args:
            results: Результаты поиска

        Returns:
            Dict[str, Any]: Извлеченные данные
        """
        extracted_data = {}

        logger.info("Извлечение данных из результатов поиска...")

        # Специальные регулярные выражения для извлечения разных типов данных
        patterns = {
            'полное наименование': r'Полное наименование юридического лица[^\n]*([^\n]+)',
            'сокращенное наименование': r'Краткое наименование юридического лица[^\n]*([^\n]+)',
            'ОГРН': r'ОГРН[^\n]*(\d+)',
            'ИНН': r'ИНН[^\n]*(\d+)',
            'адрес': r'Юридический адрес[^\n]*([^\n]+)',
            'наименование ОНВОС': r'Наименование\s+объекта\s+негативного\s+воздействия[^\n]*([^\n]+)',
            'местонахождение ОНВОС': r'Место нахождения \(адрес\)\s+объекта НВОС[^\n]*([^\n]+)',
            'категория ОНВОС': r'Категория объекта[^\n]*([^\n]+)',
            'код ОНВОС': r'Код объекта[^\n]*([^\n]+)',
            'сроки реализации': r'(Программа|Проект).*(разработан|разработана)\s+для.*на срок\s+([^\n]+)',
            'область применения НДТ': r'Производственная деятельность.*относится к области применения наилучших доступных технологий[^\n]*([^\n]+)',
            'применяемые ИТС НДТ': r'ИТС\s+НДТ\s+(\d+-\d+)[^\n]*',
            'категория водного объекта': r'Категория водного объекта[^\n]*([^\n]+)',
        }

        # Обрабатываем каждый пункт чек-листа
        for item_id, item_results in results.items():
            # Пропускаем пункты с ошибками
            if 'error' in item_results:
                extracted_data[item_id] = {
                    'value': f"Ошибка: {item_results['error']}",
                    'source': None
                }
                continue

            # Получаем тексты документов
            documents = item_results.get('documents', [])

            if not documents:
                extracted_data[item_id] = {
                    'value': "Информация не найдена",
                    'source': None
                }
                continue

            # Объединяем тексты документов для поиска
            combined_text = "\n".join([doc['content'] for doc in documents])

            # Ищем подходящий паттерн
            value_found = False
            for key, pattern in patterns.items():
                if key.lower() in item_results['query'].lower():
                    matches = re.findall(pattern, combined_text, re.IGNORECASE)
                    if matches:
                        value = matches[0].strip() if isinstance(matches[0], str) else matches[0][-1].strip()
                        extracted_data[item_id] = {
                            'value': value,
                            'source': {
                                'section': documents[0]['metadata'].get('section', 'Неизвестный раздел'),
                                'content_type': documents[0]['metadata'].get('content_type', 'unknown')
                            }
                        }
                        value_found = True
                        break

            # Если не нашли через паттерны, добавляем первый найденный фрагмент
            if not value_found:
                value = documents[0]['content']
                # Ограничиваем длину
                if len(value) > 500:
                    value = value[:497] + "..."

                extracted_data[item_id] = {
                    'value': value,
                    'source': {
                        'section': documents[0]['metadata'].get('section', 'Неизвестный раздел'),
                        'content_type': documents[0]['metadata'].get('content_type', 'unknown')
                    }
                }

        logger.info(f"Извлечены данные для {len(extracted_data)} пунктов чек-листа")
        return extracted_data

    def analyze_application(
            self,
            checklist_path: str,
            application_id: str,
            limit: int = 3
    ) -> Dict[str, Any]:
        """
        Анализирует заявку на соответствие чек-листу.

        Args:
            checklist_path: Путь к файлу чек-листа
            application_id: ID заявки
            limit: Максимальное количество результатов

        Returns:
            Dict[str, Any]: Результаты анализа
        """
        # Проверяем существование заявки
        application_ids = self.qdrant_manager.get_application_ids()
        if application_id not in application_ids:
            logger.error(f"Заявка с ID {application_id} не найдена")
            return {"error": f"Заявка с ID {application_id} не найдена"}

        # Парсим чек-лист
        checklist_items = self.parse_checklist(checklist_path)

        if not checklist_items:
            logger.error("Ошибка: Не удалось найти пункты чек-листа")
            return {"error": "Не удалось найти пункты чек-листа"}

        # Выполняем поиск
        search_results = self.search_checklist_items(
            checklist_items=checklist_items,
            application_id=application_id,
            limit=limit
        )

        # Извлекаем данные
        extracted_data = self.extract_data_from_results(search_results)

        # Формируем результаты
        results = {
            "application_id": application_id,
            "checklist_path": checklist_path,
            "timestamp": datetime.now().isoformat(),
            "checklist_items": {item['id']: item['content'] for item in checklist_items},
            "extracted_data": extracted_data
        }

        return results

    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """
        Сохраняет результаты анализа в файл.

        Args:
            results: Результаты анализа
            output_file: Путь к выходному файлу
        """
        # Создаем директорию для выходного файла, если она не существует
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Сохраняем результаты в JSON файл
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Результаты сохранены в файл {output_file}")