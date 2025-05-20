"""
Реализация класса для разделения документов ППЭЭ с учетом их структуры
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PPEEDocumentSplitter:
    """Класс для разделения документов ППЭЭ с учетом их структуры"""

    def __init__(
            self,
            text_chunk_size: int = 1500,  # Увеличен с 800 до 1500
            table_chunk_size: int = 2000,
            chunk_overlap: int = 150
    ):
        """
        Инициализирует разделитель документов ППЭЭ.

        Args:
            text_chunk_size: Размер фрагмента для обычного текста
            table_chunk_size: Размер фрагмента для таблиц
            chunk_overlap: Перекрытие между фрагментами
        """
        self.text_chunk_size = text_chunk_size
        self.table_chunk_size = table_chunk_size
        self.chunk_overlap = chunk_overlap

    def split_by_sections(self, text: str) -> List[str]:
        """
        Возвращает весь текст как одну секцию без разделения.

        Args:
            text: Текст документа

        Returns:
            List[str]: Список, содержащий весь текст как одну секцию
        """
        return [text]

    def identify_content_type(self, text: str) -> str:
        """
        Определяет тип содержимого фрагмента.

        Args:
            text: Текст фрагмента

        Returns:
            str: Тип содержимого ("table" или "text")
        """
        # Проверка на таблицу (содержит | и типичные разделители таблиц в markdown)
        if "|" in text and re.search(r'\|[\s-]+\|', text):
            return "table"

        # По умолчанию - обычный текст (включая списки)
        return "text"

    def extract_section_info(self, text: str) -> Dict[str, str]:
        """
        Извлекает информацию о разделе и подразделе.

        Args:
            text: Текст фрагмента

        Returns:
            Dict[str, str]: Информация о разделе
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

    def detect_special_content(self, text: str) -> Dict[str, bool]:
        """
        Определяет наличие специального содержимого.

        Args:
            text: Текст фрагмента

        Returns:
            Dict[str, bool]: Флаги специального содержимого
        """
        return {
            "contains_values": bool(re.search(r'\d+[.,]\d+', text)),
            "contains_dates": bool(re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', text))
        }

    def extract_page_number(self, text: str) -> Optional[int]:
        """
        Пытается извлечь номер страницы.

        Args:
            text: Текст фрагмента

        Returns:
            Optional[int]: Номер страницы, если найден
        """
        # Поиск номера страницы в конце текста или в заголовке
        page_match = re.search(r'(\d+)\s*$', text)
        if page_match:
            try:
                return int(page_match.group(1))
            except ValueError:
                pass
        return None

    def split_section(self, section_text: str) -> List[str]:
        """
        Разделяет секцию на фрагменты с учетом типа содержимого.
        Гарантирует, что заголовок всегда остается с содержимым.

        Args:
            section_text: Текст секции

        Returns:
            List[str]: Список фрагментов
        """
        content_type = self.identify_content_type(section_text)

        # Проверка на наличие заголовка и выделение содержимого
        lines = section_text.split('\n')
        header = None
        content_lines = []

        # Проверяем первую строку на наличие заголовка
        if lines and lines[0].strip().startswith('##'):
            header = lines[0].strip()
            content_lines = lines[1:]
        else:
            content_lines = lines

        content_text = '\n'.join(content_lines).strip()

        # Если нет содержимого после заголовка, возвращаем пустой список
        if not content_text:
            return []

        # Выбираем размер фрагмента в зависимости от типа контента
        if content_type == "table":
            # Для таблиц: проверка размера
            if len(section_text) <= self.table_chunk_size:
                return [section_text]  # Хранить таблицу целиком
            else:
                # Таблица слишком большая - делим по строкам
                rows = content_text.split('\n')
                table_header = []

                # Сохраняем заголовок таблицы (первые 2 строки)
                if len(rows) >= 2 and '|' in rows[0] and '|' in rows[1]:
                    table_header = rows[0:2]
                    rows = rows[2:]

                chunks = []
                current_chunk = []

                # Если есть заголовок раздела, добавляем его первый раз
                if header:
                    current_chunk.append(header)

                if table_header:
                    current_chunk.extend(table_header)

                for row in rows:
                    if len('\n'.join(current_chunk + [row])) > self.table_chunk_size:
                        if current_chunk:
                            chunks.append('\n'.join(current_chunk))
                        current_chunk = table_header.copy()
                    current_chunk.append(row)

                if current_chunk:
                    chunks.append('\n'.join(current_chunk))

                return chunks

        else:  # Обычный текст (включая списки)
            # Создаем полный текст для разделения
            full_text = section_text if not header else content_text

            # Разделяем текст
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.text_chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", ", ", " ", ""],
                keep_separator=True
            )

            chunks = text_splitter.split_text(full_text)

            # Если есть заголовок раздела, добавляем его к первому фрагменту
            if header and chunks:
                chunks[0] = f"{header}\n{chunks[0]}"

            return chunks


    def split_section(self, section_text: str) -> List[str]:
        """
        Разделяет секцию на фрагменты с учетом типа содержимого.
        Гарантирует, что заголовок не отделяется от содержимого.

        Args:
            section_text: Текст секции

        Returns:
            List[str]: Список фрагментов
        """
        content_type = self.identify_content_type(section_text)

        # Выбираем размер фрагмента в зависимости от типа контента
        if content_type == "table":
            # Для таблиц: проверка размера
            if len(section_text) <= self.table_chunk_size:
                return [section_text]  # Хранить таблицу целиком
            else:
                # Таблица слишком большая - делим по строкам
                rows = section_text.split('\n')
                header = rows[0:2]  # Заголовок и разделительная строка

                chunks = []
                current_chunk = header.copy()

                for row in rows[2:]:
                    current_chunk.append(row)
                    if len('\n'.join(current_chunk)) > self.table_chunk_size:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = header.copy()

                if current_chunk and len(current_chunk) > len(header):
                    chunks.append('\n'.join(current_chunk))

                return chunks

        else:  # Обычный текст (включая списки)
            # Проверяем, есть ли заголовок в тексте
            header_match = re.search(r'^##\s+[^\n]+', section_text, re.MULTILINE)

            if header_match:
                # Получаем заголовок
                header = header_match.group(0)

                # Разделяем текст
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.text_chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", ", ", " ", ""],
                    keep_separator=True
                )

                chunks = text_splitter.split_text(section_text)

                # Проверяем каждый фрагмент
                processed_chunks = []
                skip_next = False

                for i in range(len(chunks)):
                    if skip_next:
                        skip_next = False
                        continue

                    chunk = chunks[i]

                    # Если фрагмент содержит только заголовок (или близок к этому)
                    if chunk.strip() == header.strip() or (len(chunk.strip()) - len(header.strip()) < 20):
                        # Если это последний фрагмент, объединяем его с предыдущим (если есть)
                        if i == len(chunks) - 1 and processed_chunks:
                            processed_chunks[-1] = processed_chunks[-1] + "\n" + chunk
                        # Если есть следующий фрагмент, объединяем с ним
                        elif i < len(chunks) - 1:
                            combined_chunk = chunk + "\n" + chunks[i+1]
                            processed_chunks.append(combined_chunk)
                            skip_next = True
                        else:
                            # В крайнем случае добавляем как есть
                            processed_chunks.append(chunk)
                    else:
                        processed_chunks.append(chunk)

                return processed_chunks
            else:
                # Если заголовка нет, просто разделяем текст
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.text_chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", ", ", " ", ""],
                    keep_separator=True
                )

                chunks = text_splitter.split_text(section_text)
                return chunks

    def process_document(self, text: str, application_id: str, document_id: str, document_name: str) -> List[Document]:
        """
        Обрабатывает документ ППЭЭ и разделяет его на фрагменты с метаданными.

        Args:
            text: Текст документа
            application_id: ID заявки
            document_id: ID документа
            document_name: Название документа

        Returns:
            List[Document]: Список фрагментов с метаданными
        """
        # Шаг 1: Разделение на крупные секции по заголовкам
        sections = self.split_by_sections(text)

        chunks = []
        chunk_index = 0

        # Шаг 2: Обработка каждой секции
        for section_text in sections:
            # Определяем информацию о разделе
            section_info = self.extract_section_info(section_text)
            content_type = self.identify_content_type(section_text)

            # Шаг 3: Разделяем секцию на фрагменты
            section_chunks = self.split_section(section_text)

            # Шаг 4: Добавляем метаданные к каждому фрагменту
            for i, chunk_text in enumerate(section_chunks):
                # Пропускаем пустые фрагменты
                if not chunk_text.strip():
                    continue

                # Проверяем наличие содержания после заголовка
                lines = chunk_text.strip().split('\n', 1)  # Делим только на 2 части
                if len(lines) == 1 and lines[0].startswith('##'):
                    # Это только заголовок без содержания - пропускаем
                    continue

                # Определяем дополнительную информацию
                special_content = self.detect_special_content(chunk_text)
                page_number = self.extract_page_number(chunk_text)

                # Создаем метаданные
                metadata = {
                    "application_id": application_id,
                    "document_id": document_id,
                    "document_name": document_name,
                    "section": section_info["section"],
                    "subsection": section_info["subsection"],
                    "section_path": section_info["section_path"],
                    "content_type": content_type,
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "contains_values": special_content["contains_values"],
                    "contains_dates": special_content["contains_dates"]
                }

                # Создаем документ
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
                chunk_index += 1

        return chunks


    def load_and_process_file(self, file_path: str, application_id: str) -> List[Document]:
        """
        Загружает файл и обрабатывает его.

        Args:
            file_path: Путь к файлу
            application_id: ID заявки

        Returns:
            List[Document]: Список фрагментов с метаданными
        """
        logger = logging.getLogger(__name__)

        # Проверяем существование файла
        if not os.path.exists(file_path):
            error_msg = f"Файл не найден: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Загрузка и обработка файла: {file_path}")

        # Определяем расширение файла
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            # Выбираем подходящий загрузчик в зависимости от типа файла
            from langchain_community.document_loaders import TextLoader, PyPDFLoader

            if ext == '.pdf':
                logger.info(f"Использование PyPDFLoader для файла: {file_path}")
                try:
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                except Exception as e:
                    logger.error(f"Ошибка PyPDFLoader: {str(e)}")
                    # Запасной вариант - просто загружаем как текст
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents = loader.load()
            else:
                logger.info(f"Использование TextLoader для файла: {file_path}")
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()

            # Если документ состоит из нескольких частей (например, страниц PDF)
            # объединяем их
            if len(documents) > 1:
                logger.info(f"Объединение {len(documents)} фрагментов документа")
                combined_text = "\n\n".join([doc.page_content for doc in documents])
                document_text = combined_text
            else:
                document_text = documents[0].page_content

            # Создаем идентификатор документа на основе имени файла
            document_id = f"doc_{os.path.basename(file_path).replace(' ', '_').replace('.', '_')}"
            document_name = os.path.basename(file_path)

            logger.info(f"Обработка документа: ID={document_id}, Имя={document_name}")

            # Обрабатываем документ
            chunks = self.process_document(
                text=document_text,
                application_id=application_id,
                document_id=document_id,
                document_name=document_name
            )

            logger.info(f"Документ успешно разделен на {len(chunks)} фрагментов")
            return chunks

        except Exception as e:
            logger.exception(f"Ошибка при загрузке и обработке файла {file_path}: {str(e)}")
            raise