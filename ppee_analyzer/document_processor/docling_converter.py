"""
Конвертер PDF в Markdown с использованием docling
"""

import os
import logging
from typing import List, Optional
import requests
import json
import base64

# Настройка логирования
logger = logging.getLogger(__name__)

# Проверяем наличие docling
try:
    import docling
    from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Библиотека docling не установлена")


class DoclingPDFConverter:
    """Класс для конвертации PDF в Markdown с использованием docling"""

    def __init__(self, preserve_tables: bool = True, enable_image_description: bool = False):
        """
        Инициализирует конвертер PDF в Markdown.

        Args:
            preserve_tables: Сохранять ли таблицы в формате Markdown
            enable_image_description: Включить описание изображений через Ollama
        """
        if not DOCLING_AVAILABLE:
            raise ImportError("Библиотека docling не установлена")

        self.preserve_tables = preserve_tables
        self.enable_image_description = enable_image_description

        # Настраиваем опции для PDF формата
        try:
            # Создаем опции для pipeline
            pipeline_options = PdfPipelineOptions()

            # Настройки для изображений
            pipeline_options.generate_picture_images = True  # Генерировать изображения
            pipeline_options.images_scale = 2                # Масштаб изображений (более высокое качество)

            # Создаем опцию для PDF
            pdf_format_option = PdfFormatOption(
                pipeline_options=pipeline_options,
                extract_images=True  # Включаем извлечение изображений
            )

            # Создаем словарь с опциями форматов
            format_options = {
                InputFormat.PDF: pdf_format_option
            }

            # Инициализируем конвертер с опциями
            self.converter = DocumentConverter(format_options=format_options)
            logger.info("Инициализирован конвертер docling с опциями для извлечения изображений")
        except Exception as e:
            # Если что-то пошло не так, используем конвертер по умолчанию
            logger.warning(f"Не удалось настроить опции для docling: {e}")
            self.converter = DocumentConverter()
            logger.info("Инициализирован конвертер docling с настройками по умолчанию")

    def get_image_description(self, image_path: str) -> str:
        """
        Получает описание изображения с помощью Ollama.

        Args:
            image_path: Путь к файлу изображения

        Returns:
            str: Описание изображения
        """
        try:
            # Проверяем существование файла
            if not os.path.exists(image_path):
                logger.error(f"Файл изображения не найден: {image_path}")
                return "Описание недоступно: файл не найден"

            # Чтение файла изображения и кодирование в base64
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Формируем запрос к Ollama API в точном соответствии с документацией
            api_url = "http://localhost:11434/api/generate"

            payload = {
                "model": "gemma3:27b",
                "prompt": "Опиши что изображено на картинке. Ответ дай на русском языке. Будь кратким и точным.",
                "stream": False,
                "images": [image_data]  # Массив с одним закодированным изображением
            }

            logger.info(f"Отправка запроса к Ollama API для описания изображения {os.path.basename(image_path)}")

            # Отправляем запрос
            response = requests.post(api_url, json=payload)

            # Проверяем успешность запроса
            if response.status_code == 200:
                result = response.json()
                # Получаем описание из ключа 'response'
                description = result.get('response', '').strip()
                logger.info(f"Получено описание от Ollama: {description[:50]}...")
                return description
            else:
                error_msg = f"Ошибка при запросе к Ollama API: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Описание недоступно: {error_msg}"

        except Exception as e:
            error_msg = f"Ошибка при получении описания изображения: {str(e)}"
            logger.error(error_msg)
            return f"Описание недоступно: {error_msg}"

    def convert_pdf_to_markdown(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Конвертирует PDF в Markdown.

        Args:
            pdf_path: Путь к PDF-файлу
            output_path: Путь для сохранения результата (если None, возвращает текст)

        Returns:
            str: Содержимое в формате Markdown
        """
        logger.info(f"Начинаем конвертацию PDF: {pdf_path}")

        if not os.path.exists(pdf_path):
            logger.error(f"Файл не найден: {pdf_path}")
            return ""

        try:
            # Конвертируем PDF с помощью docling
            result = self.converter.convert(pdf_path)

            # Получаем Markdown представление
            markdown_content = result.document.export_to_markdown()

            # Проверяем как выглядят маркеры изображений в тексте
            image_marker_count = markdown_content.count("<!-- image -->")
            logger.info(f"Количество маркеров изображений в Markdown: {image_marker_count}")

            # Если указан путь для сохранения
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                logger.info(f"Результат сохранен в файл: {output_path}")

            # Добавляем метод извлечения изображений через PyMuPDF
            if output_path and image_marker_count > 0:
                logger.info("Пробуем извлечь изображения с помощью PyMuPDF")
                try:
                    import fitz  # PyMuPDF

                    # Создаем директорию для изображений
                    images_dir = os.path.join(os.path.dirname(output_path), "images")
                    os.makedirs(images_dir, exist_ok=True)

                    # Открываем PDF
                    doc = fitz.open(pdf_path)
                    image_count = 0
                    image_paths = []  # Сохраняем пути к изображениям для описания

                    # Обрабатываем каждую страницу
                    for page_index, page in enumerate(doc):
                        # Получаем изображения со страницы
                        image_list = page.get_images(full=True)

                        # Обрабатываем каждое изображение
                        for img_index, img in enumerate(image_list):
                            xref = img[0]  # Ссылка на изображение
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]

                            # Сохраняем изображение
                            img_filename = f"image_p{page_index+1}_{img_index+1}.png"
                            img_path = os.path.join(images_dir, img_filename)

                            with open(img_path, 'wb') as img_file:
                                img_file.write(image_bytes)

                            image_paths.append((image_count, img_path, img_filename))
                            image_count += 1

                    # Если нашли изображения, обновляем Markdown с корректными ссылками
                    if image_count > 0:
                        logger.info(f"Извлечено {image_count} изображений с помощью PyMuPDF")

                        # Загружаем текущее содержимое файла
                        with open(output_path, 'r', encoding='utf-8') as f:
                            updated_markdown = f.read()

                        # Генерируем описания изображений, если включено
                        if self.enable_image_description:
                            logger.info("Генерация описаний изображений с помощью Ollama")

                            # Для каждого изображения получаем описание и обновляем Markdown
                            for i, img_path, img_filename in image_paths:
                                if i < image_marker_count:  # Убеждаемся, что у нас достаточно маркеров
                                    # Получаем описание изображения
                                    description = self.get_image_description(img_path)

                                    if description and not description.startswith("Описание недоступно"):
                                        logger.info(f"Получено описание для {img_filename}: {description[:50]}...")

                                        # Заменяем маркер на изображение с описанием
                                        marker = "<!-- image -->"
                                        img_tag = f"![Изображение {i+1}](images/{img_filename})\n\n*Описание: {description}*"
                                        updated_markdown = updated_markdown.replace(marker, img_tag, 1)
                                    else:
                                        # Если описание не получено, просто вставляем изображение
                                        marker = "<!-- image -->"
                                        img_tag = f"![Изображение {i+1}](images/{img_filename})"
                                        updated_markdown = updated_markdown.replace(marker, img_tag, 1)
                        else:
                            # Без описаний - просто заменяем маркеры на изображения
                            for i, img_path, img_filename in image_paths:
                                if i < image_marker_count:  # Убеждаемся, что у нас достаточно маркеров
                                    marker = "<!-- image -->"
                                    img_tag = f"![Изображение {i+1}](images/{img_filename})"
                                    updated_markdown = updated_markdown.replace(marker, img_tag, 1)

                        # Сохраняем обновленный Markdown
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(updated_markdown)
                        logger.info(f"Обновлен Markdown-файл с ссылками на изображения")

                except ImportError:
                    logger.warning("PyMuPDF (fitz) не установлен. Невозможно извлечь изображения.")
                except Exception as e:
                    logger.error(f"Ошибка при извлечении изображений с помощью PyMuPDF: {str(e)}")

            return markdown_content

        except Exception as e:
            logger.error(f"Ошибка при конвертации PDF в Markdown: {str(e)}")
            return ""

    # Метод batch_convert остается без изменений