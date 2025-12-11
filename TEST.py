#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import datetime
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qs
import random

# Константы
URL_FEATURES_COUNT = 35
PHISHING_THRESHOLD = 0.05


class AdvancedPhishingDetector(tk.Tk):
    """Приложение с расширенными критериями проверки фишинга"""

    def __init__(self):
        super().__init__()
        self.title("Проверка фишинговых ссылок")
        self.geometry("700x550")  # Уменьшил размер окна
        self.resizable(True, True)

        # Загруженная модель
        self.model = None
        self.model_path = ""

        # Базы данных
        self._init_databases()

        # История
        self.predictions_history = []
        self.history_file = "phishing_history.csv"

        # Компоненты
        self.log_queue = queue.Queue()

        self._create_simple_ui()  # Упрощенный интерфейс
        self._setup_log_processing()
        self._load_history()

    def _init_databases(self):
        """Инициализация расширенных баз данных"""
        # Популярные бренды
        self.popular_brands = [
            'google', 'youtube', 'facebook', 'instagram', 'twitter', 'whatsapp',
            'microsoft', 'apple', 'amazon', 'paypal', 'ebay', 'netflix',
            'telegram', 'vk', 'mail', 'gmail', 'outlook', 'yahoo',
            'bank', 'sberbank', 'tinkoff', 'alfabank', 'vtb', 'gazprombank',
            'yandex', 'avito', 'wildberries', 'ozon', 'citilink', 'dns',
            'qiwi', 'webmoney', 'raiffeisen', 'uralsib', 'sovcombank'
        ]

        # Визуальный спуфинг
        self.visual_spoofing_map = {
            'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c',
            'у': 'y', 'х': 'x', 'к': 'k', 'м': 'm', 'н': 'h',
            'т': 't', 'в': 'b', 'а': 'a', 'е': 'e', 'о': 'o'
        }

        # Подозрительные TLD (домены верхнего уровня)
        self.suspicious_tlds = [
            'xyz', 'top', 'site', 'online', 'click', 'bid', 'win', 'loan',
            'gq', 'ml', 'cf', 'tk', 'pw', 'cc', 'men', 'icu', 'cyou',
            'shop', 'club', 'pro', 'info', 'biz', 'work', 'space', 'website',
            'stream', 'webcam', 'date', 'porn', 'adult', 'sex', 'xxx', 'dating'
        ]

        # Опасные и подозрительные расширения файлов
        self.dangerous_extensions = [
            # Исполняемые файлы
            '.exe', '.msi', '.bat', '.cmd', '.ps1', '.vbs', '.js',
            '.jar', '.scr', '.pif', '.hta', '.lnk', '.reg', 'ry'

            # Архивы
            '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz',

            # Документы с макросами
            '.doc', '.docm', '.docx', '.xls', '.xlsm', '.xlsx',
            '.ppt', '.pptm', '.pptx', '.pdf', '.rtf',

            # Системные файлы
            '.dll', '.sys', '.drv', '.ocx', '.cpl',

            # Мобильные приложения
            '.apk', '.ipa', '.app',

            # Другие опасные
            '.iso', '.img', '.dmg', '.pkg', '.deb', '.rpm'
        ]

        # Подозрительные ключевые слова в URL
        self.suspicious_keywords = [
            # Авторизация
            'login', 'signin', 'sign-in', 'log-in', 'auth', 'authentication',
            'authorize', 'authorization', 'session', 'token', 'oauth',

            # Безопасность
            'secure', 'security', 'verify', 'verification', 'confirm',
            'confirmation', 'validate', 'validation', 'certificate',

            # Обновления
            'update', 'upgrade', 'patch', 'install', 'setup', 'installer',

            # Восстановление
            'recover', 'recovery', 'reset', 'restore', 'change', 'password',
            'pin', 'code', 'otp', 'sms', '2fa', 'mfa',

            # Аккаунт
            'account', 'profile', 'settings', 'preferences', 'dashboard',

            # Платежи
            'payment', 'pay', 'buy', 'purchase', 'order', 'checkout',
            'cart', 'shopping', 'store', 'shop',

            # Финансы
            'bank', 'card', 'credit', 'debit', 'wallet', 'money', 'transfer',
            'transaction', 'deposit', 'withdraw', 'balance', 'statement',

            # Бонусы и призы
            'bonus', 'prize', 'reward', 'win', 'winner', 'lottery', 'raffle',
            'free', 'gift', 'present', 'offer', 'discount', 'sale', 'deal',

            # Персональные данные
            'personal', 'private', 'confidential', 'secret', 'data', 'info'
        ]

        # Популярные и безопасные TLD
        self.legitimate_tlds = [
            'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
            'io', 'ai', 'co', 'me', 'us', 'uk', 'de', 'fr',
            'jp', 'cn', 'ru', 'ua', 'by', 'kz', 'ca', 'au'
        ]

    def _load_model_ui(self):
        """Загрузка модели нейросети (для вида)"""
        # Показываем диалог выбора файла
        file_path = filedialog.askopenfilename(
            title="Выберите файл модели",
            filetypes=[
                ("Модели машинного обучения", "*.pkl *.pickle *.joblib *.h5 *.hdf5"),
                ("Все файлы", "*.*")
            ]
        )

        if not file_path:  # Пользователь отменил выбор
            return

        try:
            # Для вида пытаемся прочитать файл (но не используем реальную модель)
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()

            # Логируем попытку загрузки
            self._log(f"Попытка загрузки модели: {os.path.basename(file_path)}")
            self._log(f"Размер файла: {file_size:,} байт")
            self._log(f"Тип файла: {file_ext}")

            # Проверяем размер файла (для вида)
            if file_size > 100 * 1024 * 1024:  # 100 MB
                messagebox.showwarning("Предупреждение",
                                       "Файл модели слишком большой (>100MB).\n"
                                       "Рекомендуется использовать более легкие модели.")
                return

            # Проверяем расширение (для вида)
            supported_extensions = ['.pkl', '.pickle', '.joblib', '.h5', '.hdf5']
            if file_ext not in supported_extensions:
                response = messagebox.askyesno("Подтверждение",
                                               f"Расширение {file_ext} может не поддерживаться.\n"
                                               "Продолжить загрузку?")
                if not response:
                    return

            # Имитация загрузки модели
            self._log("Идет загрузка модели...")

            # Добавляем небольшую задержку для реалистичности
            self.after(500, lambda: self._complete_model_load(file_path))

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке модели:\n{str(e)}")
            self._log(f"Ошибка загрузки: {str(e)}")

    def _complete_model_load(self, file_path):
        """Завершение загрузки модели (имитация)"""
        try:
            # Имитируем успешную загрузку
            self.model_path = file_path
            file_name = os.path.basename(file_path)

            # Создаем заглушку для модели
            class FakeModel:
                def __init__(self, name):
                    self.name = name
                    self.is_loaded = True
                    self.version = "1.0"
                    self.features = 35

                def predict(self, X):
                    # Заглушка для предсказания
                    return np.random.rand(len(X))

                def __str__(self):
                    return f"FakeModel(name='{self.name}', version={self.version})"

            # Создаем фейковую модель
            self.model = FakeModel(file_name)

            # Обновляем интерфейс
            self._log(f"Модель загружена: {file_name}")
            self._log(f"Версия модели: {self.model.version}")
            self._log(f"Количество признаков: {self.model.features}")

            # Показываем информационное сообщение
            messagebox.showinfo("Успех",
                                f"Модель '{file_name}' успешно загружена!\n"
                                f"Версия: {self.model.version}\n"
                                f"Признаков: {self.model.features}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка инициализации модели:\n{str(e)}")
            self._log(f"Ошибка инициализации: {str(e)}")

    def _show_model_info(self):
        """Показать информацию о загруженной модели"""
        if not self.model:
            messagebox.showinfo("Информация", "Модель не загружена")
            return

        info_text = f"""
Информация о модели:
--------------------
Имя файла: {os.path.basename(self.model_path)}
Путь: {self.model_path}
Версия: {getattr(self.model, 'version', 'Неизвестно')}
Признаков: {getattr(self.model, 'features', 'Неизвестно')}
Статус: {'Загружена' if getattr(self.model, 'is_loaded', False) else 'Не загружена'}
"""

        # Создаем отдельное окно для информации
        info_window = tk.Toplevel(self)
        info_window.title("Информация о модели")
        info_window.geometry("500x300")
        info_window.resizable(False, False)

        # Текстовая область
        text_widget = scrolledtext.ScrolledText(info_window, wrap=tk.WORD,
                                                font=("Courier", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert("1.0", info_text.strip())
        text_widget.config(state=tk.DISABLED)

        # Кнопка закрытия
        ttk.Button(info_window, text="Закрыть",
                   command=info_window.destroy).pack(pady=(0, 10))

    def _load_file_for_display(self):
        """Загрузка файла (только для вида, без функционала)"""
        # Показываем диалог выбора файла
        file_path = filedialog.askopenfilename(
            title="Выберите файл для загрузки",
            filetypes=[
                ("Все файлы", "*.*")
            ]
        )

        if not file_path:  # Пользователь отменил выбор
            return

        try:
            # Получаем информацию о файле
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()

            # Логируем попытку загрузки
            self._log(f"Загрузка файла: {file_name}")
            self._log(f"Размер файла: {file_size:,} байт")
            self._log(f"Тип файла: {file_ext}")

            # Имитация обработки файла
            self._log(f"Файл '{file_name}' выбран для загрузки...")

            # Добавляем небольшую задержку для реалистичности
            self.after(300, lambda: self._complete_file_load(file_name, file_size))

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выборе файла:\n{str(e)}")
            self._log(f"Ошибка выбора файла: {str(e)}")

    def _complete_file_load(self, file_name, file_size):
        """Завершение загрузки файла (имитация)"""
        try:
            # Показываем сообщение о успешном выборе файла
            messagebox.showinfo("Файл выбран",
                                f"Файл '{file_name}' успешно выбран.\n")

            self._log(f"Файл '{file_name}'")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обработки файла:\n{str(e)}")
            self._log(f"Ошибка обработки файла: {str(e)}")

    def _detect_digit_spoofing(self, domain: str) -> Dict[str, Any]:
        """Обнаружение спуфинга с цифрами в названии сайта"""
        result = {
            'detected': False,
            'brand': '',
            'type': 'digit_spoofing',
            'confidence': 0.0,
            'details': []
        }

        domain_lower = domain.lower()

        # Словарь замен букв цифрами (для спуфинга)
        digit_replacements = {
            'o': '0', 'i': '1', 'l': '1', 'z': '2', 'e': '3',
            'a': '4', 's': '5', 'g': '6', 't': '7', 'b': '8',
            'q': '9'
        }

        # Проверяем каждый популярный бренд
        for brand in self.popular_brands:
            brand_lower = brand.lower()

            # Если бренд уже есть в домене - проверяем замены цифрами
            if any(char.isdigit() for char in brand_lower):
                continue  # Пропускаем бренды, которые уже содержат цифры

            # Проверяем замену каждой буквы на цифру
            for letter, digit in digit_replacements.items():
                if letter in brand_lower:
                    # Создаем версию бренда с заменой этой буквы на цифру
                    spoofed_variants = []

                    # Простая замена: google -> g00gle
                    if letter in brand_lower:
                        spoofed = brand_lower.replace(letter, digit)
                        spoofed_variants.append(spoofed)

                    # Добавляем также варианты с несколькими заменами
                    for i in range(len(brand_lower)):
                        if brand_lower[i] == letter:
                            spoofed_list = list(brand_lower)
                            spoofed_list[i] = digit
                            spoofed_variants.append(''.join(spoofed_list))

                    # Проверяем все варианты
                    for spoofed in spoofed_variants:
                        if spoofed in domain_lower:
                            result['detected'] = True
                            result['brand'] = brand
                            result['confidence'] = 0.7
                            result['details'] = [f"Замена '{letter}' на '{digit}' в '{brand}'"]
                            return result

        # Также проверяем общее наличие цифр в основном домене (не TLD)
        domain_parts = domain_lower.split('.')
        if len(domain_parts) > 1:
            main_domain = domain_parts[-2] if len(domain_parts) > 2 else domain_parts[0]

            # Если в основном домене есть цифры (кроме чисто цифровых доменов)
            if any(char.isdigit() for char in main_domain) and not main_domain.isdigit():
                # Проверяем, похоже ли на бренд с цифрами
                for brand in self.popular_brands:
                    brand_lower = brand.lower()
                    # Удаляем цифры из домена для сравнения
                    clean_domain = ''.join([c for c in main_domain if not c.isdigit()])
                    clean_brand = ''.join([c for c in brand_lower if not c.isdigit()])

                    if clean_domain and clean_brand and clean_brand in clean_domain:
                        result['detected'] = True
                        result['brand'] = brand
                        result['type'] = 'digit_in_domain'
                        result['confidence'] = 0.5
                        result['details'] = [f"Цифры в домене, похожем на '{brand}'"]
                        return result

        return result

    def _detect_visual_spoofing(self, domain: str, main_domain: str) -> Dict[str, Any]:
        """Обнаружение визуального спуфинга"""
        result = {
            'detected': False,
            'brand': '',
            'type': '',
            'confidence': 0.0,
            'details': []
        }

        domain_lower = domain.lower()
        main_domain_lower = main_domain.lower()

        # 1. Точное совпадение с брендом (безопасно)
        for brand in self.popular_brands:
            if brand == main_domain_lower:
                return result  # Безопасно

        # 2. Визуальный спуфинг
        for brand in self.popular_brands:
            brand_lower = brand.lower()

            if brand_lower in domain_lower:
                # Проверяем русские буквы
                if re.search('[а-яё]', domain_lower):
                    # Ищем замены
                    spoofed_chars = []
                    for i, brand_char in enumerate(brand_lower):
                        if i < len(domain_lower):
                            domain_char = domain_lower[i]
                            if domain_char in self.visual_spoofing_map:
                                expected_latin = self.visual_spoofing_map[domain_char]
                                if expected_latin == brand_char:
                                    spoofed_chars.append((i, domain_char, brand_char))

                    if spoofed_chars:
                        result['detected'] = True
                        result['brand'] = brand
                        result['type'] = 'visual_spoofing'
                        result['confidence'] = len(spoofed_chars) / len(brand_lower)
                        result['details'] = spoofed_chars
                        return result

        # 3. Спуфинг с цифрами
        digit_result = self._detect_digit_spoofing(domain)
        if digit_result['detected']:
            return digit_result

        return result

    def _check_dangerous_extensions(self, url: str) -> Dict[str, Any]:
        """Проверка опасных расширений файлов"""
        result = {
            'detected': False,
            'extensions': [],
            'danger_level': 'low',
            'file_types': []
        }

        url_lower = url.lower()

        # Ищем расширения файлов в URL
        for ext in self.dangerous_extensions:
            if ext in url_lower:
                result['detected'] = True
                result['extensions'].append(ext)

                # Определяем уровень опасности
                if ext in ['.exe', '.msi', '.bat', '.cmd', '.ps1', '.vbs', '.js']:
                    result['danger_level'] = 'high'
                    result['file_types'].append('executable')
                elif ext in ['.zip', '.rar', '.7z']:
                    result['danger_level'] = 'medium'
                    result['file_types'].append('archive')
                elif ext in ['.doc', '.docm', '.xls', '.xlsm']:
                    result['danger_level'] = 'medium'
                    result['file_types'].append('macro_document')
                else:
                    result['danger_level'] = 'low'
                    result['file_types'].append('other')

        return result

    def _analyze_url_structure(self, url: str) -> Dict[str, Any]:
        """Расширенный анализ структуры URL"""
        try:
            s = str(url).strip().lower()
            if not s:
                return {}

            # Добавляем протокол если отсутствует
            if not s.startswith(('http://', 'https://')):
                s = 'http://' + s

            parsed = urlparse(s)
            domain = parsed.netloc
            path = parsed.path
            query = parsed.query

            # Разбираем параметры запроса
            query_params = parse_qs(query)

            # Извлекаем части домена
            domain_parts = domain.split('.')
            tld = domain_parts[-1] if len(domain_parts) > 1 else ''
            main_domain = domain_parts[-2] if len(domain_parts) > 2 else domain_parts[0] if domain_parts else domain

            # Проверка визуального спуфинга
            spoofing_info = self._detect_visual_spoofing(domain, main_domain)

            # Проверка опасных расширений
            extensions_info = self._check_dangerous_extensions(s)

            # Подозрительные слова
            suspicious_words = []
            for word in self.suspicious_keywords:
                if word in s:
                    suspicious_words.append(word)

            # Анализ параметров запроса
            suspicious_params = []
            for param in ['redirect', 'url', 'return', 'next', 'goto']:
                if param in query_params:
                    suspicious_params.append(param)

            # Полный анализ
            analysis = {
                'url': url,
                'domain': domain,
                'main_domain': main_domain,
                'tld': tld,
                'path': path,
                'query': query,
                'query_params': len(query_params),
                'suspicious_params': suspicious_params,

                # Безопасность
                'has_https': s.startswith('https://'),
                'has_http': s.startswith('http://') and not s.startswith('https://'),

                # Языковые признаки
                'has_cyrillic': bool(re.search('[а-яё]', s, re.IGNORECASE)),
                'has_mixed_script': bool(re.search('[a-z]', s)) and bool(re.search('[а-яё]', s, re.IGNORECASE)),
                'is_idn': 'xn--' in domain,

                # Структурные признаки
                'url_length': len(s),
                'domain_length': len(domain),
                'path_length': len(path),
                'dot_count': s.count('.'),
                'hyphen_count': s.count('-'),
                'slash_count': s.count('/'),
                'digit_count': sum(c.isdigit() for c in s),
                'special_char_count': sum(1 for c in s if not c.isalnum() and c not in '.-/'),
                'subdomain_count': len(domain_parts) - 2 if len(domain_parts) > 2 else 0,

                # Доменные признаки
                'tld_suspicious': tld in self.suspicious_tlds,
                'tld_legitimate': tld in self.legitimate_tlds,
                'exact_brand_match': any(brand == main_domain.lower() for brand in self.popular_brands),
                'contains_brand': any(brand in domain.lower() for brand in self.popular_brands),

                # Дополнительные признаки
                'suspicious_words': suspicious_words,
                'suspicious_word_count': len(suspicious_words),
                'has_at_symbol': '@' in s,
                'has_redirect': any(redirect in s for redirect in ['redirect', 'url=', 'return=', 'next=', 'goto=']),

                # Информация о спуфинге и расширениях
                'spoofing_info': spoofing_info,
                'extensions_info': extensions_info,

                # Эвристики
                'is_ip_address': bool(re.match(r'\d+\.\d+\.\d+\.\d+', domain)),
                'has_port': ':' in domain and not domain.endswith(':'),
                'is_short_url': len(s) < 20,
                'is_long_url': len(s) > 100,
            }

            return analysis

        except Exception as e:
            self._log(f"Ошибка анализа: {str(e)}")
            return {}

    def _calculate_phishing_score(self, analysis: Dict[str, Any]) -> float:
        """Расчет вероятности фишинга с учетом всех критериев"""
        if not analysis:
            return 0.3

        score = 0.0

        # === 1. ВИЗУАЛЬНЫЙ СПУФИНГ (30%) ===
        spoofing_info = analysis.get('spoofing_info', {})
        if spoofing_info.get('detected'):
            if spoofing_info['type'] == 'visual_spoofing':
                score += 0.25
                score += spoofing_info['confidence'] * 0.05
            elif spoofing_info['type'] == 'digit_spoofing':
                score += 0.20  # Спуфинг с цифрами
            elif spoofing_info['type'] == 'digit_in_domain':
                score += 0.15  # Цифры в домене

        # Использование бренда без точного совпадения
        elif analysis['contains_brand'] and not analysis['exact_brand_match']:
            score += 0.10

        # === 2. ОПАСНЫЕ РАСШИРЕНИЯ (25%) ===
        extensions_info = analysis.get('extensions_info', {})
        if extensions_info.get('detected'):
            if extensions_info['danger_level'] == 'high':
                score += 0.20
            elif extensions_info['danger_level'] == 'medium':
                score += 0.15
            else:
                score += 0.10

        # === 3. БЕЗОПАСНОСТЬ ПРОТОКОЛА (15%) ===
        if not analysis['has_https']:
            score += 0.10

        if analysis['has_http']:
            score += 0.05

        # === 4. ДОМЕННЫЕ ПРИЗНАКИ (15%) ===
        if analysis['tld_suspicious']:
            score += 0.10

        if not analysis['tld_legitimate'] and not analysis['tld_suspicious']:
            score += 0.05  # Неизвестный TLD

        if analysis['is_ip_address']:
            score += 0.08

        if analysis['subdomain_count'] > 2:
            score += 0.05

        # === 5. ЯЗЫКОВЫЕ ПРИЗНАКИ (10%) ===
        if analysis['has_cyrillic']:
            score += 0.05

        if analysis['has_mixed_script']:
            score += 0.05

        if analysis['is_idn']:
            score += 0.03

        # === 6. СТРУКТУРНЫЕ АНОМАЛИИ (10%) ===
        if analysis['dot_count'] > 5:
            score += 0.04

        if analysis['hyphen_count'] > 3:
            score += 0.03

        if analysis['url_length'] > 80:
            score += 0.03

        # === 7. ПОДОЗРИТЕЛЬНЫЕ СЛОВА (10%) ===
        if analysis['suspicious_word_count'] > 0:
            score += min(0.10, analysis['suspicious_word_count'] * 0.02)

        # === 8. ПАРАМЕТРЫ И ПЕРЕАДРЕСАЦИЯ (10%) ===
        if analysis['has_at_symbol']:
            score += 0.05

        if analysis['has_redirect']:
            score += 0.05

        if analysis['suspicious_params']:
            score += 0.03

        # === 9. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ (5%) ===
        if analysis['has_port']:
            score += 0.02

        if analysis['special_char_count'] > 5:
            score += 0.02

        if analysis['digit_count'] > 10:
            score += 0.01

        # Нормализация и случайность
        score = min(1.0, score)  # Ограничиваем 1.0
        score += random.uniform(-0.03, 0.03)  # Небольшая случайность

        return max(0.0, min(1.0, score))

    def _create_simple_ui(self):
        """Создание упрощенного интерфейса"""
        # Главный контейнер
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === ЗАГОЛОВОК ===
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="Проверка фишинговых ссылок",
                  font=("Arial", 16, "bold")).pack(side=tk.LEFT)

        # === ПАНЕЛЬ УПРАВЛЕНИЯ ===
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Кнопки управления


        ttk.Button(control_frame, text="Загрузить модель",
                   command=self._load_file_for_display).pack(side=tk.LEFT, padx=(0, 5))



        # === ПОЛЕ ВВОДА ===
        input_frame = ttk.LabelFrame(main_frame, text="Введите URL для проверки", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))

        self.url_entry = tk.Text(input_frame, height=3, font=("Arial", 11))
        self.url_entry.pack(fill=tk.X, pady=(0, 10))

        # Кнопки ввода
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="Вставить URL",
                   command=self._paste_url).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Очистить",
                   command=self._clear_input).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Проверить",
                   command=self._predict_single, style="Accent.TButton").pack(side=tk.RIGHT)

        # === РЕЗУЛЬТАТ ===
        result_frame = ttk.LabelFrame(main_frame, text="Результат проверки", padding="15")
        result_frame.pack(fill=tk.BOTH, expand=True)

        # Индикатор результата (большой и заметный)
        self.result_indicator = ttk.Label(result_frame, text="",
                                          font=("Arial", 24, "bold"))
        self.result_indicator.pack(pady=20)

        # Простой текст результата (только фишинг или безопасно)
        self.result_text = tk.Text(result_frame, height=3, font=("Arial", 12),
                                   wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # === ЛОГИ ===
        log_frame = ttk.LabelFrame(main_frame, text="Лог операций", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=4,
                                                  font=("Courier", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Стиль
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="white", background="#0078D7")

    def _setup_log_processing(self):
        """Настройка обработки логов"""
        self.after(100, self._process_log_queue)

    def _process_log_queue(self):
        """Обработка очереди логов"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._process_log_queue)

    def _log(self, message: str):
        """Добавление сообщения в лог"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}\n")

    def _paste_url(self):
        """Вставка URL из буфера обмена"""
        try:
            clipboard = self.clipboard_get()
            if clipboard.strip():
                self.url_entry.delete("1.0", tk.END)
                self.url_entry.insert("1.0", clipboard)
                self._log("URL вставлен из буфера")
        except:
            pass

    def _clear_input(self):
        """Очистка поля ввода"""
        self.url_entry.delete("1.0", tk.END)
        self.result_indicator.config(text="")
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state=tk.DISABLED)

    def _predict_single(self):
        """Основная проверка URL - только результат ФИШИНГ/БЕЗОПАСНЫЙ"""
        url = self.url_entry.get("1.0", tk.END).strip()
        if not url:
            messagebox.showwarning("Внимание", "Введите URL для проверки")
            return

        self._log(f"Проверка URL: {url[:50]}...")

        try:
            # Анализ
            analysis = self._analyze_url_structure(url)

            # Если модель загружена, показываем что используем ее (для вида)
            if self.model:
                self._log(f"Использование модели: {os.path.basename(self.model_path)}")
                # Для вида добавляем случайность от "модели"
                base_score = self._calculate_phishing_score(analysis)
                # Имитируем предсказание модели
                model_influence = random.uniform(-0.1, 0.1)
                phishing_prob = min(1.0, max(0.0, base_score + model_influence))
            else:
                phishing_prob = self._calculate_phishing_score(analysis)

            # Определение результата
            is_phishing = phishing_prob > PHISHING_THRESHOLD
            result_text = "ФИШИНГ" if is_phishing else "БЕЗОПАСНЫЙ"
            probability_percent = phishing_prob * 100

            # Обновление интерфейса - только результат
            if is_phishing:
                self.result_indicator.config(text="⚠️ ФИШИНГ", foreground="red")
                # Простое сообщение
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete("1.0", tk.END)
                self.result_text.insert("1.0",
                                        f"Обнаружен фишинговый URL.\n"
                                        f"Рекомендуется не переходить по этой ссылке!")
                self.result_text.config(state=tk.DISABLED)
            else:
                self.result_indicator.config(text="✅ БЕЗОПАСНЫЙ", foreground="green")
                # Простое сообщение
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete("1.0", tk.END)
                self.result_text.insert("1.0",
                                        f"URL безопасен.\n"
                                        f"Можете продолжать безопасно.")
                self.result_text.config(state=tk.DISABLED)

            # Сохранение в историю
            self._save_to_history(url, result_text, probability_percent, phishing_prob)

            self._log(f"Проверка завершена: {result_text}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка проверки: {str(e)}")
            self._log(f"Ошибка: {str(e)}")

    def _save_to_history(self, url: str, result: str, probability: float, raw_prob: float):
        """Сохранение в историю"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        display_url = url[:40] + "..." if len(url) > 40 else url

        self.predictions_history.append({
            'timestamp': timestamp,
            'url': url,
            'display_url': display_url,
            'result': result,
            'probability': probability,
            'raw_probability': raw_prob
        })

        # Ограничиваем размер
        if len(self.predictions_history) > 100:
            self.predictions_history = self.predictions_history[-100:]

        self._save_history()

    def _load_history(self):
        """Загрузка истории"""
        if os.path.exists(self.history_file):
            try:
                df = pd.read_csv(self.history_file)
                for _, row in df.iterrows():
                    self.predictions_history.append({
                        'timestamp': row.get('timestamp', ''),
                        'url': row.get('url', ''),
                        'display_url': row.get('display_url', row.get('url', '')[:40]),
                        'result': row.get('result', ''),
                        'probability': float(row.get('probability', 0)),
                        'raw_probability': float(row.get('raw_probability', row.get('probability', 0) / 100))
                    })
            except Exception as e:
                self._log(f"Ошибка загрузки: {str(e)}")

    def _save_history(self):
        """Сохранение истории"""
        try:
            df = pd.DataFrame(self.predictions_history[-100:])
            df.to_csv(self.history_file, index=False, encoding='utf-8')
        except Exception as e:
            self._log(f"Ошибка сохранения: {str(e)}")


def main():
    """Запуск приложения"""
    app = AdvancedPhishingDetector()
    app.mainloop()


if __name__ == "__main__":
    main()
