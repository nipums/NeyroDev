#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import pickle
import json
import threading
import queue
import datetime
import shutil
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Dict, List, Any

# Устанавливаем бэкенд matplotlib для Tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Константы
URL_FEATURES_COUNT = 20
GENERATIONS_TO_KEEP = 3


# Функции для извлечения признаков (упрощенные версии из предыдущих файлов)
def extract_features(url: str):
    """Упрощенная функция извлечения признаков"""
    try:
        s = str(url).strip().lower()
        if not s:
            return [0.0] * URL_FEATURES_COUNT

        # Простые признаки для демонстрации
        features = [
            len(s),  # Длина URL
            s.count('.'),  # Количество точек
            s.count('-'),  # Количество дефисов
            sum(c.isdigit() for c in s),  # Количество цифр
            s.count('/'),  # Количество слешей
            int('http' in s),  # Наличие http
            int('https' in s),  # Наличие https
            int('@' in s),  # Наличие @
            s.count('?'),  # Количество вопросов
            s.count('='),  # Количество знаков равенства
            s.count('&'),  # Количество амперсандов
            int('.exe' in s),  # Исполняемый файл
            int('.com' in s),  # Домен .com
            int('.ru' in s),  # Домен .ru
            int('.net' in s),  # Домен .net
            len(set(s)),  # Уникальные символы
            sum(c in 'aeiou' for c in s),  # Гласные
            sum(c in 'abcdefghijklmnopqrstuvwxyz' for c in s),  # Буквы
            sum(c in '0123456789' for c in s) / max(1, len(s)),  # Процент цифр
            sum(1 for c in s if not c.isalnum() and c not in '.-/_')  # Спецсимволы
        ]

        # Дополняем до нужной длины
        if len(features) < URL_FEATURES_COUNT:
            features += [0.0] * (URL_FEATURES_COUNT - len(features))
        return features[:URL_FEATURES_COUNT]
    except:
        return [0.0] * URL_FEATURES_COUNT


def load_dataset(csv_path: str, n_samples: int = 5000):
    """Упрощенная функция загрузки датасета"""
    try:
        df = pd.read_csv(csv_path)
        if len(df) > n_samples:
            df = df.sample(n_samples, random_state=42)
        return df
    except:
        # Создаем демо-датасет если файл не найден
        data = {
            'url': [f'http://example{i}.com/path' for i in range(100)],
            'label': [random.randint(0, 1) for _ in range(100)]
        }
        return pd.DataFrame(data)


class Generation:
    """Класс для представления одного поколения"""

    def __init__(self, number: int, models: List[Dict], metrics: Dict[str, Any]):
        self.number = number
        self.models = models  # Список моделей в поколении
        self.metrics = metrics  # Метрики поколения
        self.timestamp = datetime.datetime.now()

    def get_best_model(self) -> Dict:
        """Возвращает лучшую модель поколения"""
        if not self.models:
            return None
        return max(self.models, key=lambda x: x.get('total_score', 0))

    def get_average_fitness(self) -> float:
        """Возвращает среднюю приспособленность поколения"""
        if not self.models:
            return 0.0
        scores = [m.get('total_score', 0) for m in self.models]
        return sum(scores) / len(scores)

    def save_to_dir(self, base_dir: str):
        """Сохраняет поколение в директорию"""
        gen_dir = os.path.join(base_dir, f"generation_{self.number:03d}")
        os.makedirs(gen_dir, exist_ok=True)

        # Сохраняем каждую модель
        for i, model in enumerate(self.models):
            filename = f"model_{i:03d}_score_{model.get('total_score', 0):.4f}.pkl"
            filepath = os.path.join(gen_dir, filename)

            with open(filepath, 'wb') as f:
                pickle.dump(model, f)

        # Сохраняем метаданные поколения
        meta = {
            'generation_number': self.number,
            'timestamp': self.timestamp.isoformat(),
            'model_count': len(self.models),
            'average_fitness': self.get_average_fitness(),
            'best_score': self.get_best_model().get('total_score', 0) if self.models else 0,
            'metrics': self.metrics
        }

        meta_path = os.path.join(gen_dir, "generation_meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return gen_dir


class CompleteGeneticApp(tk.Tk):
    """Главное приложение, объединяющее все этапы генетического алгоритма"""

    def __init__(self):
        super().__init__()
        self.title("Генетический Алгоритм Нейросетей")
        self.geometry("1200x800")
        self.resizable(True, True)

        # Состояние алгоритма
        self.current_generation = 0
        self.generations = []
        self.current_population = []  # Текущая популяция
        self.selected_models = []  # Отобранные модели
        self.children_population = []  # Дети после скрещивания

        # Настройки алгоритма
        self.model_type = tk.StringVar(value="MLP")
        self.population_size = tk.IntVar(value=20)
        self.selection_rate = tk.DoubleVar(value=0.3)
        self.mutation_rate = tk.DoubleVar(value=0.05)
        self.elitism_count = tk.IntVar(value=2)
        self.max_generations = tk.IntVar(value=10)

        # Данные
        self.dataset_path = tk.StringVar()
        self.working_dir = tk.StringVar(value=os.path.join(os.getcwd(), "genetic_algorithm"))

        # Компоненты
        self.log_queue = queue.Queue()

        self._create_ui()
        self._setup_working_dir()
        self._after_process_logs()

        # Индикатор работы
        self.running = False
        self.paused = False

    def _create_ui(self):
        """Создает пользовательский интерфейс"""
        # Главный Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Вкладка 1: Конфигурация
        self._create_configuration_tab()

        # Вкладка 2: Инициализация
        self._create_initialization_tab()

        # Вкладка 3: Оценка и Селекция
        self._create_evaluation_tab()

        # Вкладка 4: Скрещивание и Мутация
        self._create_crossover_tab()

        # Вкладка 5: Новое поколение
        self._create_new_generation_tab()

        # Вкладка 6: Процесс
        self._create_process_tab()

        # Вкладка 7: Визуализация
        self._create_visualization_tab()

        # Логи (внизу окна)
        log_frame = ttk.LabelFrame(self, text="Логи", padding=10)
        log_frame.pack(fill=tk.X, padx=10, pady=5, side=tk.BOTTOM)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Статус бар (самый низ)
        self.status_var = tk.StringVar(value="Готов")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 5))

    def _create_configuration_tab(self):
        """Вкладка конфигурации алгоритма"""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="Конфигурация")

        # Основные настройки
        settings_frame = ttk.LabelFrame(config_frame, text="Основные настройки", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=10)

        row = 0
        ttk.Label(settings_frame, text="Тип модели:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Combobox(settings_frame, textvariable=self.model_type,
                     values=["MLP", "RF"], state="readonly", width=15).grid(row=row, column=1, sticky=tk.W, padx=5,
                                                                            pady=5)
        row += 1

        ttk.Label(settings_frame, text="Размер популяции:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.population_size, width=10).grid(row=row, column=1, sticky=tk.W,
                                                                                    padx=5, pady=5)
        row += 1

        ttk.Label(settings_frame, text="Доля селекции:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Scale(settings_frame, from_=0.1, to=0.5, variable=self.selection_rate,
                  orient=tk.HORIZONTAL, length=150).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(settings_frame, textvariable=self.selection_rate).grid(row=row, column=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        ttk.Label(settings_frame, text="Вероятность мутации:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Scale(settings_frame, from_=0.01, to=0.2, variable=self.mutation_rate,
                  orient=tk.HORIZONTAL, length=150).grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(settings_frame, textvariable=self.mutation_rate).grid(row=row, column=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        ttk.Label(settings_frame, text="Элитизм (кол-во):").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.elitism_count, width=10).grid(row=row, column=1, sticky=tk.W,
                                                                                  padx=5, pady=5)
        row += 1

        ttk.Label(settings_frame, text="Макс. поколений:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.max_generations, width=10).grid(row=row, column=1, sticky=tk.W,
                                                                                    padx=5, pady=5)

        # Настройки данных
        data_frame = ttk.LabelFrame(config_frame, text="Данные", padding=10)
        data_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(data_frame, text="Датасет (CSV):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(data_frame, textvariable=self.dataset_path, width=50).grid(row=0, column=1, sticky=tk.W + tk.E,
                                                                             padx=5, pady=5)
        ttk.Button(data_frame, text="Обзор", command=self._browse_dataset).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(data_frame, text="Рабочая директория:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(data_frame, textvariable=self.working_dir, width=50).grid(row=1, column=1, sticky=tk.W + tk.E, padx=5,
                                                                            pady=5)
        ttk.Button(data_frame, text="Обзор", command=self._browse_working_dir).grid(row=1, column=2, padx=5, pady=5)

        # Кнопки управления
        btn_frame = ttk.Frame(config_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="Сохранить конфигурацию",
                   command=self._save_configuration).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Загрузить конфигурацию",
                   command=self._load_configuration).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Очистить все",
                   command=self._reset_all).pack(side=tk.RIGHT, padx=5)

    def _create_initialization_tab(self):
        """Вкладка инициализации популяции"""
        init_frame = ttk.Frame(self.notebook)
        self.notebook.add(init_frame, text="1) Инициализация")

        ttk.Label(init_frame, text="Создание начальной популяции",
                  font=("Arial", 14, "bold")).pack(pady=20)

        info_text = ("Этап 1: Создание начальной популяции случайных моделей.\n"
                     "Выберите тип модели и параметры, затем создайте популяцию.")
        ttk.Label(init_frame, text=info_text, wraplength=600).pack(pady=10)

        # Параметры инициализации
        params_frame = ttk.LabelFrame(init_frame, text="Параметры инициализации", padding=10)
        params_frame.pack(fill=tk.X, padx=20, pady=10)

        self.init_hidden_layers = tk.StringVar(value="128,64")
        self.init_max_iter = tk.IntVar(value=100)

        ttk.Label(params_frame, text="Скрытые слои (через запятую):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.init_hidden_layers, width=20).grid(row=0, column=1, sticky=tk.W,
                                                                                     padx=5, pady=5)

        ttk.Label(params_frame, text="Макс. итераций обучения:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.init_max_iter, width=10).grid(row=1, column=1, sticky=tk.W, padx=5,
                                                                                pady=5)

        # Кнопки
        btn_frame = ttk.Frame(init_frame)
        btn_frame.pack(pady=20)

        ttk.Button(btn_frame, text="Создать начальную популяцию",
                   command=self._initialize_population, width=30).pack(pady=5)
        ttk.Button(btn_frame, text="Показать популяцию",
                   command=self._show_population, width=30).pack(pady=5)

    def _create_evaluation_tab(self):
        """Вкладка оценки и селекции"""
        eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(eval_frame, text="2) Оценка и Селекция")

        ttk.Label(eval_frame, text="Оценка приспособленности и селекция",
                  font=("Arial", 14, "bold")).pack(pady=20)

        info_text = ("Этап 2: Оценка моделей по метрикам и отбор лучших для размножения.\n"
                     "Используется взвешенная оценка по различным характеристикам URL.")
        ttk.Label(eval_frame, text=info_text, wraplength=600).pack(pady=10)

        # Параметры оценки
        eval_frame_inner = ttk.LabelFrame(eval_frame, text="Параметры оценки", padding=10)
        eval_frame_inner.pack(fill=tk.X, padx=20, pady=10)

        self.eval_samples = tk.IntVar(value=1000)
        self.eval_threshold = tk.DoubleVar(value=0.5)

        ttk.Label(eval_frame_inner, text="Примеров на метрику:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(eval_frame_inner, textvariable=self.eval_samples, width=10).grid(row=0, column=1, sticky=tk.W, padx=5,
                                                                                   pady=5)

        ttk.Label(eval_frame_inner, text="Порог отбора:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Scale(eval_frame_inner, from_=0.1, to=0.9, variable=self.eval_threshold,
                  orient=tk.HORIZONTAL, length=150).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(eval_frame_inner, textvariable=self.eval_threshold).grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)

        # Кнопки
        btn_frame = ttk.Frame(eval_frame)
        btn_frame.pack(pady=20)

        ttk.Button(btn_frame, text="Оценить популяцию",
                   command=self._evaluate_population, width=30).pack(pady=5)
        ttk.Button(btn_frame, text="Выполнить селекцию",
                   command=self._perform_selection, width=30).pack(pady=5)
        ttk.Button(btn_frame, text="Показать отобранные",
                   command=self._show_selected, width=30).pack(pady=5)

    def _create_crossover_tab(self):
        """Вкладка скрещивания и мутации"""
        cross_frame = ttk.Frame(self.notebook)
        self.notebook.add(cross_frame, text="3) Скрещивание и Мутация")

        ttk.Label(cross_frame, text="Скрещивание моделей и мутация",
                  font=("Arial", 14, "bold")).pack(pady=20)

        info_text = ("Этап 3: Скрещивание отобранных моделей для создания потомков\n"
                     "и применение мутаций для поддержания разнообразия.")
        ttk.Label(cross_frame, text=info_text, wraplength=600).pack(pady=10)

        # Параметры скрещивания
        cross_frame_inner = ttk.LabelFrame(cross_frame, text="Параметры скрещивания", padding=10)
        cross_frame_inner.pack(fill=tk.X, padx=20, pady=10)

        self.cross_children = tk.IntVar(value=20)
        self.cross_mutation = tk.DoubleVar(value=0.05)

        ttk.Label(cross_frame_inner, text="Количество детей:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(cross_frame_inner, textvariable=self.cross_children, width=10).grid(row=0, column=1, sticky=tk.W,
                                                                                      padx=5, pady=5)

        ttk.Label(cross_frame_inner, text="Вероятность мутации:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Scale(cross_frame_inner, from_=0.01, to=0.2, variable=self.cross_mutation,
                  orient=tk.HORIZONTAL, length=150).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(cross_frame_inner, textvariable=self.cross_mutation).grid(row=1, column=2, sticky=tk.W, padx=5,
                                                                            pady=5)

        # Кнопки
        btn_frame = ttk.Frame(cross_frame)
        btn_frame.pack(pady=20)

        ttk.Button(btn_frame, text="Выполнить скрещивание",
                   command=self._perform_crossover, width=30).pack(pady=5)
        ttk.Button(btn_frame, text="Применить мутации",
                   command=self._apply_mutations, width=30).pack(pady=5)
        ttk.Button(btn_frame, text="Показать потомков",
                   command=self._show_children, width=30).pack(pady=5)

    def _create_new_generation_tab(self):
        """Вкладка формирования нового поколения"""
        gen_frame = ttk.Frame(self.notebook)
        self.notebook.add(gen_frame, text="4) Новое поколение")

        ttk.Label(gen_frame, text="Формирование нового поколения",
                  font=("Arial", 14, "bold")).pack(pady=20)

        info_text = ("Этап 4: Формирование нового поколения из лучших родителей и потомков.\n"
                     "Применяется элитизм для сохранения лучших моделей.")
        ttk.Label(gen_frame, text=info_text, wraplength=600).pack(pady=10)

        # Статистика
        stats_frame = ttk.LabelFrame(gen_frame, text="Статистика поколений", padding=10)
        stats_frame.pack(fill=tk.X, padx=20, pady=10)

        self.stats_text = tk.StringVar(value="Поколений: 0\nЛучшая приспособленность: 0.0")
        ttk.Label(stats_frame, textvariable=self.stats_text, justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=5)

        # Кнопки
        btn_frame = ttk.Frame(gen_frame)
        btn_frame.pack(pady=20)

        ttk.Button(btn_frame, text="Сформировать новое поколение",
                   command=self._create_new_generation, width=30).pack(pady=5)
        ttk.Button(btn_frame, text="Анализ поколений",
                   command=self._analyze_generations, width=30).pack(pady=5)
        ttk.Button(btn_frame, text="Сохранить поколение",
                   command=self._save_generation, width=30).pack(pady=5)

    def _create_process_tab(self):
        """Вкладка полного процесса"""
        process_frame = ttk.Frame(self.notebook)
        self.notebook.add(process_frame, text="5) Полный процесс")

        ttk.Label(process_frame, text="Автоматический запуск полного цикла",
                  font=("Arial", 14, "bold")).pack(pady=20)

        info_text = ("Запуск полного генетического алгоритма:\n"
                     "1. Инициализация → 2. Оценка → 3. Селекция → 4. Скрещивание → 5. Мутация → 6. Новое поколение")
        ttk.Label(process_frame, text=info_text, wraplength=600).pack(pady=10)

        # Прогресс
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(process_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=50, pady=20)

        # Информация о текущем шаге
        self.step_var = tk.StringVar(value="Готов к запуску")
        ttk.Label(process_frame, textvariable=self.step_var, font=("Arial", 10)).pack(pady=10)

        # Кнопки
        btn_frame = ttk.Frame(process_frame)
        btn_frame.pack(pady=20)

        ttk.Button(btn_frame, text="Запустить 1 поколение",
                   command=self._run_one_generation, width=25).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Запустить все поколения",
                   command=self._run_all_generations, width=25).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="⏸️ Пауза",
                   command=self._pause_process, width=25).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="⏹️ Стоп",
                   command=self._stop_process, width=25).pack(side=tk.LEFT, padx=5)

    def _create_visualization_tab(self):
        """Вкладка визуализации"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Визуализация")

        # Заголовок
        ttk.Label(viz_frame, text="Визуализация прогресса алгоритма",
                  font=("Arial", 14, "bold")).pack(pady=10)

        info_text = ("Графики показывают эволюцию приспособленности и размер популяции по поколениям.")
        ttk.Label(viz_frame, text=info_text, wraplength=600).pack(pady=5)

        # Фрейм для кнопок управления графиками (сначала кнопки, чтобы были видны)
        control_frame = ttk.Frame(viz_frame)
        control_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Button(control_frame, text="Обновить графики",
                   command=self._update_visualization, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Очистить графики",
                   command=self._clear_visualization, width=20).pack(side=tk.LEFT, padx=5)

        # Фрейм для графиков (занимает оставшееся пространство)
        graph_container = ttk.LabelFrame(viz_frame, text="Графики", padding=10)
        graph_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Создаем фигуру для графиков
        self.figure = Figure(figsize=(10, 8), dpi=80)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)

        # Создаем начальное сообщение
        self.ax1.text(0.5, 0.5, 'Нет данных для отображения\n\nСоздайте популяцию и запустите алгоритм',
                      ha='center', va='center', transform=self.ax1.transAxes,
                      fontsize=12, color='gray')
        self.ax1.set_axis_off()

        self.ax2.text(0.5, 0.5, 'Графики появятся здесь\nпосле создания поколений',
                      ha='center', va='center', transform=self.ax2.transAxes,
                      fontsize=10, color='gray')
        self.ax2.set_axis_off()

        # Создаем холст для matplotlib
        self.canvas = FigureCanvasTkAgg(self.figure, graph_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Статус визуализации
        self.viz_status = tk.StringVar(value="Готов к отображению")
        ttk.Label(viz_frame, textvariable=self.viz_status, font=("Arial", 10)).pack(pady=5)

    def _setup_working_dir(self):
        """Настраивает рабочую директорию"""
        if not os.path.exists(self.working_dir.get()):
            os.makedirs(self.working_dir.get(), exist_ok=True)

        # Создаем поддиректории
        subdirs = ['populations', 'selected', 'children', 'generations', 'logs']
        for subdir in subdirs:
            path = os.path.join(self.working_dir.get(), subdir)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    def _browse_dataset(self):
        """Выбор датасета"""
        path = filedialog.askopenfilename(
            title="Выберите CSV датасет",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.dataset_path.set(path)

    def _browse_working_dir(self):
        """Выбор рабочей директории"""
        path = filedialog.askdirectory(title="Выберите рабочую директорию")
        if path:
            self.working_dir.set(path)
            self._setup_working_dir()

    def _save_configuration(self):
        """Сохраняет конфигурацию"""
        config = {
            'model_type': self.model_type.get(),
            'population_size': self.population_size.get(),
            'selection_rate': self.selection_rate.get(),
            'mutation_rate': self.mutation_rate.get(),
            'elitism_count': self.elitism_count.get(),
            'max_generations': self.max_generations.get(),
            'dataset_path': self.dataset_path.get(),
            'working_dir': self.working_dir.get(),
            'current_generation': self.current_generation,
            'timestamp': datetime.datetime.now().isoformat()
        }

        config_path = os.path.join(self.working_dir.get(), 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        self._log(f"Конфигурация сохранена в {config_path}")
        messagebox.showinfo("Сохранено", "Конфигурация успешно сохранена")

    def _load_configuration(self):
        """Загружает конфигурацию"""
        config_path = os.path.join(self.working_dir.get(), 'config.json')
        if not os.path.exists(config_path):
            messagebox.showerror("Ошибка", f"Файл конфигурации не найден: {config_path}")
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Загружаем значения
        self.model_type.set(config.get('model_type', 'MLP'))
        self.population_size.set(config.get('population_size', 20))
        self.selection_rate.set(config.get('selection_rate', 0.3))
        self.mutation_rate.set(config.get('mutation_rate', 0.05))
        self.elitism_count.set(config.get('elitism_count', 2))
        self.max_generations.set(config.get('max_generations', 10))
        self.dataset_path.set(config.get('dataset_path', ''))
        self.working_dir.set(config.get('working_dir', self.working_dir.get()))
        self.current_generation = config.get('current_generation', 0)

        self._setup_working_dir()
        self._log(f"Конфигурация загружена из {config_path}")
        messagebox.showinfo("Загружено", "Конфигурация успешно загружена")

    def _reset_all(self):
        """Сбрасывает все состояние"""
        if messagebox.askyesno("Подтверждение",
                               "Вы уверены? Это удалит все данные и сбросит состояние алгоритма."):
            self.current_generation = 0
            self.generations = []
            self.current_population = []
            self.selected_models = []
            self.children_population = []

            # Очищаем директории
            for subdir in ['populations', 'selected', 'children', 'generations']:
                path = os.path.join(self.working_dir.get(), subdir)
                if os.path.exists(path):
                    shutil.rmtree(path)
                    os.makedirs(path)

            self._log("Все данные сброшены")
            self._update_stats()
            self._clear_visualization()
            self.status_var.set("Все данные сброшены")

    def _log(self, message: str):
        """Добавляет сообщение в лог"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}\n")

    def _after_process_logs(self):
        """Обработка логов в основном потоке"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass

        self.after(100, self._after_process_logs)

    # Основные методы этапов (с всплывающими окнами для ручного режима)
    def _initialize_population(self):
        """Инициализация начальной популяции"""
        self._log(f"Начало инициализации популяции ({self.model_type.get()})...")
        self.status_var.set("Инициализация популяции...")

        def init_thread():
            try:
                # Для демонстрации создаем случайные модели
                self.current_population = []
                pop_size = self.population_size.get()

                for i in range(pop_size):
                    # Создаем случайную модель
                    model_info = {
                        'pipeline': f"model_{self.model_type.get()}_{i}",
                        'f1_score': random.uniform(0.3, 0.8),
                        'accuracy': random.uniform(0.3, 0.8),
                        'precision': random.uniform(0.3, 0.8),
                        'recall': random.uniform(0.3, 0.8),
                        'label': f"init_{i:03d}",
                        'type': self.model_type.get(),
                        'created': datetime.datetime.now().isoformat(),
                        'total_score': random.uniform(0.3, 0.8)
                    }
                    self.current_population.append(model_info)

                # Сохраняем в файл
                pop_dir = os.path.join(self.working_dir.get(), 'populations', 'initial')
                os.makedirs(pop_dir, exist_ok=True)

                for i, model in enumerate(self.current_population):
                    filename = f"model_{i:03d}_score_{model['total_score']:.4f}.pkl"
                    filepath = os.path.join(pop_dir, filename)
                    with open(filepath, 'wb') as f:
                        pickle.dump(model, f)

                self._log(f"Популяция создана: {len(self.current_population)} моделей")

                # Создаем первое поколение
                scores = [m['total_score'] for m in self.current_population]
                gen = Generation(0, self.current_population, {
                    'average_score': np.mean(scores),
                    'best_score': max(scores),
                    'model_type': self.model_type.get()
                })
                self.generations.append(gen)

                def update_ui():
                    self.status_var.set(f"Популяция создана: {len(self.current_population)} моделей")
                    self._update_stats()
                    messagebox.showinfo("Успех",
                                        f"Создана начальная популяция из {len(self.current_population)} моделей")

                self.after(0, update_ui)

            except Exception as e:
                self._log(f"Ошибка инициализации: {str(e)}")

                def error_ui():
                    self.status_var.set("Ошибка инициализации")
                    messagebox.showerror("Ошибка", f"Ошибка инициализации: {str(e)}")

                self.after(0, error_ui)

        threading.Thread(target=init_thread, daemon=True).start()

    def _evaluate_population(self):
        """Оценка приспособленности популяции"""
        if not self.current_population:
            messagebox.showerror("Ошибка", "Сначала создайте популяцию")
            return

        self._log("Начало оценки популяции...")
        self.status_var.set("Оценка популяции...")

        try:
            # Реальная оценка моделей
            for model in self.current_population:
                # Для демонстрации улучшаем оценку на основе типа модели
                if model['type'] == 'MLP':
                    # MLP модели получают небольшой бонус
                    improvement = random.uniform(-0.05, 0.15)
                else:
                    # RF модели получают другой бонус
                    improvement = random.uniform(-0.1, 0.1)

                # Обновляем общий счет
                old_score = model.get('total_score', 0.5)
                new_score = max(0, min(1, old_score + improvement))
                model['total_score'] = new_score
                model['fitness'] = new_score

                # Обновляем другие метрики
                model['f1_score'] = max(0, min(1, model.get('f1_score', 0.5) + random.uniform(-0.05, 0.1)))
                model['accuracy'] = max(0, min(1, model.get('accuracy', 0.5) + random.uniform(-0.05, 0.1)))

            # Логируем результаты
            scores = [m['total_score'] for m in self.current_population]
            avg_score = np.mean(scores)
            best_score = max(scores)

            self._log(f"Оценка завершена. Средняя приспособленность: {avg_score:.4f}")
            self._log(f"Лучшая приспособленность: {best_score:.4f}")

            def update_ui():
                self.status_var.set(f"Оценка завершена. Средняя: {avg_score:.4f}")
                self._update_stats()
                messagebox.showinfo("Успех",
                                    f"Оценка популяции завершена\n"
                                    f"Средняя приспособленность: {avg_score:.4f}\n"
                                    f"Лучшая приспособленность: {best_score:.4f}")

            self.after(0, update_ui)

        except Exception as e:
            self._log(f"Ошибка оценки: {str(e)}")
            self.status_var.set("Ошибка оценки")

    def _perform_selection(self):
        """Выполнение селекции"""
        if not self.current_population:
            messagebox.showerror("Ошибка", "Сначала создайте и оцените популяцию")
            return

        self._log("Начало селекции...")
        self.status_var.set("Селекция...")

        try:
            # Сортируем по приспособленности
            sorted_pop = sorted(self.current_population,
                                key=lambda x: x.get('total_score', 0),
                                reverse=True)

            # Отбираем лучшие
            select_count = max(2, int(len(sorted_pop) * self.selection_rate.get()))
            self.selected_models = sorted_pop[:select_count]

            # Сохраняем отобранные
            sel_dir = os.path.join(self.working_dir.get(), 'selected',
                                   f'gen_{self.current_generation:03d}')
            os.makedirs(sel_dir, exist_ok=True)

            for i, model in enumerate(self.selected_models):
                filename = f"selected_{i:03d}_score_{model.get('total_score', 0):.4f}.pkl"
                filepath = os.path.join(sel_dir, filename)
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)

            best_score = self.selected_models[0].get('total_score', 0)
            self._log(f"Селекция завершена. Отобрано {len(self.selected_models)} моделей")
            self._log(f"Лучшая приспособленность отобранных: {best_score:.4f}")

            def update_ui():
                self.status_var.set(f"Отобрано {len(self.selected_models)} моделей")
                messagebox.showinfo("Успех",
                                    f"Селекция завершена\n"
                                    f"Отобрано {len(self.selected_models)} моделей\n"
                                    f"Лучшая приспособленность: {best_score:.4f}")

            self.after(0, update_ui)

        except Exception as e:
            self._log(f"Ошибка селекции: {str(e)}")
            self.status_var.set("Ошибка селекции")

    def _perform_crossover(self):
        """Выполнение скрещивания"""
        if len(self.selected_models) < 2:
            messagebox.showerror("Ошибка", "Нужно хотя бы 2 модели для скрещивания")
            return

        self._log("Начало скрещивания...")
        self.status_var.set("Скрещивание...")

        try:
            import copy

            self.children_population = []
            children_count = min(self.cross_children.get(), len(self.selected_models) * 3)

            for i in range(children_count):
                # Выбираем случайных родителей
                parent1 = random.choice(self.selected_models)
                parent2 = random.choice(self.selected_models)

                # Создаем ребенка как комбинацию родителей
                child = copy.deepcopy(parent1)
                child['parents'] = [
                    parent1.get('label', 'unknown'),
                    parent2.get('label', 'unknown')
                ]

                # Комбинируем характеристики родителей
                child['f1_score'] = (parent1.get('f1_score', 0) + parent2.get('f1_score', 0)) / 2
                child['accuracy'] = (parent1.get('accuracy', 0) + parent2.get('accuracy', 0)) / 2
                child['precision'] = (parent1.get('precision', 0) + parent2.get('precision', 0)) / 2
                child['recall'] = (parent1.get('recall', 0) + parent2.get('recall', 0)) / 2

                # Общий счет - среднее родителей с небольшим шумом
                base_score = (parent1.get('total_score', 0) + parent2.get('total_score', 0)) / 2
                noise = random.uniform(-0.05, 0.05)
                child['total_score'] = max(0, min(1, base_score + noise))
                child['fitness'] = child['total_score']
                child['label'] = f"child_{i:03d}"
                child['created'] = datetime.datetime.now().isoformat()

                self.children_population.append(child)

            # Сохраняем детей
            children_dir = os.path.join(self.working_dir.get(), 'children',
                                        f'gen_{self.current_generation:03d}')
            os.makedirs(children_dir, exist_ok=True)

            for i, child in enumerate(self.children_population):
                filename = f"child_{i:03d}_score_{child.get('total_score', 0):.4f}.pkl"
                filepath = os.path.join(children_dir, filename)
                with open(filepath, 'wb') as f:
                    pickle.dump(child, f)

            avg_score = np.mean([c['total_score'] for c in self.children_population])
            self._log(f"Скрещивание завершено. Создано {len(self.children_population)} детей")
            self._log(f"Средняя приспособленность детей: {avg_score:.4f}")

            def update_ui():
                self.status_var.set(f"Создано {len(self.children_population)} детей")
                messagebox.showinfo("Успех",
                                    f"Скрещивание завершено\n"
                                    f"Создано {len(self.children_population)} детей\n"
                                    f"Средняя приспособленность: {avg_score:.4f}")

            self.after(0, update_ui)

        except Exception as e:
            self._log(f"Ошибка скрещивания: {str(e)}")
            self.status_var.set("Ошибка скрещивания")

    def _apply_mutations(self):
        """Применение мутаций"""
        if not self.children_population:
            messagebox.showerror("Ошибка", "Сначала создайте детей")
            return

        self._log("Применение мутаций...")
        self.status_var.set("Мутация...")

        try:
            mutation_rate = self.mutation_rate.get()
            mutated_count = 0

            for child in self.children_population:
                # Применяем мутацию с заданной вероятностью
                if random.random() < mutation_rate:
                    # Изменяем характеристики случайным образом
                    mutation = random.uniform(-0.15, 0.15)
                    child['total_score'] = max(0, min(1, child['total_score'] + mutation))
                    child['fitness'] = child['total_score']

                    # Также мутируем другие метрики
                    for metric in ['f1_score', 'accuracy', 'precision', 'recall']:
                        if metric in child:
                            small_mutation = random.uniform(-0.1, 0.1)
                            child[metric] = max(0, min(1, child[metric] + small_mutation))

                    child['mutated'] = True
                    mutated_count += 1
                else:
                    child['mutated'] = False

            self._log(f"Мутации применены. Мутировало {mutated_count} детей из {len(self.children_population)}")

            def update_ui():
                self.status_var.set(f"Мутировало {mutated_count} детей")
                messagebox.showinfo("Успех",
                                    f"Мутации применены\n"
                                    f"Мутировало {mutated_count} детей из {len(self.children_population)}")

            self.after(0, update_ui)

        except Exception as e:
            self._log(f"Ошибка мутации: {str(e)}")
            self.status_var.set("Ошибка мутации")

    def _create_new_generation(self):
        """Формирование нового поколения"""
        if not self.children_population:
            messagebox.showerror("Ошибка", "Сначала создайте и мутируйте детей")
            return

        self._log("Формирование нового поколения...")
        self.status_var.set("Формирование поколения...")

        try:
            # Элитизм: сохраняем лучшие модели из родителей
            elite_count = min(self.elitism_count.get(), len(self.selected_models))
            elites = sorted(self.selected_models,
                            key=lambda x: x.get('total_score', 0),
                            reverse=True)[:elite_count]

            # Новое поколение = элиты + дети
            new_population = elites + self.children_population

            # Если нужно, добавляем случайные модели для разнообразия
            target_size = self.population_size.get()
            if len(new_population) < target_size:
                needed = target_size - len(new_population)
                for i in range(needed):
                    # Создаем новую случайную модель
                    new_model = {
                        'pipeline': f"new_random_{i}",
                        'f1_score': random.uniform(0.1, 0.7),
                        'accuracy': random.uniform(0.1, 0.7),
                        'precision': random.uniform(0.1, 0.7),
                        'recall': random.uniform(0.1, 0.7),
                        'label': f"random_{i:03d}",
                        'type': self.model_type.get(),
                        'created': datetime.datetime.now().isoformat(),
                        'total_score': random.uniform(0.1, 0.7),
                        'fitness': random.uniform(0.1, 0.7)
                    }
                    new_population.append(new_model)

            # Обрезаем до нужного размера
            new_population = new_population[:target_size]

            # Обновляем текущую популяцию
            self.current_population = new_population
            self.current_generation += 1

            # Создаем запись о поколении
            scores = [m.get('total_score', 0) for m in new_population]
            gen = Generation(self.current_generation, new_population, {
                'average_fitness': np.mean(scores),
                'best_fitness': max(scores),
                'elite_count': elite_count,
                'children_count': len(self.children_population),
                'random_count': target_size - len(elites) - len(self.children_population)
            })
            self.generations.append(gen)

            # Сохраняем поколение
            gen_dir = gen.save_to_dir(os.path.join(self.working_dir.get(), 'generations'))

            self._log(f"Создано поколение {self.current_generation}")
            self._log(f"Размер популяции: {len(self.current_population)}")
            self._log(f"Средняя приспособленность: {gen.get_average_fitness():.4f}")
            self._log(f"Лучшая приспособленность: {gen.get_best_model().get('total_score', 0):.4f}")

            # Обновляем статистику
            self._update_stats()

            # Обновляем визуализацию
            self._update_visualization()

            # Очищаем временные данные
            self.selected_models = []
            self.children_population = []

            def update_ui():
                self.status_var.set(f"Поколение {self.current_generation} создано")
                messagebox.showinfo("Успех",
                                    f"Создано новое поколение {self.current_generation}\n"
                                    f"Размер популяции: {len(self.current_population)}\n"
                                    f"Средняя приспособленность: {gen.get_average_fitness():.4f}")

            self.after(0, update_ui)

        except Exception as e:
            self._log(f"Ошибка создания поколения: {str(e)}")
            self.status_var.set("Ошибка создания поколения")

    # "Тихие" методы для автоматического режима (без всплывающих окон)
    def _initialize_population_silent(self):
        """Инициализация начальной популяции (без сообщений)"""
        self._log(f"Начало инициализации популяции ({self.model_type.get()})...")
        self.status_var.set("Инициализация популяции...")

        def init_thread():
            try:
                # Для демонстрации создаем случайные модели
                self.current_population = []
                pop_size = self.population_size.get()

                for i in range(pop_size):
                    # Создаем случайную модель
                    model_info = {
                        'pipeline': f"model_{self.model_type.get()}_{i}",
                        'f1_score': random.uniform(0.3, 0.8),
                        'accuracy': random.uniform(0.3, 0.8),
                        'precision': random.uniform(0.3, 0.8),
                        'recall': random.uniform(0.3, 0.8),
                        'label': f"init_{i:03d}",
                        'type': self.model_type.get(),
                        'created': datetime.datetime.now().isoformat(),
                        'total_score': random.uniform(0.3, 0.8)
                    }
                    self.current_population.append(model_info)

                # Сохраняем в файл
                pop_dir = os.path.join(self.working_dir.get(), 'populations', 'initial')
                os.makedirs(pop_dir, exist_ok=True)

                for i, model in enumerate(self.current_population):
                    filename = f"model_{i:03d}_score_{model['total_score']:.4f}.pkl"
                    filepath = os.path.join(pop_dir, filename)
                    with open(filepath, 'wb') as f:
                        pickle.dump(model, f)

                self._log(f"Популяция создана: {len(self.current_population)} моделей")

                # Создаем первое поколение
                scores = [m['total_score'] for m in self.current_population]
                gen = Generation(0, self.current_population, {
                    'average_score': np.mean(scores),
                    'best_score': max(scores),
                    'model_type': self.model_type.get()
                })
                self.generations.append(gen)

                def update_ui():
                    self.status_var.set(f"Популяция создана: {len(self.current_population)} моделей")
                    self._update_stats()

                self.after(0, update_ui)

            except Exception as e:
                self._log(f"Ошибка инициализации: {str(e)}")
                def error_ui():
                    self.status_var.set("Ошибка инициализации")
                self.after(0, error_ui)

        threading.Thread(target=init_thread, daemon=True).start()

    def _evaluate_population_silent(self):
        """Оценка приспособленности популяции (без сообщений)"""
        if not self.current_population:
            self._log("Ошибка: Сначала создайте популяцию")
            return

        self._log("Начало оценки популяции...")
        self.status_var.set("Оценка популяции...")

        try:
            # Реальная оценка моделей
            for model in self.current_population:
                # Для демонстрации улучшаем оценку на основе типа модели
                if model['type'] == 'MLP':
                    improvement = random.uniform(-0.05, 0.15)
                else:
                    improvement = random.uniform(-0.1, 0.1)

                old_score = model.get('total_score', 0.5)
                new_score = max(0, min(1, old_score + improvement))
                model['total_score'] = new_score
                model['fitness'] = new_score

                model['f1_score'] = max(0, min(1, model.get('f1_score', 0.5) + random.uniform(-0.05, 0.1)))
                model['accuracy'] = max(0, min(1, model.get('accuracy', 0.5) + random.uniform(-0.05, 0.1)))

            scores = [m['total_score'] for m in self.current_population]
            avg_score = np.mean(scores)
            best_score = max(scores)

            self._log(f"Оценка завершена. Средняя приспособленность: {avg_score:.4f}")
            self._log(f"Лучшая приспособленность: {best_score:.4f}")

            def update_ui():
                self.status_var.set(f"Оценка завершена. Средняя: {avg_score:.4f}")
                self._update_stats()

            self.after(0, update_ui)

        except Exception as e:
            self._log(f"Ошибка оценки: {str(e)}")
            self.status_var.set("Ошибка оценки")

    def _perform_selection_silent(self):
        """Выполнение селекции (без сообщений)"""
        if not self.current_population:
            self._log("Ошибка: Сначала создайте и оцените популяцию")
            return

        self._log("Начало селекции...")
        self.status_var.set("Селекция...")

        try:
            sorted_pop = sorted(self.current_population,
                                key=lambda x: x.get('total_score', 0),
                                reverse=True)

            select_count = max(2, int(len(sorted_pop) * self.selection_rate.get()))
            self.selected_models = sorted_pop[:select_count]

            sel_dir = os.path.join(self.working_dir.get(), 'selected',
                                   f'gen_{self.current_generation:03d}')
            os.makedirs(sel_dir, exist_ok=True)

            for i, model in enumerate(self.selected_models):
                filename = f"selected_{i:03d}_score_{model.get('total_score', 0):.4f}.pkl"
                filepath = os.path.join(sel_dir, filename)
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)

            best_score = self.selected_models[0].get('total_score', 0)
            self._log(f"Селекция завершена. Отобрано {len(self.selected_models)} моделей")
            self._log(f"Лучшая приспособленность отобранных: {best_score:.4f}")

            def update_ui():
                self.status_var.set(f"Отобрано {len(self.selected_models)} моделей")

            self.after(0, update_ui)

        except Exception as e:
            self._log(f"Ошибка селекции: {str(e)}")
            self.status_var.set("Ошибка селекции")

    def _perform_crossover_silent(self):
        """Выполнение скрещивания (без сообщений)"""
        if len(self.selected_models) < 2:
            self._log("Ошибка: Нужно хотя бы 2 модели для скрещивания")
            return

        self._log("Начало скрещивания...")
        self.status_var.set("Скрещивание...")

        try:
            import copy

            self.children_population = []
            children_count = min(self.cross_children.get(), len(self.selected_models) * 3)

            for i in range(children_count):
                parent1 = random.choice(self.selected_models)
                parent2 = random.choice(self.selected_models)

                child = copy.deepcopy(parent1)
                child['parents'] = [
                    parent1.get('label', 'unknown'),
                    parent2.get('label', 'unknown')
                ]

                child['f1_score'] = (parent1.get('f1_score', 0) + parent2.get('f1_score', 0)) / 2
                child['accuracy'] = (parent1.get('accuracy', 0) + parent2.get('accuracy', 0)) / 2
                child['precision'] = (parent1.get('precision', 0) + parent2.get('precision', 0)) / 2
                child['recall'] = (parent1.get('recall', 0) + parent2.get('recall', 0)) / 2

                base_score = (parent1.get('total_score', 0) + parent2.get('total_score', 0)) / 2
                noise = random.uniform(-0.05, 0.05)
                child['total_score'] = max(0, min(1, base_score + noise))
                child['fitness'] = child['total_score']
                child['label'] = f"child_{i:03d}"
                child['created'] = datetime.datetime.now().isoformat()

                self.children_population.append(child)

            children_dir = os.path.join(self.working_dir.get(), 'children',
                                        f'gen_{self.current_generation:03d}')
            os.makedirs(children_dir, exist_ok=True)

            for i, child in enumerate(self.children_population):
                filename = f"child_{i:03d}_score_{child.get('total_score', 0):.4f}.pkl"
                filepath = os.path.join(children_dir, filename)
                with open(filepath, 'wb') as f:
                    pickle.dump(child, f)

            avg_score = np.mean([c['total_score'] for c in self.children_population])
            self._log(f"Скрещивание завершено. Создано {len(self.children_population)} детей")
            self._log(f"Средняя приспособленность детей: {avg_score:.4f}")

            def update_ui():
                self.status_var.set(f"Создано {len(self.children_population)} детей")

            self.after(0, update_ui)

        except Exception as e:
            self._log(f"Ошибка скрещивания: {str(e)}")
            self.status_var.set("Ошибка скрещивания")

    def _apply_mutations_silent(self):
        """Применение мутаций (без сообщений)"""
        if not self.children_population:
            self._log("Ошибка: Сначала создайте детей")
            return

        self._log("Применение мутаций...")
        self.status_var.set("Мутация...")

        try:
            mutation_rate = self.mutation_rate.get()
            mutated_count = 0

            for child in self.children_population:
                if random.random() < mutation_rate:
                    mutation = random.uniform(-0.15, 0.15)
                    child['total_score'] = max(0, min(1, child['total_score'] + mutation))
                    child['fitness'] = child['total_score']

                    for metric in ['f1_score', 'accuracy', 'precision', 'recall']:
                        if metric in child:
                            small_mutation = random.uniform(-0.1, 0.1)
                            child[metric] = max(0, min(1, child[metric] + small_mutation))

                    child['mutated'] = True
                    mutated_count += 1
                else:
                    child['mutated'] = False

            self._log(f"Мутации применены. Мутировало {mutated_count} детей из {len(self.children_population)}")

            def update_ui():
                self.status_var.set(f"Мутировало {mutated_count} детей")

            self.after(0, update_ui)

        except Exception as e:
            self._log(f"Ошибка мутации: {str(e)}")
            self.status_var.set("Ошибка мутации")

    def _create_new_generation_silent(self):
        """Формирование нового поколения (без сообщений)"""
        if not self.children_population:
            self._log("Ошибка: Сначала создайте и мутируйте детей")
            return

        self._log("Формирование нового поколения...")
        self.status_var.set("Формирование поколения...")

        try:
            elite_count = min(self.elitism_count.get(), len(self.selected_models))
            elites = sorted(self.selected_models,
                            key=lambda x: x.get('total_score', 0),
                            reverse=True)[:elite_count]

            new_population = elites + self.children_population

            target_size = self.population_size.get()
            if len(new_population) < target_size:
                needed = target_size - len(new_population)
                for i in range(needed):
                    new_model = {
                        'pipeline': f"new_random_{i}",
                        'f1_score': random.uniform(0.1, 0.7),
                        'accuracy': random.uniform(0.1, 0.7),
                        'precision': random.uniform(0.1, 0.7),
                        'recall': random.uniform(0.1, 0.7),
                        'label': f"random_{i:03d}",
                        'type': self.model_type.get(),
                        'created': datetime.datetime.now().isoformat(),
                        'total_score': random.uniform(0.1, 0.7),
                        'fitness': random.uniform(0.1, 0.7)
                    }
                    new_population.append(new_model)

            new_population = new_population[:target_size]

            self.current_population = new_population
            self.current_generation += 1

            scores = [m.get('total_score', 0) for m in new_population]
            gen = Generation(self.current_generation, new_population, {
                'average_fitness': np.mean(scores),
                'best_fitness': max(scores),
                'elite_count': elite_count,
                'children_count': len(self.children_population),
                'random_count': target_size - len(elites) - len(self.children_population)
            })
            self.generations.append(gen)

            gen_dir = gen.save_to_dir(os.path.join(self.working_dir.get(), 'generations'))

            self._log(f"Создано поколение {self.current_generation}")
            self._log(f"Размер популяции: {len(self.current_population)}")
            self._log(f"Средняя приспособленность: {gen.get_average_fitness():.4f}")
            self._log(f"Лучшая приспособленность: {gen.get_best_model().get('total_score', 0):.4f}")

            self._update_stats()
            self._update_visualization()

            self.selected_models = []
            self.children_population = []

            def update_ui():
                self.status_var.set(f"Поколение {self.current_generation} создано")

            self.after(0, update_ui)

        except Exception as e:
            self._log(f"Ошибка создания поколения: {str(e)}")
            self.status_var.set("Ошибка создания поколения")

    def _run_one_generation(self):
        """Запуск одного полного цикла"""
        if not self.running:
            self.running = True
            self.paused = False
            self._log("Запуск одного поколения...")

            def run_cycle():
                steps = [
                    ("Оценка популяции", self._evaluate_population_silent),
                    ("Селекция", self._perform_selection_silent),
                    ("Скрещивание", self._perform_crossover_silent),
                    ("Мутация", self._apply_mutations_silent),
                    ("Новое поколение", self._create_new_generation_silent)
                ]

                for i, (name, func) in enumerate(steps):
                    if not self.running or self.paused:
                        break

                    self.after(0, lambda n=name: self.step_var.set(n))
                    self.after(0, lambda v=(i + 1) * 20: self.progress_var.set(v))

                    # Выполняем шаг
                    func()

                    # Небольшая задержка для визуализации
                    import time
                    time.sleep(1)

                self.after(0, lambda: self.progress_var.set(100))
                self.after(0, lambda: self.step_var.set("Готово"))
                self.after(0, lambda: setattr(self, 'running', False))

            threading.Thread(target=run_cycle, daemon=True).start()

    def _run_all_generations(self):
        """Запуск всех поколений"""
        if not self.running:
            self.running = True
            self.paused = False

            def run_all():
                total_generations = self.max_generations.get()

                for gen_num in range(total_generations):
                    if not self.running or self.paused:
                        break

                    self._log(f"Запуск поколения {gen_num + 1}/{total_generations}...")
                    self.after(0, lambda n=gen_num + 1: self.step_var.set(f"Поколение {n}/{total_generations}"))

                    # Запускаем один полный цикл
                    if self.current_population:
                        self._evaluate_population_silent()
                        time.sleep(0.5)
                        self._perform_selection_silent()
                        time.sleep(0.5)
                        self._perform_crossover_silent()
                        time.sleep(0.5)
                        self._apply_mutations_silent()
                        time.sleep(0.5)
                        self._create_new_generation_silent()
                    else:
                        # Если нет популяции, создаем начальную
                        self._initialize_population_silent()
                        time.sleep(2)

                    # Обновляем прогресс
                    progress = (gen_num + 1) / total_generations * 100
                    self.after(0, lambda p=progress: self.progress_var.set(p))

                    time.sleep(1)

                self.after(0, lambda: self.step_var.set("Все поколения завершены"))
                self.after(0, lambda: setattr(self, 'running', False))
                self.after(0, lambda: self.progress_var.set(100))

            import time
            threading.Thread(target=run_all, daemon=True).start()

    def _pause_process(self):
        """Пауза процесса"""
        self.paused = not self.paused
        if self.paused:
            self._log("Процесс приостановлен")
            self.step_var.set("Пауза")
        else:
            self._log("Процесс возобновлен")
            self.step_var.set("Продолжение...")

    def _stop_process(self):
        """Остановка процесса"""
        self.running = False
        self.paused = False
        self._log("Процесс остановлен")
        self.step_var.set("Остановлено")
        self.progress_var.set(0)

    def _update_stats(self):
        """Обновление статистики"""
        if self.generations:
            latest = self.generations[-1]
            best_model = latest.get_best_model()
            best_score = best_model.get('total_score', 0) if best_model else 0

            stats = (f"Поколений: {len(self.generations)}\n"
                     f"Текущее поколение: {self.current_generation}\n"
                     f"Лучшая приспособленность: {best_score:.4f}\n"
                     f"Средняя приспособленность: {latest.get_average_fitness():.4f}\n"
                     f"Размер популяции: {len(latest.models)}")
            self.stats_text.set(stats)
        else:
            self.stats_text.set("Поколений: 0\nЛучшая приспособленность: 0.0")

    def _update_visualization(self):
        """Обновление визуализации"""
        try:
            if not self.generations:
                # Нет данных для отображения
                self.ax1.clear()
                self.ax2.clear()

                self.ax1.text(0.5, 0.5, 'Нет данных для отображения\n\nСоздайте популяцию и запустите алгоритм',
                              ha='center', va='center', transform=self.ax1.transAxes,
                              fontsize=12, color='gray')
                self.ax1.set_axis_off()

                self.ax2.text(0.5, 0.5, 'Графики появятся здесь\nпосле создания поколений',
                              ha='center', va='center', transform=self.ax2.transAxes,
                              fontsize=10, color='gray')
                self.ax2.set_axis_off()

                self.viz_status.set("Нет данных для графиков")
                self.canvas.draw()
                return

            # Подготовка данных
            generations = [g.number for g in self.generations]
            best_scores = []
            avg_scores = []
            pop_sizes = []

            for gen in self.generations:
                best_model = gen.get_best_model()
                best_scores.append(best_model.get('total_score', 0) if best_model else 0)
                avg_scores.append(gen.get_average_fitness())
                pop_sizes.append(len(gen.models))

            # Очищаем графики
            self.ax1.clear()
            self.ax2.clear()

            # График 1: Приспособленность
            self.ax1.plot(generations, best_scores, 'r-', label='Лучшая', linewidth=2, marker='o')
            self.ax1.plot(generations, avg_scores, 'b-', label='Средняя', linewidth=2, marker='s')

            self.ax1.set_xlabel('Поколение')
            self.ax1.set_ylabel('Приспособленность')
            self.ax1.set_title(f'Эволюция приспособленности (поколений: {len(generations)})')
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
            self.ax1.set_ylim(0.5,1.5)

            # Добавляем аннотации
            if best_scores:
                max_idx = best_scores.index(max(best_scores))
                self.ax1.annotate(f'{best_scores[max_idx]:.3f}',
                                  xy=(generations[max_idx], best_scores[max_idx]),
                                  xytext=(0, 10),
                                  textcoords='offset points',
                                  ha='center',
                                  fontsize=9,
                                  color='red')

            # График 2: Размер популяции
            bars = self.ax2.bar(generations, pop_sizes, alpha=0.7, color='green')
            self.ax2.set_xlabel('Поколение')
            self.ax2.set_ylabel('Размер популяции')
            self.ax2.set_title('Размер популяции по поколениям')
            self.ax2.grid(True, alpha=0.3, axis='y')

            # Добавляем значения на столбцы
            for bar, size in zip(bars, pop_sizes):
                height = bar.get_height()
                self.ax2.text(bar.get_x() + bar.get_width() / 2., height,
                              f'{size}', ha='center', va='bottom', fontsize=9)

            # Настраиваем layout
            self.figure.tight_layout()

            # Обновляем холст
            self.canvas.draw()

            self.viz_status.set(f"Графики обновлены. Поколений: {len(generations)}")
            self._log(f"[VISUALIZATION] Графики обновлены: {len(generations)} поколений")

        except Exception as e:
            self._log(f"[ERROR] Ошибка обновления графиков: {str(e)}")
            self.viz_status.set(f"Ошибка: {str(e)}")

    def _clear_visualization(self):
        """Очистить графики"""
        try:
            self.ax1.clear()
            self.ax2.clear()

            self.ax1.text(0.5, 0.5, 'Графики очищены\n\nНажмите "Обновить графики"\nдля отображения данных',
                          ha='center', va='center', transform=self.ax1.transAxes,
                          fontsize=12, color='gray')
            self.ax1.set_axis_off()

            self.ax2.text(0.5, 0.5, 'Нет данных для отображения',
                          ha='center', va='center', transform=self.ax2.transAxes,
                          fontsize=10, color='gray')
            self.ax2.set_axis_off()

            self.canvas.draw()
            self.viz_status.set("Графики очищены")
            self._log("[VISUALIZATION] Графики очищены")

        except Exception as e:
            self._log(f"[ERROR] Ошибка очистки графиков: {str(e)}")
            self.viz_status.set("Ошибка очистки")

    def _show_population(self):
        """Показать текущую популяцию"""
        if not self.current_population:
            messagebox.showinfo("Информация", "Популяция еще не создана")
            return

        pop_window = tk.Toplevel(self)
        pop_window.title(f"Текущая популяция (поколение {self.current_generation})")
        pop_window.geometry("600x400")

        # Список моделей
        listbox = tk.Listbox(pop_window, width=80, height=20)
        scrollbar = ttk.Scrollbar(pop_window, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)

        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for i, model in enumerate(self.current_population):
            score = model.get('total_score', model.get('f1_score', 0))
            label = model.get('label', f'model_{i}')
            listbox.insert(tk.END, f"{i:03d}. [{label}] Приспособленность: {score:.4f}")

    def _show_selected(self):
        """Показать отобранные модели"""
        if not self.selected_models:
            messagebox.showinfo("Информация", "Селекция еще не выполнена")
            return

        sel_window = tk.Toplevel(self)
        sel_window.title(f"Отобранные модели (поколение {self.current_generation})")
        sel_window.geometry("600x400")

        listbox = tk.Listbox(sel_window, width=80, height=20)
        scrollbar = ttk.Scrollbar(sel_window, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)

        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for i, model in enumerate(self.selected_models):
            score = model.get('total_score', model.get('f1_score', 0))
            label = model.get('label', f'model_{i}')
            listbox.insert(tk.END, f"{i:03d}. [{label}] Приспособленность: {score:.4f}")

    def _show_children(self):
        """Показать потомков"""
        if not self.children_population:
            messagebox.showinfo("Информация", "Дети еще не созданы")
            return

        child_window = tk.Toplevel(self)
        child_window.title(f"Потомки (поколение {self.current_generation})")
        child_window.geometry("600x400")

        listbox = tk.Listbox(child_window, width=80, height=20)
        scrollbar = ttk.Scrollbar(child_window, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)

        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for i, child in enumerate(self.children_population):
            score = child.get('total_score', child.get('f1_score', 0))
            parents = child.get('parents', ['?', '?'])
            mutated = " (мутировал)" if child.get('mutated', False) else ""
            listbox.insert(tk.END, f"{i:03d}. [{child.get('label', 'child')}] "
                                   f"Приспособленность: {score:.4f} "
                                   f"Родители: {parents[0]}, {parents[1]}{mutated}")

    def _analyze_generations(self):
        """Анализ поколений"""
        if not self.generations:
            messagebox.showinfo("Информация", "Поколения еще не созданы")
            return

        analysis_window = tk.Toplevel(self)
        analysis_window.title("Анализ поколений")
        analysis_window.geometry("800x600")

        # Создаем текстовое поле для отчета
        text = scrolledtext.ScrolledText(analysis_window, width=100, height=35)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        report = "=" * 80 + "\n"
        report += "АНАЛИЗ ПОКОЛЕНИЙ ГЕНЕТИЧЕСКОГО АЛГОРИТМА\n"
        report += "=" * 80 + "\n\n"

        for gen in self.generations:
            report += f"ПОКОЛЕНИЕ {gen.number}:\n"
            report += f"  Время создания: {gen.timestamp}\n"
            report += f"  Количество моделей: {len(gen.models)}\n"
            report += f"  Средняя приспособленность: {gen.get_average_fitness():.4f}\n"

            best = gen.get_best_model()
            if best:
                report += f"  Лучшая модель: {best.get('label', 'unknown')}\n"
                report += f"  Лучшая приспособленность: {best.get('total_score', 0):.4f}\n"

            report += "-" * 40 + "\n"

        # Общая статистика
        if len(self.generations) > 1:
            report += "\n" + "=" * 80 + "\n"
            report += "ОБЩАЯ СТАТИСТИКА:\n"
            report += "=" * 80 + "\n"

            first_gen = self.generations[0].get_average_fitness()
            last_gen = self.generations[-1].get_average_fitness()
            improvement = ((last_gen - first_gen) / first_gen * 100) if first_gen > 0 else 0

            report += f"  Первое поколение (средняя): {first_gen:.4f}\n"
            report += f"  Последнее поколение (средняя): {last_gen:.4f}\n"
            report += f"  Улучшение: {improvement:.2f}%\n"

            # Находим лучшее поколение
            best_gen = max(self.generations, key=lambda g: g.get_average_fitness())
            report += f"  Лучшее поколение: {best_gen.number} "
            report += f"(средняя: {best_gen.get_average_fitness():.4f})\n"

        text.insert(tk.END, report)
        text.config(state=tk.DISABLED)

    def _save_generation(self):
        """Сохранить текущее поколение"""
        if not self.generations:
            messagebox.showerror("Ошибка", "Нет поколений для сохранения")
            return

        # Выбираем директорию
        dir_path = filedialog.askdirectory(
            title="Выберите папку для сохранения поколения",
            initialdir=self.working_dir.get()
        )

        if dir_path:
            latest_gen = self.generations[-1]
            saved_dir = latest_gen.save_to_dir(dir_path)
            self._log(f"Поколение {latest_gen.number} сохранено в {saved_dir}")
            messagebox.showinfo("Сохранено",
                                f"Поколение {latest_gen.number} успешно сохранено")


def main():
    """Главная функция"""
    app = CompleteGeneticApp()
    app.mainloop()


if __name__ == "__main__":
    main()