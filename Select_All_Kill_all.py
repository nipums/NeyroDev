#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSelection.py
Программа для селекции нейросетевых моделей с графическим интерфейсом.
С исправленной оценкой специализации метрик.
"""

import os
import pickle
import threading
import queue
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
import re
from sklearn.preprocessing import StandardScaler

# БОЛЬШЕ РАНДОМА
from random import uniform, randint

# Используем те же функции из предыдущего скрипта
SUSPICIOUS_WORDS = {
    "secure": uniform(0, 1.5),
    "login": uniform(0, 1.5),
    "verify": uniform(0, 1.5),
    "bank": uniform(0, 1.5),
    "account": uniform(0, 1.5),
    "update": uniform(0, 1.5),
    "confirm": uniform(0, 1.5),
    "paypal": uniform(0, 1.5)
}

EXEC_WEIGHTS = {
    "exe": uniform(0, 1.5),
    "bat": uniform(0, 1.5),
    "cmd": uniform(0, 1.5),
    "sh": uniform(0, 1.5),
    "scr": uniform(0, 1.5),
    "pif": uniform(0, 1.5),
    "jar": uniform(0, 1.5),
    "msi": uniform(0, 1.5),
    "com": uniform(0, 1.5),
    "vbs": uniform(0, 1.5)
}

URL_FEATURES_COUNT = 20


def shannon_entropy(s_in: str) -> float:
    if not s_in:
        return 0.0
    probs = [float(s_in.count(c)) / len(s_in) for c in set(s_in)]
    import math
    return - sum(p * math.log2(p) for p in probs if p > 0)


def extract_features(url: str) -> List[float]:
    try:
        s = str(url).strip()
        if s == '':
            return [0.0] * URL_FEATURES_COUNT
        s_low = s.lower()
        protocol_removed = re.sub(r'^https?://', '', s_low)
        host = protocol_removed.split('/')[0]
        path = protocol_removed[len(host):] if len(protocol_removed) > len(host) else ''

        len_url = len(s)
        len_host = len(host)
        cnt_dots = s.count('.')
        cnt_hyphen = s.count('-')
        cnt_digits = sum(c.isdigit() for c in s)
        digit_ratio = cnt_digits / len_url if len_url > 0 else 0.0
        subdomains = host.count('.')
        is_ip = 1.0 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host) else 0.0

        susp_count = 0
        susp_weighted = 0.0
        tokens = re.split(r'[^a-z0-9]+', s_low)
        for t in tokens:
            if not t:
                continue
            if t in SUSPICIOUS_WORDS:
                susp_count += 1
                try:
                    susp_weighted += float(SUSPICIOUS_WORDS[t])
                except:
                    susp_weighted += 1.0

        ext = ''
        if '.' in host:
            ext = host.split('.')[-1]
        ext_flag = 1.0 if ext in EXEC_WEIGHTS else 0.0
        ext_weight = float(EXEC_WEIGHTS.get(ext, 0.0)) if EXEC_WEIGHTS else 0.0

        has_at = 1.0 if '@' in s else 0.0
        cnt_double_slash = s.count('//') - (2 if s_low.startswith('http://') or s_low.startswith('https://') else 0)
        cnt_double_slash = max(0, cnt_double_slash)

        entropy_host = shannon_entropy(host)

        special_chars = sum(1 for c in s if not c.isalnum())
        ratio_special = special_chars / len_url if len_url > 0 else 0.0

        cnt_slash = s.count('/') - (1 if s_low.startswith('http://') or s_low.startswith('https://') else 0)
        cnt_slash = max(0, cnt_slash)
        starts_http = 1.0 if s_low.startswith('http://') or s_low.startswith('https://') else 0.0
        num_letters = sum(1 for c in s if c.isalpha())
        vowels = set('aeiou')
        num_vowels = sum(1 for c in s.lower() if c in vowels)
        ratio_vowels = (num_vowels / num_letters) if num_letters > 0 else 0.0

        feats = [
            float(len_url), float(len_host), float(cnt_dots), float(cnt_hyphen),
            float(cnt_digits), float(digit_ratio), float(subdomains), float(is_ip),
            float(susp_count), float(susp_weighted), float(ext_flag), float(ext_weight),
            float(has_at), float(cnt_double_slash), float(entropy_host), float(ratio_special),
            float(cnt_slash), float(starts_http), float(num_letters), float(ratio_vowels)
        ]
        if len(feats) < URL_FEATURES_COUNT:
            feats += [0.0] * (URL_FEATURES_COUNT - len(feats))
        elif len(feats) > URL_FEATURES_COUNT:
            feats = feats[:URL_FEATURES_COUNT]
        return feats
    except Exception:
        return [0.0] * URL_FEATURES_COUNT


def load_dataset(csv_path: str, n_samples: int = 20000, sep=',') -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, sep=sep)
    except Exception:
        df = pd.read_csv(csv_path, header=None, sep=sep)

    cols = [c.lower() for c in df.columns]
    df.columns = cols

    domain_col = None
    label_col = None
    for c in cols:
        if any(k in c for k in ['domain_name', 'url', 'domain']):
            domain_col = c
        if any(k in c for k in ['is_phishing', 'label', 'phish']):
            label_col = c

    if domain_col is None or label_col is None:
        if len(df.columns) >= 2:
            df.columns = ['url', 'label'] + list(df.columns[2:])
            domain_col, label_col = 'url', 'label'
        else:
            raise ValueError("Не удалось определить колонки домена и метки в CSV")

    df = df[[domain_col, label_col]].copy()
    df = df.dropna(subset=[domain_col, label_col])

    try:
        df[label_col] = df[label_col].astype(int)
    except:
        df[label_col] = df[label_col].apply(
            lambda x: 1 if str(x).strip().lower() in ('1', 'true', 'phish', 'yes') else 0)

    df.rename(columns={domain_col: 'url', label_col: 'label'}, inplace=True)

    if n_samples is None or n_samples <= 0 or n_samples >= len(df):
        return df.reset_index(drop=True)

    try:
        def sample_group(group, n_samples, total_len):
            sample_size = max(1, int(n_samples * len(group) / total_len))
            return group.sample(sample_size, random_state=42) if len(group) > 0 else group

        df_sampled = df.groupby('label', group_keys=False).apply(
            lambda x: sample_group(x, n_samples, len(df))
        )
        if len(df_sampled) > n_samples:
            df_sampled = df_sampled.sample(n_samples, random_state=42)
        return df_sampled.reset_index(drop=True)
    except Exception:
        return df.sample(min(n_samples, len(df)), random_state=42).reset_index(drop=True)


class ModelSelector:
    def __init__(self):
        self.metric_names = [
            "length_url", "length_host", "dots_count", "hyphen_count",
            "digits_count", "digit_ratio", "subdomains", "is_ip",
            "suspicious_count", "suspicious_weight", "exec_flag", "exec_weight",
            "has_at", "double_slash", "entropy", "special_chars_ratio",
            "slash_count", "starts_http", "letter_count", "vowel_ratio"
        ]

        self.metric_descriptions = {
            "length_url": "Длина URL",
            "length_host": "Длина домена",
            "dots_count": "Количество точек",
            "hyphen_count": "Количество дефисов",
            "digits_count": "Количество цифр",
            "digit_ratio": "Доля цифр",
            "subdomains": "Количество поддоменов",
            "is_ip": "IP-адрес вместо домена",
            "suspicious_count": "Количество подозрительных слов",
            "suspicious_weight": "Вес подозрительных слов",
            "exec_flag": "Наличие исполняемого расширения",
            "exec_weight": "Вес исполняемого расширения",
            "has_at": "Наличие символа @",
            "double_slash": "Количество двойных слешей",
            "entropy": "Энтропия домена",
            "special_chars_ratio": "Доля специальных символов",
            "slash_count": "Количество слешей",
            "starts_http": "Начинается с HTTP/HTTPS",
            "letter_count": "Количество букв",
            "vowel_ratio": "Доля гласных букв"
        }

    def evaluate_metric_performance(self, model, X_metric: np.ndarray, y_true: np.ndarray,
                                    metric_index: int, threshold: float = 0.5) -> float:
        """Улучшенная оценка производительности с диагностикой"""
        try:
            # Проверка входных данных
            if len(X_metric) == 0 or len(y_true) == 0:
                return 0.0

            # Проверяем баланс классов
            unique_classes, class_counts = np.unique(y_true, return_counts=True)
            if len(unique_classes) < 2:
                return 0.0  # Только один класс - невозможно оценить

            # Получаем предсказания
            y_pred = model.predict(X_metric)

            # Проверяем, что предсказания не все одинаковые
            if len(np.unique(y_pred)) == 1:
                # Все предсказания одинаковые - проверяем accuracy
                baseline_accuracy = max(class_counts) / len(y_true)
                if baseline_accuracy > 0.9:  # Если один класс доминирует
                    return 0.0

            # Вычисляем метрики
            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)

            # Комбинированная оценка
            combined_score = 0.5 * f1 + 0.3 * precision + 0.2 * recall

            # Проверяем, лучше ли модель случайного угадывания
            baseline_accuracy = max(class_counts) / len(y_true)
            random_guess = 0.5  # Для сбалансированных классов

            # Используем более мягкий порог
            required_improvement = 0.05  # 5% улучшение вместо 10%

            if combined_score <= max(baseline_accuracy, random_guess) + required_improvement:
                return 0.0

            return min(combined_score, 1.0)  # Ограничиваем сверху 1.0

        except Exception as e:
            print(f"Ошибка при оценке метрики {metric_index}: {e}")
            return 0.0

    def create_specialized_metric_dataset(self, full_dataset: pd.DataFrame, metric_index: int,
                                          n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Улучшенная версия создания специализированного датасета с проверкой наличия метрики"""
        try:
            # Увеличиваем выборку для лучшего покрытия
            sample_size = min(n_samples * 5, len(full_dataset))
            sample_df = full_dataset.sample(sample_size, random_state=42 + metric_index)

            # Извлекаем фичи
            X_list = []
            for url in sample_df['url'].tolist():
                feats = extract_features(url)
                X_list.append(feats)

            X_full = np.array(X_list, dtype=float)
            y_full = sample_df['label'].values.astype(int)

            # Анализируем распределение метрики
            metric_values = X_full[:, metric_index]

            # Проверяем, есть ли вариация в данных
            if np.std(metric_values) < 1e-6:
                self._log_metric_stats(metric_index, metric_values, "Нет вариации")
                # Возвращаем больше данных для анализа
                return X_full[:min(500, len(X_full))], y_full[:min(500, len(y_full))]

            # Улучшенная стратегия отбора: используем несколько квантилей
            thresholds = [
                np.percentile(metric_values, 80),  # Верхние 20%
                np.percentile(metric_values, 60),  # Верхние 40%
                np.percentile(metric_values, 40),  # Нижние 40%
                np.percentile(metric_values, 20)  # Нижние 20%
            ]

            selected_indices = []
            for threshold in thresholds:
                high_indices = np.where(metric_values >= threshold)[0]
                low_indices = np.where(metric_values <= np.percentile(metric_values, 20))[0]

                # Балансируем выборку для каждого порога
                for indices in [high_indices, low_indices]:
                    if len(indices) > 0:
                        balanced = self._get_balanced_indices(indices, y_full, n_samples // 8)
                        selected_indices.extend(balanced)

            # Убираем дубликаты
            selected_indices = list(set(selected_indices))

            # Если не набрали достаточно примеров, добавляем случайные
            if len(selected_indices) < n_samples // 2:
                all_indices = np.arange(len(X_full))
                remaining_indices = np.setdiff1d(all_indices, selected_indices)

                if len(remaining_indices) > 0:
                    additional_needed = n_samples - len(selected_indices)
                    additional_indices = np.random.choice(
                        remaining_indices,
                        min(additional_needed, len(remaining_indices)),
                        replace=False
                    )
                    selected_indices.extend(additional_indices)

            # Гарантируем минимальный размер датасета
            if len(selected_indices) < 200:
                # Добавляем больше случайных примеров
                all_indices = np.arange(len(X_full))
                additional_indices = np.random.choice(
                    all_indices,
                    min(500, len(all_indices)),
                    replace=False
                )
                selected_indices = list(set(selected_indices + list(additional_indices)))

            # Проверяем баланс классов в финальной выборке
            final_labels = y_full[selected_indices]
            unique_classes, counts = np.unique(final_labels, return_counts=True)

            if len(unique_classes) < 2:
                # Добавляем примеры недостающего класса
                missing_class = 1 if 0 in unique_classes else 0
                missing_indices = np.where(y_full == missing_class)[0]
                if len(missing_indices) > 0:
                    additional_missing = np.random.choice(
                        missing_indices,
                        min(100, len(missing_indices)),
                        replace=False
                    )
                    selected_indices = list(set(selected_indices + list(additional_missing)))

            self._log_metric_stats(metric_index, metric_values[selected_indices],
                                   f"Финальный датасет: {len(selected_indices)} примеров")

            return X_full[selected_indices], y_full[selected_indices]

        except Exception as e:
            print(f"Ошибка создания датасета для метрики {metric_index}: {e}")
            # Возвращаем большой случайный датасет в случае ошибки
            sample_size = min(1000, len(full_dataset))
            sample_df = full_dataset.sample(sample_size, random_state=42)
            X_list = [extract_features(url) for url in sample_df['url'].tolist()]
            return np.array(X_list, dtype=float), sample_df['label'].values.astype(int)

    def _get_balanced_indices(self, indices: np.ndarray, y_full: np.ndarray, target_size: int) -> List[int]:
        """Балансирует индексы по классам"""
        if len(indices) == 0:
            return []

        pos_indices = indices[y_full[indices] == 1]
        neg_indices = indices[y_full[indices] == 0]

        n_each = min(target_size // 2, len(pos_indices), len(neg_indices))
        if n_each == 0:
            # Если один из классов отсутствует, берем то что есть
            return list(indices[:min(target_size, len(indices))])

        selected_pos = np.random.choice(pos_indices, n_each, replace=False)
        selected_neg = np.random.choice(neg_indices, n_each, replace=False)

        return list(selected_pos) + list(selected_neg)

    def _log_metric_stats(self, metric_index: int, values: np.ndarray, message: str):
        """Логирует статистику метрики"""
        stats = f"Метрика {metric_index}: {message}, min={np.min(values):.2f}, max={np.max(values):.2f}, mean={np.mean(values):.2f}, std={np.std(values):.2f}"
        print(stats)

    def get_metric_development_level(self, score: float) -> str:
        """Определяет уровень развития метрики"""
        if score >= 0.8:
            return "Отлично"
        elif score >= 0.7:
            return "Очень хорошо"
        elif score >= 0.6:
            return "Хорошо"
        elif score >= 0.5:
            return "Удовлетворительно"
        elif score >= 0.4:
            return "Слабо"
        elif score >= 0.3:
            return "Очень слабо"
        else:
            return "Не развита"


class MetricVisualizationWindow(tk.Toplevel):
    def __init__(self, parent, model_info: Dict, metric_scores: Dict):
        super().__init__(parent)
        self.title("Анализ развития метрик индивида")
        self.geometry("1000x700")
        self.resizable(True, True)

        self.model_info = model_info
        self.metric_scores = metric_scores
        self.selector = ModelSelector()

        self._build_ui()
        self._update_display()

    def _build_ui(self):
        # Main frames
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header with model info
        header_frame = ttk.LabelFrame(main_frame, text="Информация о модели", padding=10)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_text = tk.StringVar()
        ttk.Label(header_frame, textvariable=self.info_text, justify=tk.LEFT).pack(anchor=tk.W)

        # Visualization area
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True)

        # Left - metrics table
        table_frame = ttk.LabelFrame(viz_frame, text="Детализация метрик", padding=10)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Create treeview for metrics
        columns = ("metric", "description", "score", "level", "status")
        self.metrics_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)

        # Define headings
        self.metrics_tree.heading("metric", text="Метрика")
        self.metrics_tree.heading("description", text="Описание")
        self.metrics_tree.heading("score", text="Оценка")
        self.metrics_tree.heading("level", text="Уровень")
        self.metrics_tree.heading("status", text="Статус")

        # Define columns
        self.metrics_tree.column("metric", width=120)
        self.metrics_tree.column("description", width=200)
        self.metrics_tree.column("score", width=80)
        self.metrics_tree.column("level", width=120)
        self.metrics_tree.column("status", width=100)

        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=scrollbar.set)

        self.metrics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Right - charts
        chart_frame = ttk.LabelFrame(viz_frame, text="Визуализация", padding=10)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 10))
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Summary frame
        summary_frame = ttk.LabelFrame(main_frame, text="Сводка", padding=10)
        summary_frame.pack(fill=tk.X, pady=(10, 0))

        self.summary_text = tk.StringVar()
        ttk.Label(summary_frame, textvariable=self.summary_text, justify=tk.LEFT).pack(anchor=tk.W)

    def _update_display(self):
        # Update header info
        model_data = self.model_info['model_data']
        info_text = (f"Модель: {model_data['filename']}\n"
                     f"Общий счет: {self.model_info['total_score']:.4f} | "
                     f"Специализация: {self.model_info['specialization']:.4f}")
        self.info_text.set(info_text)

        # Update metrics table
        self._update_metrics_table()

        # Update charts
        self._update_charts()

        # Update summary
        self._update_summary()

    def _update_metrics_table(self):
        # Clear existing items
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)

        # Add metrics data
        for metric_name, score in self.metric_scores.items():
            description = self.selector.metric_descriptions.get(metric_name, "Нет описания")
            level = self.selector.get_metric_development_level(score)

            # Determine status
            if score >= 0.7:
                status = "✅ Сильная"
            elif score >= 0.5:
                status = "⚠️ Средняя"
            else:
                status = "❌ Слабая"

            self.metrics_tree.insert("", tk.END, values=(
                metric_name, description, f"{score:.4f}", level, status
            ))

    def _update_charts(self):
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Prepare data for plotting
        metrics = list(self.metric_scores.keys())
        scores = list(self.metric_scores.values())

        # Chart 1: Bar chart of all metrics
        colors = ['green' if s >= 0.7 else 'orange' if s >= 0.5 else 'red' for s in scores]
        bars = self.ax1.bar(range(len(metrics)), scores, color=colors, alpha=0.7)
        self.ax1.set_title('Развитие метрик индивида', fontsize=12, fontweight='bold')
        self.ax1.set_ylabel('Оценка метрики')
        self.ax1.set_ylim(0, 1.0)
        self.ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            self.ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                          f'{score:.3f}', ha='center', va='bottom', fontsize=8)

        # Chart 2: Pie chart of metric categories
        strong_count = sum(1 for s in scores if s >= 0.7)
        medium_count = sum(1 for s in scores if s >= 0.5 and s < 0.7)
        weak_count = sum(1 for s in scores if s < 0.5)

        categories = ['Сильные', 'Средние', 'Слабые']
        counts = [strong_count, medium_count, weak_count]
        colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']

        if sum(counts) > 0:
            self.ax2.pie(counts, labels=categories, colors=colors_pie, autopct='%1.1f%%',
                         startangle=90, shadow=True)
            self.ax2.set_title('Распределение метрик по категориям', fontsize=12, fontweight='bold')
        else:
            self.ax2.text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                          transform=self.ax2.transAxes, fontsize=14)

        self.canvas.draw()

    def _update_summary(self):
        scores = list(self.metric_scores.values())

        strong_metrics = [m for m, s in self.metric_scores.items() if s >= 0.7]
        weak_metrics = [m for m, s in self.metric_scores.items() if s < 0.3]

        summary = (f"Всего метрик: {len(scores)} | "
                   f"Сильных (≥0.7): {len(strong_metrics)} | "
                   f"Слабых (<0.3): {len(weak_metrics)}\n"
                   f"Средняя оценка: {np.mean(scores):.4f} | "
                   f"Максимальная: {max(scores):.4f} | "
                   f"Минимальная: {min(scores):.4f}\n")

        if strong_metrics:
            summary += f"Лучшие метрики: {', '.join(strong_metrics[:3])}\n"

        if len(strong_metrics) >= 5:
            summary += "✅ Высокий уровень специализации"
        elif len(strong_metrics) >= 3:
            summary += "⚠️ Средний уровень специализации"
        else:
            summary += "❌ Низкий уровень специализации"

        self.summary_text.set(summary)


class SelectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neuro Model Selector")
        self.geometry("1200x800")
        self.resizable(True, True)

        # State
        self.loaded_models = []
        self.dataset = None
        self.metric_datasets = {}
        self.selection_results = {}
        self.current_group = 0
        self.total_groups = 0

        # GUI vars
        self.models_path = tk.StringVar()
        self.dataset_path = tk.StringVar()
        self.group_size = tk.IntVar(value=10)
        self.samples_per_metric = tk.IntVar(value=1000)
        self.control_group_threshold = tk.DoubleVar(value=0.5)
        self.deviation_percent = tk.DoubleVar(value=10.0)
        self.specialization_threshold = tk.DoubleVar(value=0.7)

        # log queue
        self.log_queue = queue.Queue()

        self.selector = ModelSelector()

        self._build_ui()
        self._after_process_logs()

    def _build_ui(self):
        # Main notebook
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Setup tab
        setup_frame = ttk.Frame(notebook, padding=10)
        notebook.add(setup_frame, text="Настройка")

        # Models selection
        ttk.Label(setup_frame, text="Папка с моделями (.pkl):", font=('Arial', 10, 'bold')).grid(row=0, column=0,
                                                                                                 sticky=tk.W,
                                                                                                 pady=(0, 5))
        model_frame = ttk.Frame(setup_frame)
        model_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W + tk.E, pady=(0, 10))
        ttk.Entry(model_frame, textvariable=self.models_path, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="Обзор", command=self._choose_models_dir).pack(side=tk.RIGHT, padx=(5, 0))

        # Dataset selection
        ttk.Label(setup_frame, text="Датасет для оценки (CSV):", font=('Arial', 10, 'bold')).grid(row=2, column=0,
                                                                                                  sticky=tk.W,
                                                                                                  pady=(10, 5))
        dataset_frame = ttk.Frame(setup_frame)
        dataset_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W + tk.E, pady=(0, 10))
        ttk.Entry(dataset_frame, textvariable=self.dataset_path, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dataset_frame, text="Обзор", command=self._choose_dataset).pack(side=tk.RIGHT, padx=(5, 0))

        # Parameters
        params_frame = ttk.LabelFrame(setup_frame, text="Параметры селекции", padding=10)
        params_frame.grid(row=4, column=0, columnspan=3, sticky=tk.W + tk.E, pady=10)

        ttk.Label(params_frame, text="Размер группы:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.group_size, width=10).grid(row=0, column=1, sticky=tk.W, padx=(5, 20))

        ttk.Label(params_frame, text="Примеров на метрику:").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.samples_per_metric, width=10).grid(row=0, column=3, sticky=tk.W,
                                                                                     padx=(5, 20))

        ttk.Label(params_frame, text="Порог контрольной группы:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.control_group_threshold, width=10).grid(row=1, column=1, sticky=tk.W,
                                                                                          padx=(5, 20))

        ttk.Label(params_frame, text="Отклонение (%):").grid(row=1, column=2, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.deviation_percent, width=10).grid(row=1, column=3, sticky=tk.W,
                                                                                    padx=(5, 20))

        ttk.Label(params_frame, text="Порог специализации:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.specialization_threshold, width=10).grid(row=2, column=3, sticky=tk.W,
                                                                                           padx=(5, 20))

        # Buttons
        btn_frame = ttk.Frame(setup_frame)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=20)

        ttk.Button(btn_frame, text="📂 Загрузить модели", command=self._load_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📊 Загрузить датасет", command=self._load_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🔧 Подготовить метрики", command=self._prepare_metrics).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🎯 Начать селекцию", command=self._start_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="👁️ Просмотр метрик", command=self._show_metric_analysis).pack(side=tk.LEFT, padx=5)

        # Results tab
        results_frame = ttk.Frame(notebook, padding=10)
        notebook.add(results_frame, text="Результаты")

        # Results display with treeview
        results_tree_frame = ttk.Frame(results_frame)
        results_tree_frame.pack(fill=tk.BOTH, expand=True)

        # Create treeview for results
        columns = ("group", "model", "total_score", "specialization", "strong_metrics", "status")
        self.results_tree = ttk.Treeview(results_tree_frame, columns=columns, show="headings", height=15)

        # Define headings
        self.results_tree.heading("group", text="Группа")
        self.results_tree.heading("model", text="Модель")
        self.results_tree.heading("total_score", text="Общий счет")
        self.results_tree.heading("specialization", text="Специализация")
        self.results_tree.heading("strong_metrics", text="Сильные метрики")
        self.results_tree.heading("status", text="Статус")

        # Define columns
        self.results_tree.column("group", width=80)
        self.results_tree.column("model", width=200)
        self.results_tree.column("total_score", width=100)
        self.results_tree.column("specialization", width=100)
        self.results_tree.column("strong_metrics", width=150)
        self.results_tree.column("status", width=100)

        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(results_tree_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind double-click event
        self.results_tree.bind("<Double-1>", self._on_model_double_click)

        # Progress
        self.progress = ttk.Progressbar(setup_frame, mode='determinate')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=tk.W + tk.E, pady=10)

        # Status
        self.status_var = tk.StringVar(value="Готов")
        ttk.Label(setup_frame, textvariable=self.status_var).grid(row=7, column=0, columnspan=3, sticky=tk.W)

        # Logs
        log_frame = ttk.LabelFrame(setup_frame, text="Логи", padding=5)
        log_frame.grid(row=8, column=0, columnspan=3, sticky=tk.W + tk.E, pady=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _choose_models_dir(self):
        path = filedialog.askdirectory(title="Выберите папку с моделями")
        if path:
            self.models_path.set(path)

    def _choose_dataset(self):
        path = filedialog.askopenfilename(title="Выберите CSV датасет",
                                          filetypes=[("CSV files", "*.csv")])
        if path:
            self.dataset_path.set(path)

    def _log(self, text: str):
        self.log_queue.put(text + "\n")

    def _after_process_logs(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, item)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(200, self._after_process_logs)

    def _load_models(self):
        path = self.models_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Ошибка", "Выберите корректную папку с моделями")
            return

        self._log("[LOAD] Загрузка моделей...")

        def load_thread():
            try:
                models = []
                pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
                self._log(f"[LOAD] Найдено {len(pkl_files)} .pkl файлов")

                for i, filename in enumerate(pkl_files):
                    try:
                        with open(os.path.join(path, filename), 'rb') as f:
                            model_data = pickle.load(f)

                        # Проверяем структуру загруженных данных
                        if isinstance(model_data, dict):
                            if 'pipeline' in model_data:
                                models.append({
                                    'pipeline': model_data['pipeline'],
                                    'filename': filename,
                                    'f1': model_data.get('f1', 0.0),
                                    'label': model_data.get('label', 'unknown')
                                })
                                self._log(f"[LOAD] Загружена модель {filename} (словарь с pipeline)")
                            else:
                                self._log(f"[LOAD] Пропущен {filename}: словарь без pipeline")
                        else:
                            # Предполагаем, что это прямая модель
                            models.append({
                                'pipeline': model_data,
                                'filename': filename,
                                'f1': 0.0,
                                'label': 'unknown'
                            })
                            self._log(f"[LOAD] Загружена модель {filename} (прямая модель)")

                    except Exception as e:
                        self._log(f"[ERROR] Ошибка загрузки {filename}: {e}")

                self.loaded_models = models
                self._log(f"[LOAD] Успешно загружено {len(models)} моделей")

                def update_ui():
                    self.status_var.set(f"Загружено {len(models)} моделей")
                    messagebox.showinfo("Успех", f"Загружено {len(models)} моделей")

                self.after(0, update_ui)

            except Exception as e:
                error_msg = str(e)
                self._log(f"[ERROR] Ошибка загрузки: {error_msg}")

                def error_ui(msg=error_msg):
                    messagebox.showerror("Ошибка", f"Ошибка загрузки: {msg}")

                self.after(0, error_ui)

        threading.Thread(target=load_thread, daemon=True).start()

    def _load_dataset(self):
        path = self.dataset_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Ошибка", "Выберите корректный CSV файл")
            return

        self._log("[DATASET] Загрузка датасета...")

        def load_thread():
            try:
                n_samples = self.samples_per_metric.get() * len(self.selector.metric_names) * 5
                df = load_dataset(path, n_samples=n_samples)
                self.dataset = df

                self._log(f"[DATASET] Загружено {len(df)} примеров")

                def update_ui():
                    self.status_var.set(f"Датасет: {len(df)} примеров")
                    messagebox.showinfo("Успех", f"Загружено {len(df)} примеров")

                self.after(0, update_ui)

            except Exception as e:
                error_msg = str(e)
                self._log(f"[ERROR] Ошибка загрузки датасета: {error_msg}")

                def error_ui(msg=error_msg):
                    messagebox.showerror("Ошибка", f"Ошибка загрузки датасета: {msg}")

                self.after(0, error_ui)

        threading.Thread(target=load_thread, daemon=True).start()

    def _prepare_metrics(self):
        if self.dataset is None:
            messagebox.showerror("Ошибка", "Сначала загрузите датасет")
            return

        self._log("[METRICS] Подготовка метрик...")

        def prepare_thread():
            try:
                self.metric_datasets = {}
                samples_per_metric = self.samples_per_metric.get()

                for i, metric_name in enumerate(self.selector.metric_names):
                    X_metric, y_metric = self.selector.create_specialized_metric_dataset(
                        self.dataset, i, samples_per_metric
                    )
                    self.metric_datasets[metric_name] = (X_metric, y_metric)
                    self._log(f"[METRICS] Подготовлена метрика {metric_name}: {len(X_metric)} примеров")

                def update_ui():
                    self.status_var.set(f"Подготовлено {len(self.metric_datasets)} метрик")
                    messagebox.showinfo("Успех", f"Подготовлено {len(self.metric_datasets)} метрик")

                self.after(0, update_ui)

            except Exception as e:
                error_msg = str(e)
                self._log(f"[ERROR] Ошибка подготовки метрик: {error_msg}")

                def error_ui(msg=error_msg):
                    messagebox.showerror("Ошибка", f"Ошибка подготовки метрик: {msg}")

                self.after(0, error_ui)

        threading.Thread(target=prepare_thread, daemon=True).start()

    def _start_selection(self):
        if not self.loaded_models:
            messagebox.showerror("Ошибка", "Сначала загрузите модели")
            return

        if not self.metric_datasets:
            messagebox.showerror("Ошибка", "Сначала подготовьте метрики")
            return

        self._log("[SELECTION] Начало селекции...")
        self._log(f"[SELECTION] Загружено моделей: {len(self.loaded_models)}")
        self._log(f"[SELECTION] Подготовлено метрик: {len(self.metric_datasets)}")

        def selection_thread():
            try:
                group_size = self.group_size.get()
                total_models = len(self.loaded_models)
                self.total_groups = (total_models + group_size - 1) // group_size

                self.selection_results = {}
                all_selected_models = []

                for group_idx in range(self.total_groups):
                    self.current_group = group_idx
                    start_idx = group_idx * group_size
                    end_idx = min((group_idx + 1) * group_size, total_models)
                    group_models = self.loaded_models[start_idx:end_idx]

                    self._log(
                        f"[SELECTION] Обработка группы {group_idx + 1}/{self.total_groups} ({len(group_models)} моделей)")

                    # Процесс селекции для группы
                    selected_models = self._process_group_selection(group_models, group_idx)
                    all_selected_models.extend(selected_models)

                    # Сохранение результатов группы
                    if selected_models:
                        self._save_group_results(selected_models, group_idx)

                    # Обновление прогресса
                    progress = (group_idx + 1) / self.total_groups * 100

                    def update_progress():
                        self.progress['value'] = progress
                        self.status_var.set(f"Обработана группа {group_idx + 1}/{self.total_groups}")

                    self.after(0, update_progress)

                def final_update():
                    self.status_var.set(f"Селекция завершена. Отобрано {len(all_selected_models)} моделей")
                    if all_selected_models:
                        self._update_results_tree(all_selected_models)
                        messagebox.showinfo("Завершено",
                                            f"Селекция завершена. Отобрано {len(all_selected_models)} моделей")
                    else:
                        messagebox.showwarning("Предупреждение", "Селекция завершена, но не отобрано ни одной модели")

                self.after(0, final_update)

            except Exception as e:
                error_msg = str(e)
                self._log(f"[ERROR] Ошибка селекции: {error_msg}")
                import traceback
                self._log(f"[ERROR] Детали: {traceback.format_exc()}")

                def error_ui(msg=error_msg):
                    messagebox.showerror("Ошибка", f"Ошибка селекции: {msg}")

                self.after(0, error_ui)

        threading.Thread(target=selection_thread, daemon=True).start()

    def _process_group_selection(self, group_models: List[Dict], group_idx: int) -> List[Dict]:
        """Процесс селекции для одной группы с улучшенными критериями"""
        selected_models = []
        control_threshold = self.control_group_threshold.get()
        deviation = self.deviation_percent.get() / 100.0
        specialization_threshold = self.specialization_threshold.get()

        # Оценка моделей по всем метрикам
        model_scores = []
        for model_idx, model_data in enumerate(group_models):
            try:
                scores = self._evaluate_model_metrics(model_data)

                # Улучшенная проверка: считаем ненулевые метрики
                non_zero_scores = {k: v for k, v in scores.items() if v > 0.0}
                if not non_zero_scores:
                    self._log(f"[GROUP {group_idx}] Модель {model_data['filename']} имеет все нулевые оценки")
                    continue

                total_score = sum(scores.values())
                specialization = self._calculate_specialization(scores)
                strong_metrics_count = sum(1 for s in scores.values() if s >= 0.7)
                passed_metrics_count = sum(1 for s in scores.values() if s >= control_threshold)

                model_scores.append({
                    'model_data': model_data,
                    'scores': scores,
                    'total_score': total_score,
                    'specialization': specialization,
                    'strong_metrics_count': strong_metrics_count,
                    'passed_metrics_count': passed_metrics_count,
                    'non_zero_metrics_count': len(non_zero_scores),
                    'group_index': group_idx,
                    'model_index': model_idx
                })

                self._log(
                    f"[GROUP {group_idx}] Оценена модель {model_data['filename']}: "
                    f"общий счет {total_score:.4f}, ненулевых метрик: {len(non_zero_scores)}"
                )

            except Exception as e:
                self._log(f"[ERROR] Ошибка оценки модели {model_data.get('filename', 'unknown')}: {e}")
                continue

        if not model_scores:
            self._log(f"[GROUP {group_idx}] Все модели получили нулевые оценки!")
            return []

        # Улучшенные критерии отбора с учетом отклонения
        for model_info in model_scores:
            # Критерий 1: Минимальное количество ненулевых метрик
            if model_info['non_zero_metrics_count'] < len(self.selector.metric_names) // 2:
                continue

            # Критерий 2: Проход через отсев по достаточному количеству метрик
            passed_metrics = model_info['passed_metrics_count']
            required_passed = max(3, len(self.selector.metric_names) // 3)

            if passed_metrics >= required_passed:
                selected_models.append(model_info)
                continue

            # Критерий 3: Высокая специализация с учетом отклонения
            # Отклонение влияет на порог специализации - делаем его более мягким
            effective_specialization_threshold = specialization_threshold * (1 - deviation)
            if (model_info['specialization'] >= effective_specialization_threshold and
                    model_info['strong_metrics_count'] >= 1):
                selected_models.append(model_info)
                continue

            # Критерий 4: Быть лучшим по хотя бы одной метрике с учетом отклонения
            if self._is_best_in_any_metric(model_info, model_scores, deviation):
                best_metric_score = max(model_info['scores'].values())
                if best_metric_score >= 0.6:
                    selected_models.append(model_info)
                    continue

            # Критерий 5: Иметь несколько очень сильных метрик
            excellent_metrics = sum(1 for s in model_info['scores'].values() if s >= 0.8)
            if excellent_metrics >= 2:
                selected_models.append(model_info)
                continue

        # Дополнительная фильтрация
        max_selected_per_group = max(1, len(group_models) // 3)

        if len(selected_models) > max_selected_per_group:
            # Сортируем по комбинированному критерию
            selected_models.sort(key=lambda x: (
                    x['strong_metrics_count'] * 2 +
                    x['total_score'] +
                    x['specialization'] +
                    x['non_zero_metrics_count'] * 0.5
            ), reverse=True)
            selected_models = selected_models[:max_selected_per_group]

        self._log(f"[GROUP {group_idx}] Из {len(group_models)} отобрано {len(selected_models)} моделей")

        if selected_models:
            avg_total = sum(m['total_score'] for m in selected_models) / len(selected_models)
            avg_strong = sum(m['strong_metrics_count'] for m in selected_models) / len(selected_models)
            avg_non_zero = sum(m['non_zero_metrics_count'] for m in selected_models) / len(selected_models)
            self._log(
                f"[GROUP {group_idx}] Средний общий счет: {avg_total:.3f}, "
                f"среднее сильных метрик: {avg_strong:.1f}, "
                f"среднее ненулевых метрик: {avg_non_zero:.1f}"
            )

        return selected_models

    def _evaluate_model_metrics(self, model_data) -> Dict[str, float]:
        """Оценивает модель по всем метрикам с улучшенной обработкой"""
        scores = {}

        # ИСПРАВЛЕНИЕ: извлекаем pipeline из model_data
        if isinstance(model_data, dict) and 'pipeline' in model_data:
            pipeline = model_data['pipeline']
        else:
            # Если передали напрямую pipeline
            pipeline = model_data

        for metric_name, (X_metric, y_metric) in self.metric_datasets.items():
            try:
                # Проверяем размер датасета метрики
                if len(X_metric) < 50:
                    self._log(f"[EVAL] Внимание: маленький датасет для {metric_name}: {len(X_metric)} примеров")
                    scores[metric_name] = 0.0
                    continue

                metric_index = self.selector.metric_names.index(metric_name)
                score = self.selector.evaluate_metric_performance(
                    pipeline,
                    X_metric, y_metric,
                    metric_index,
                    self.control_group_threshold.get()
                )

                # Дополнительная проверка: если датасет слишком однороден, score может быть занижен
                if score == 0.0:
                    # Проверяем вариативность данных
                    metric_values = X_metric[:, metric_index]
                    if np.std(metric_values) > 0.1:  # Есть вариация, но модель не справилась
                        scores[metric_name] = 0.0
                    else:
                        # Данные однородны - это проблема датасета, а не модели
                        scores[metric_name] = 0.0
                else:
                    scores[metric_name] = score

            except Exception as e:
                self._log(f"[EVAL] Ошибка оценки {metric_name}: {e}")
                scores[metric_name] = 0.0

        return scores

    def _calculate_specialization(self, scores: Dict[str, float]) -> float:
        """Вычисляет степень специализации модели с улучшенной формулой"""
        if not scores:
            return 0.0

        values = list(scores.values())

        # Игнорируем нулевые оценки при расчете специализации
        non_zero_values = [v for v in values if v > 0.0]
        if not non_zero_values:
            return 0.0

        max_score = max(non_zero_values)
        other_scores = [s for s in non_zero_values if s < max_score]

        if not other_scores:
            return 1.0

        avg_other = sum(other_scores) / len(other_scores)
        if avg_other == 0:
            return 1.0

        # Улучшенная формула: учитываем разброс оценок
        specialization = max_score / (max_score + avg_other)

        # Дополнительный boost если есть несколько сильных метрик
        strong_metrics = sum(1 for s in non_zero_values if s >= 0.7)
        if strong_metrics >= 2:
            specialization = min(1.0, specialization * 1.2)

        return specialization

    def _is_best_in_any_metric(self, model_info: Dict, all_models: List[Dict], deviation: float) -> bool:
        """Строгая проверка, является ли модель лучшей по хотя бы одной метрике с учетом отклонения"""
        for metric_name in self.selector.metric_names:
            model_score = model_info['scores'][metric_name]

            # Пропускаем слабые метрики
            if model_score < 0.5:
                continue

            is_best = True
            best_score = model_score

            for other_model in all_models:
                if other_model is model_info:
                    continue

                other_score = other_model['scores'][metric_name]
                # Учитываем отклонение: модель считается лучшей если ее оценка в пределах отклонения от максимальной
                if other_score > best_score * (1 + deviation):
                    is_best = False
                    best_score = other_score
                    break

            # Дополнительное условие: быть значительно лучше среднего
            if is_best:
                other_scores = [m['scores'][metric_name] for m in all_models
                                if m is not model_info and m['scores'][metric_name] > 0]
                if other_scores:
                    avg_other = sum(other_scores) / len(other_scores)
                    if model_score >= avg_other * (1 + deviation):
                        return True

        return False

    def _save_group_results(self, selected_models: List[Dict], group_idx: int):
        """Сохраняет результаты селекции группы с JSON файлом"""
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = f"selected/selection_results_group_{group_idx + 1}_{timestamp}"
            os.makedirs(results_dir, exist_ok=True)

            # Сортируем модели по рейтингу для определения места в группе
            selected_models.sort(key=lambda x: (
                    x['strong_metrics_count'] * 2 +
                    x['total_score'] +
                    x['specialization']
            ), reverse=True)

            # Сохраняем отобранные модели и создаем JSON информацию
            group_json_data = {
                "group_index": group_idx,
                "total_models_in_group": len(selected_models),
                "selection_timestamp": timestamp,
                "selection_criteria": {
                    "control_group_threshold": self.control_group_threshold.get(),
                    "deviation_percent": self.deviation_percent.get(),
                    "specialization_threshold": self.specialization_threshold.get(),
                    "group_size": self.group_size.get()
                },
                "selected_models": []
            }

            for i, model_info in enumerate(selected_models):
                model_data = model_info['model_data']
                position_in_group = i + 1  # Место в группе (1-based)

                filename = f"selected_model_{position_in_group}_{model_data['filename']}"
                filepath = os.path.join(results_dir, filename)

                # Сохраняем модель
                with open(filepath, 'wb') as f:
                    # Сохраняем в старом формате для обратной совместимости
                    pickle.dump({
                        'pipeline': model_data['pipeline'],
                        'filename': model_data['filename'],  # Старое поле вместо original_filename
                        'f1': model_data.get('f1', 0.0),  # Сохраняем оригинальные поля
                        'label': model_data.get('label', 'unknown'),
                        'scores': model_info['scores'],
                        'total_score': model_info['total_score'],
                        'specialization': model_info['specialization'],
                        'strong_metrics_count': model_info['strong_metrics_count'],
                        'group_index': group_idx,
                        'position_in_group': position_in_group,
                        'selection_timestamp': datetime.datetime.now()
                    }, f)

                # Добавляем информацию в JSON
                model_json_info = {
                    "position_in_group": position_in_group,
                    "original_filename": model_data['filename'],
                    "saved_filename": filename,
                    "total_score": float(model_info['total_score']),
                    "specialization": float(model_info['specialization']),
                    "strong_metrics_count": model_info['strong_metrics_count'],
                    "non_zero_metrics_count": model_info['non_zero_metrics_count'],
                    "metric_scores": {k: float(v) for k, v in model_info['scores'].items()},
                    "top_metrics": dict(sorted(model_info['scores'].items(),
                                               key=lambda x: x[1], reverse=True)[:5])
                }
                group_json_data["selected_models"].append(model_json_info)

            # Сохраняем JSON файл
            json_filepath = os.path.join(results_dir, f"group_{group_idx + 1}_selection_info.json")
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(group_json_data, f, ensure_ascii=False, indent=2)

            # Сохраняем текстовую информацию о группе
            info_file = os.path.join(results_dir, "group_info.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"Информация о группе селекции #{group_idx + 1}\n")
                f.write("==========================================\n")
                f.write(f"Время: {datetime.datetime.now()}\n")
                f.write(f"Всего моделей в группе: {len(selected_models)}\n")
                f.write(f"Отобрано моделей: {len(selected_models)}\n")
                f.write(f"Критерии отбора:\n")
                f.write(f"  - Порог контрольной группы: {self.control_group_threshold.get()}\n")
                f.write(f"  - Отклонение: {self.deviation_percent.get()}%\n")
                f.write(f"  - Порог специализации: {self.specialization_threshold.get()}\n\n")

                f.write("Отобранные модели (в порядке убывания рейтинга):\n")
                for i, model_info in enumerate(selected_models):
                    model_data = model_info['model_data']
                    f.write(f"{i + 1}. {model_data['filename']}\n")
                    f.write(f"   Общий счет: {model_info['total_score']:.4f}\n")
                    f.write(f"   Специализация: {model_info['specialization']:.4f}\n")
                    f.write(f"   Сильных метрик: {model_info['strong_metrics_count']}\n")
                    f.write(f"   Ненулевых метрик: {model_info['non_zero_metrics_count']}\n")

                    # Топ-3 метрики
                    top_metrics = sorted(model_info['scores'].items(),
                                         key=lambda x: x[1], reverse=True)[:3]
                    f.write(f"   Лучшие метрики: ")
                    f.write(", ".join([f"{m[0]}: {m[1]:.4f}" for m in top_metrics]) + "\n\n")

            self._log(f"[SAVE] Группа {group_idx + 1} сохранена в {results_dir}")
            self._log(f"[SAVE] JSON информация сохранена в {json_filepath}")

            self.selection_results[group_idx] = {
                'directory': results_dir,
                'selected_count': len(selected_models),
                'models': selected_models,
                'json_file': json_filepath
            }

        except Exception as e:
            self._log(f"[ERROR] Ошибка сохранения группы {group_idx}: {e}")

    def _update_results_tree(self, selected_models: List[Dict]):
        """Обновляет дерево результатов"""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Add selected models to treeview
        for model_info in selected_models:
            model_data = model_info['model_data']

            # Count strong metrics (score >= 0.7)
            strong_metrics = [m for m, s in model_info['scores'].items() if s >= 0.7]
            strong_count = len(strong_metrics)

            # Count non-zero metrics
            non_zero_count = model_info['non_zero_metrics_count']

            # Determine status
            if strong_count >= 5:
                status = "Отлично"
            elif strong_count >= 3:
                status = "Хорошо"
            elif non_zero_count >= len(self.selector.metric_names) // 2:
                status = "Удовлетворительно"
            else:
                status = "Слабо"

            group_index = model_info.get('group_index', 'N/A')
            group_display = f"Группа {group_index + 1}" if group_index != 'N/A' else "Группа N/A"

            self.results_tree.insert("", tk.END, values=(
                group_display,
                model_data['filename'],
                f"{model_info['total_score']:.4f}",
                f"{model_info['specialization']:.4f}",
                f"{strong_count} сильных",
                status
            ))

    def _on_model_double_click(self, event):
        """Обработчик двойного клика по модели в дереве результатов"""
        selection = self.results_tree.selection()
        if selection:
            item = selection[0]
            values = self.results_tree.item(item, "values")
            model_info = self._find_model_by_values(values)
            if model_info:
                self._show_model_metrics(model_info)

    def _find_model_by_values(self, values):
        """Находит модель по значениям в дереве"""
        filename = values[1]
        for group_idx, group_info in self.selection_results.items():
            for model_info in group_info['models']:
                if model_info['model_data']['filename'] == filename:
                    return model_info
        return None

    def _show_metric_analysis(self):
        """Показывает анализ метрик для выбранной модели"""
        if not hasattr(self, 'selection_results') or not self.selection_results:
            messagebox.showwarning("Предупреждение", "Сначала выполните селекцию моделей")
            return

        # Создаем диалог выбора модели
        selection_dialog = tk.Toplevel(self)
        selection_dialog.title("Выбор модели для анализа")
        selection_dialog.geometry("600x400")
        selection_dialog.transient(self)
        selection_dialog.grab_set()

        # Создаем список моделей
        model_listbox = tk.Listbox(selection_dialog, width=80, height=20)
        model_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Заполняем список моделей
        all_models = []
        for group_idx, group_info in self.selection_results.items():
            for model_info in group_info['models']:
                model_data = model_info['model_data']
                display_text = (f"Группа {group_idx + 1}: {model_data['filename']} "
                                f"(Общий счет: {model_info['total_score']:.4f}, "
                                f"Специализация: {model_info['specialization']:.4f})")
                model_listbox.insert(tk.END, display_text)
                all_models.append(model_info)

        def on_select():
            selection = model_listbox.curselection()
            if selection:
                model_info = all_models[selection[0]]
                selection_dialog.destroy()
                self._show_model_metrics(model_info)

        # Кнопка выбора
        btn_frame = ttk.Frame(selection_dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Анализировать", command=on_select).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="Отмена", command=selection_dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _show_model_metrics(self, model_info: Dict):
        """Показывает детальный анализ метрик модели"""
        MetricVisualizationWindow(self, model_info, model_info['scores'])


def main():
    app = SelectionApp()
    app.mainloop()


if __name__ == '__main__':
    main()