#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crossingover.py
Финальный рабочий билд для кроссинговера нейросетевых пайплайнов и анализа созданных детей.
Требует модуль Select_All_Kill_all.py с классом ModelSelector.
"""
import os
import pickle
import random
import datetime
import copy
import re
import math
import json

from typing import List, Dict, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from Select_All_Kill_all import ModelSelector

# --- Константы и веса (можно изменить) ---
SUSPICIOUS_WORDS = {
    "secure": 1.0, "login": 0.9, "verify": 0.8, "bank": 1.2,
    "account": 1.0, "update": 0.7, "confirm": 0.6, "paypal": 1.5
}

EXEC_WEIGHTS = {
    "exe": 1.0, "bat": 0.8, "cmd": 0.8, "sh": 0.7, "scr": 0.9,
    "pif": 0.9, "jar": 1.0, "msi": 1.2, "com": 0.8, "vbs": 0.9
}

URL_FEATURES_COUNT = 20

# --------------------------
# === Вспомогательные функции
# --------------------------

def extract_pipeline(obj):
    """
    Пытается извлечь sklearn Pipeline из объекта.
    Возможные варианты:
    - obj — уже Pipeline
    - obj — dict, где есть ключи pipeline / model / clf / estimator
    - obj — dict, где pipeline вложен глубже
    """
    if obj is None:
        return None

    if isinstance(obj, Pipeline):
        return obj

    if isinstance(obj, dict):
        # популярные ключи
        for key in ["pipeline", "model", "clf", "classifier", "estimator", "pipeline_obj"]:
            if key in obj and obj[key] is not None:
                if isinstance(obj[key], Pipeline):
                    return obj[key]

        # более глубокий поиск
        for v in obj.values():
            if isinstance(v, Pipeline):
                return v
            elif isinstance(v, dict):
                # рекурсивный поиск во вложенных словарях
                pipeline = extract_pipeline(v)
                if pipeline is not None:
                    return pipeline

    return None

def shannon_entropy(s_in: str) -> float:
    if not s_in:
        return 0.0
    probs = [float(s_in.count(c)) / len(s_in) for c in set(s_in)]
    return - sum(p * math.log2(p) for p in probs if p > 0)


def extract_features(url: str) -> List[float]:
    """ТОЧНАЯ КОПИЯ функции из селекции"""
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

        # ВАЖНО: исправленная логика для cnt_slash
        cnt_slash = s.count('/')
        # Учитываем начальные слеши в протоколе
        if s_low.startswith('http://'):
            cnt_slash -= 2  # http://
        elif s_low.startswith('https://'):
            cnt_slash -= 2  # https://
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


def safe_get_meta_map_from_json_folder(folder: str) -> Dict[str, dict]:
    """
    Ищет первый .json файл в папке и парсит его как структуру с информацией по моделям.
    Возвращает map: original_filename -> meta_dict
    """
    if not os.path.exists(folder):
        return {}

    json_files = [f for f in os.listdir(folder) if f.lower().endswith('.json')]
    print(f"[DEBUG] Найдено JSON файлов в {folder}: {json_files}")

    for fname in json_files:
        path = os.path.join(folder, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            mapping = {}
            print(f"[DEBUG] Загружаем JSON: {fname}, тип данных: {type(data)}")

            if isinstance(data, dict) and 'selected_models' in data and isinstance(data['selected_models'], list):
                print(f"[DEBUG] Найдена структура с selected_models, количество: {len(data['selected_models'])}")
                for i, entry in enumerate(data['selected_models']):
                    orig = entry.get('original_filename') or entry.get('filename') or entry.get('saved_filename')
                    if orig:
                        mapping[orig] = entry
                        print(f"[DEBUG] Добавлено сопоставление: {orig} -> индекс {i}")
                    else:
                        print(f"[DEBUG] Пропущена запись {i}: нет имени файла")
            elif isinstance(data, list):
                print(f"[DEBUG] JSON является списком, количество элементов: {len(data)}")
                for i, entry in enumerate(data):
                    if isinstance(entry, dict):
                        orig = entry.get('original_filename') or entry.get('filename') or entry.get('saved_filename')
                        if orig:
                            mapping[orig] = entry
                            print(f"[DEBUG] Добавлено сопоставление: {orig} -> индекс {i}")
                        else:
                            print(f"[DEBUG] Пропущена запись {i}: нет имени файла")

            print(f"[DEBUG] Итоговое сопоставление: {len(mapping)} записей")
            return mapping

        except Exception as e:
            print(f"[WARN] Не удалось прочитать JSON {path}: {e}")
            continue

    print(f"[DEBUG] В папке {folder} не найдено подходящих JSON файлов")
    return {}


def pick_weighted(models: List[dict], rng: random.Random) -> Optional[dict]:
    """
    Roulette-wheel selection based on metadata fitness.
    metadata expected keys: total_score, specialization, strong_metrics_count, top_metrics (dict)
    Fallback: if meta absent, fallback to 1.0 uniform weight.
    """
    weights = []
    for m in models:
        meta = m.get('meta') or m.get('meta_info') or m.get('pipeline_meta') or {}
        total = 0.0
        try:
            total = float(meta.get('total_score', 0.0))
        except Exception:
            total = 0.0
        spec = 0.0
        try:
            spec = float(meta.get('specialization', 0.0))
        except Exception:
            spec = 0.0
        strong = 0
        try:
            strong = int(meta.get('strong_metrics_count', 0))
        except Exception:
            strong = 0
        top = meta.get('top_metrics', {}) or {}
        top_n = len(top) if isinstance(top, dict) else 0

        base = max(0.0, total)
        if base <= 0:
            base = 1.0

        weight = base * (1.0 + spec) * (1.0 + (strong / 20.0)) * (1.0 + (top_n / 5.0))
        weights.append(max(weight, 0.0001))

    s = sum(weights)
    if s <= 0:
        return rng.choice(models) if models else None

    pick = rng.random() * s
    cum = 0.0
    for w, m in zip(weights, models):
        cum += w
        if pick <= cum:
            return m
    return models[-1] if models else None


def sanitize_params_for_pipeline(pipeline: Pipeline, params: dict) -> dict:
    """
    Убирает из params ключи которых не понимает целевой pipeline (get_params().keys()).
    """
    valid_keys = set(pipeline.get_params().keys())
    return {k: v for k, v in params.items() if k in valid_keys}

# ниже добавил изменил
def mutate_params(params: dict, rng: random.Random, mutation_rate: float = 0.03) -> dict:
    """
    Улучшенная мутация: различные типы мутаций для разных типов параметров.
    Поддерживает числовые параметры, категориальные, булевы и сложные структуры.
    """
    new = copy.deepcopy(params)

    for k, v in list(new.items()):
        if rng.random() > mutation_rate:
            continue

        # Мутация для float параметров
        if isinstance(v, float):
            # Несколько стратегий мутации для float
            strategy = rng.choice(['gaussian', 'log_scale', 'uniform'])

            if strategy == 'gaussian':
                # Гауссов шум с относительным стандартным отклонением
                std_dev = abs(v) * 0.1 + 0.01  # Минимум 0.01 чтобы избежать 0
                new[k] = float(v + rng.gauss(0, std_dev))

            elif strategy == 'log_scale':
                # Мультипликативная мутация в логарифмической шкале
                factor = 10 ** rng.uniform(-0.2, 0.2)
                new[k] = float(v * factor)

            elif strategy == 'uniform':
                # Равномерная мутация в пределах ±20%
                new[k] = float(v * rng.uniform(0.8, 1.2))

            # Гарантируем разумные пределы для некоторых известных параметров
            if 'alpha' in k.lower() or 'C' in k.lower():
                new[k] = max(1e-6, min(new[k], 1000.0))
            elif 'rate' in k.lower() or 'ratio' in k.lower():
                new[k] = max(0.0, min(new[k], 1.0))

        # Мутация для int параметров
        elif isinstance(v, int) and not isinstance(v, bool) and v > 0:
            # Для положительных целых (типа n_estimators, max_depth и т.д.)
            if v <= 10:
                # Малые значения - небольшие изменения
                delta = rng.randint(-2, 2)
                new[k] = max(1, v + delta)
            else:
                # Большие значения - процентное изменение
                change_percent = rng.uniform(-0.3, 0.3)
                delta = int(v * change_percent)
                new[k] = max(1, v + delta)

        # Мутация для tuple/list параметров (например, hidden_layer_sizes)
        elif isinstance(v, (tuple, list)) and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            new_list = []
            for x in v:
                if rng.random() < 0.3:  # Вероятность мутации отдельного элемента
                    if x <= 10:
                        delta = rng.randint(-2, 2)
                        new_x = max(1, x + delta)
                    else:
                        delta = int(x * rng.uniform(-0.2, 0.2))
                        new_x = max(1, x + delta)
                    new_list.append(new_x)
                else:
                    new_list.append(x)

            # Иногда добавляем/удаляем слои в нейросетях
            if 'hidden_layer' in k.lower() and rng.random() < 0.1:
                if len(new_list) > 1 and rng.random() < 0.5:
                    # Удаляем случайный слой
                    idx = rng.randint(0, len(new_list) - 1)
                    new_list.pop(idx)
                else:
                    # Добавляем новый слой
                    new_size = rng.choice([32, 64, 128, 256])
                    insert_pos = rng.randint(0, len(new_list))
                    new_list.insert(insert_pos, new_size)

            new[k] = tuple(new_list) if isinstance(v, tuple) else new_list

        # Мутация для строковых (категориальных) параметров
        elif isinstance(v, str):
            # Для некоторых известных категориальных параметров
            if 'activation' in k.lower():
                alternatives = ['relu', 'tanh', 'logistic', 'identity']
                if v in alternatives and rng.random() < 0.3:
                    new[k] = rng.choice([x for x in alternatives if x != v])

            elif 'solver' in k.lower():
                alternatives = ['lbfgs', 'sgd', 'adam']
                if v in alternatives and rng.random() < 0.3:
                    new[k] = rng.choice([x for x in alternatives if x != v])

            elif 'criterion' in k.lower():
                alternatives = ['gini', 'entropy', 'log_loss']
                if v in alternatives and rng.random() < 0.3:
                    new[k] = rng.choice([x for x in alternatives if x != v])

        # Мутация для булевых параметров
        elif isinstance(v, bool):
            if rng.random() < 0.2:  # 20% шанс инвертировать булево значение
                new[k] = not v

    return new

# ниже добавил
def advanced_mutation(child_pipeline: Pipeline, rng: random.Random, mutation_rate: float = 0.03) -> Pipeline:
    """
    Продвинутая мутация, которая включает не только параметры, но и структурные изменения.
    """
    mutated_pipeline = copy.deepcopy(child_pipeline)

    # 1. Мутация гиперпараметров (как раньше)
    params = mutated_pipeline.get_params()
    mutated_params = mutate_params(params, rng, mutation_rate)
    safe_params = sanitize_params_for_pipeline(mutated_pipeline, mutated_params)

    try:
        mutated_pipeline.set_params(**safe_params)
    except Exception as e:
        print(f"[MUTATION] Warning: could not set some parameters: {e}")

    # 2. Структурная мутация: иногда меняем порядок или состав шагов pipeline
    if rng.random() < mutation_rate * 2:  # Более низкая вероятность структурных изменений
        mutated_pipeline = structural_mutation(mutated_pipeline, rng)

    # 3. Мутация весов нейронных сетей (если есть)
    randomize_mlp_weights_if_present(mutated_pipeline, rng)

    # 4. Дополнительная мутация для конкретных типов моделей
    mutated_pipeline = model_specific_mutation(mutated_pipeline, rng, mutation_rate)

    return mutated_pipeline

# ниже добавил
def structural_mutation(pipeline: Pipeline, rng: random.Random) -> Pipeline:
    """
    Структурная мутация: изменяет состав или порядок шагов в pipeline.
    """
    steps = pipeline.steps.copy()

    if len(steps) <= 1:
        return pipeline

    mutation_type = rng.choice(['swap', 'remove', 'duplicate'])

    try:
        if mutation_type == 'swap' and len(steps) >= 2:
            # Меняем местами два случайных шага
            i, j = rng.sample(range(len(steps)), 2)
            steps[i], steps[j] = steps[j], steps[i]

        elif mutation_type == 'remove' and len(steps) >= 2:
            # Удаляем случайный шаг (кроме последнего - классификатора)
            if len(steps) > 2:  # Оставляем как минимум 2 шага
                remove_idx = rng.randint(0, len(steps) - 2)  # Не удаляем классификатор
                steps.pop(remove_idx)

        elif mutation_type == 'duplicate':
            # Дублируем случайный шаг
            dup_idx = rng.randint(0, len(steps) - 1)
            step_name, step_obj = steps[dup_idx]
            new_name = f"{step_name}_copy"
            steps.insert(dup_idx + 1, (new_name, copy.deepcopy(step_obj)))

    except Exception as e:
        print(f"[STRUCTURAL MUTATION] Error: {e}")

    pipeline.steps = steps
    return pipeline

# ниже добавил
def model_specific_mutation(pipeline: Pipeline, rng: random.Random, mutation_rate: float) -> Pipeline:
    """
    Специфические мутации для разных типов моделей.
    """
    for name, step in pipeline.steps:
        try:
            step_class = type(step).__name__

            # Для Random Forest / Decision Trees
            if 'Forest' in step_class or 'Tree' in step_class:
                if hasattr(step, 'max_depth') and rng.random() < mutation_rate:
                    # Мутация max_depth
                    if step.max_depth is not None:
                        new_depth = step.max_depth + rng.randint(-3, 3)
                        step.max_depth = max(1, new_depth)

                if hasattr(step, 'n_estimators') and rng.random() < mutation_rate:
                    # Мутация n_estimators
                    new_estimators = step.n_estimators + rng.randint(-20, 20)
                    step.n_estimators = max(10, new_estimators)

            # Для SVM
            elif 'SVC' in step_class or 'SVM' in step_class:
                if hasattr(step, 'C') and rng.random() < mutation_rate:
                    # Мутация параметра C
                    step.C = step.C * 10 ** rng.uniform(-0.3, 0.3)

            # Для методов масштабирования/нормализации
            elif 'Scaler' in step_class:
                if hasattr(step, 'with_mean') and rng.random() < mutation_rate * 2:
                    step.with_mean = not step.with_mean
                if hasattr(step, 'with_std') and rng.random() < mutation_rate * 2:
                    step.with_std = not step.with_std

        except Exception as e:
            # Игнорируем ошибки в специфических мутациях
            continue

    return pipeline

# ниже добавил
def adaptive_mutation_rate(child: dict, base_rate: float = 0.03) -> float:
    """
    Адаптивная скорость мутации на основе качества ребенка.
    Хуже дети -> больше мутаций, лучше дети -> меньше мутаций.
    """
    scores = child.get('scores', {})
    if not scores:
        return base_rate

    # Средняя оценка по всем метрикам
    avg_score = sum(scores.values()) / len(scores)

    # Адаптируем скорость мутации: хуже оценки -> выше мутация
    if avg_score < 0.5:
        return min(base_rate * 2.0, 0.1)  # Увеличиваем мутацию для слабых детей
    elif avg_score > 0.8:
        return base_rate * 0.5  # Уменьшаем мутацию для сильных детей
    else:
        return base_rate


def randomize_mlp_weights_if_present(pipeline: Pipeline, rng: random.Random):
    """
    Если в pipeline есть шаг с классом, у которого есть атрибуты coefs_ / intercepts_ (sklearn MLPClassifier),
    - создаём случайные coefs_ и intercepts_ совместимые с текущей архитектурой.
    """
    try:
        for name, step in pipeline.steps:
            if hasattr(step, 'coefs_') or hasattr(step, 'n_outputs_') or hasattr(step, 'hidden_layer_sizes'):
                if hasattr(step, 'hidden_layer_sizes'):
                    layer_sizes = step.hidden_layer_sizes
                    if isinstance(layer_sizes, int):
                        layer_sizes = (layer_sizes,)
                    if hasattr(step, 'coefs_') and step.coefs_:
                        n_features = step.coefs_[0].shape[0]
                        n_outputs = step.coefs_[-1].shape[1]
                    else:
                        # не знаем входных признаков — пропускаем
                        continue

                    all_sizes = [n_features] + list(layer_sizes) + [n_outputs]
                    coefs = []
                    intercepts = []
                    for i in range(len(all_sizes) - 1):
                        n_in = all_sizes[i]
                        n_out = all_sizes[i + 1]
                        limit = math.sqrt(6.0 / (n_in + n_out))
                        w = np.random.uniform(-limit, limit, (n_in, n_out))
                        b = np.random.uniform(-limit * 0.1, limit * 0.1, n_out)
                        coefs.append(np.array(w, dtype=float))
                        intercepts.append(np.array(b, dtype=float))
                    step.coefs_ = coefs
                    step.intercepts_ = intercepts
    except Exception:
        pass

# --------------------------
# === Основное GUI приложение
# --------------------------
class CrossBreedingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neuro Crossbreeding (A × B)")
        self.geometry("980x640")

        self.dir_a = tk.StringVar()
        self.dir_b = tk.StringVar()
        self.num_children = tk.IntVar(value=20)
        self.random_seed = tk.IntVar(value=0)
        self.mutation_rate = tk.DoubleVar(value=0.03)

        self.pop_a: List[dict] = []
        self.pop_b: List[dict] = []
        # ПЕРЕИМЕНОВАНИЕ: чтобы не конфликтовать с tkinter internal .children
        self.generated_children: List[dict] = []

        self.selector = ModelSelector()

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)

        # Row: Pop A
        ttk.Label(frm, text="Популяция A (директория):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(frm, textvariable=self.dir_a, width=70).grid(row=1, column=0, sticky=tk.W+tk.E, padx=(0,6))
        ttk.Button(frm, text="Обзор A", command=lambda: self._choose_dir(self.dir_a)).grid(row=1, column=1)

        # Row: Pop B
        ttk.Label(frm, text="Популяция B (директория):").grid(row=2, column=0, sticky=tk.W, pady=(8,0))
        ttk.Entry(frm, textvariable=self.dir_b, width=70).grid(row=3, column=0, sticky=tk.W+tk.E, padx=(0,6))
        ttk.Button(frm, text="Обзор B", command=lambda: self._choose_dir(self.dir_b)).grid(row=3, column=1)

        # Params
        ttk.Label(frm, text="Число детей:").grid(row=4, column=0, sticky=tk.W, pady=(10,0))
        ttk.Entry(frm, textvariable=self.num_children, width=8).grid(row=4, column=0, sticky=tk.W, padx=(100,0))
        ttk.Label(frm, text="Случайное зерно (0=random):").grid(row=4, column=0, sticky=tk.W, padx=(200,0))
        ttk.Entry(frm, textvariable=self.random_seed, width=8).grid(row=4, column=0, sticky=tk.W, padx=(380,0))
        ttk.Label(frm, text="Mutation rate:").grid(row=4, column=0, sticky=tk.W, padx=(500,0))
        ttk.Entry(frm, textvariable=self.mutation_rate, width=6).grid(row=4, column=0, sticky=tk.W, padx=(590,0))

        # Buttons
        ttk.Button(frm, text="Загрузить популяции", command=self._load_both_pops).grid(row=5, column=0, pady=(12,6), sticky=tk.W)
        ttk.Button(frm, text="Выполнить кроссинговер", command=self._run_crossbreeding).grid(row=5, column=0, pady=(12,6))
        ttk.Button(frm, text="Анализ детей", command=self._open_children_browser).grid(row=5, column=0, pady=(12,6), sticky=tk.E)

        # Log area
        self.log = tk.Text(frm, height=24)
        self.log.grid(row=6, column=0, columnspan=2, sticky=tk.W+tk.E, pady=(10,0))

    def _choose_dir(self, var: tk.StringVar):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    def _log(self, text: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.insert(tk.END, f"[{ts}] {text}\n")
        self.log.see(tk.END)

    def _load_population_from_dir(self, path: str) -> List[dict]:
        """
        Загружает все .pkl в папке и сопоставляет метаданные из JSON (если есть).
        Возвращает список dict с keys: pipeline, filename, filepath, meta
        """
        result = []
        if not path or not os.path.exists(path):
            return result
        meta_map = safe_get_meta_map_from_json_folder(path)

        pkl_files = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith('.pkl'):
                    pkl_files.append(os.path.join(root, f))

        self._log(f"[LOAD] Найдено {len(pkl_files)} .pkl файлов в {path}")

        for fp in pkl_files:
            f = os.path.basename(fp)
            try:
                with open(fp, 'rb') as fh:
                    data = pickle.load(fh)

                pipeline = extract_pipeline(data)
                if pipeline is None:
                    self._log(f"[WARN] {f}: pipeline не найден или невалиден — пропуск.")
                    continue

                meta = None
                if isinstance(data, dict):
                    # если в PKL был словарь, то пробуем взять meta из него
                    meta = data.get("meta") or data.get("meta_info") or data.get("pipeline_meta")

                    # Ищем метаданные в JSON
                    if f in meta_map and isinstance(meta_map[f], dict):
                        if meta is None:
                            meta = {}
                        # Объединяем метаданные из JSON с метаданными из PKL
                        meta.update(meta_map[f])
                    else:
                        # Пробуем найти по оригинальному имени
                        orig = data.get('original_filename') or data.get('filename')
                        if orig and orig in meta_map:
                            if meta is None:
                                meta = {}
                            meta.update(meta_map[orig])
                else:
                    # Если data - это просто pipeline, ищем метаданные в JSON
                    if f in meta_map:
                        meta = meta_map[f]
                    else:
                        # Ищем по любому совпадению
                        for k, v in meta_map.items():
                            if isinstance(v, dict) and (
                                    v.get('saved_filename') == f or v.get('original_filename') == f):
                                meta = v
                                break

                result.append({
                    'pipeline': pipeline,
                    'filename': f,
                    'filepath': fp,
                    'meta': meta or {}
                })
                self._log(f"[LOAD] Успешно загружена модель {f}")

            except Exception as e:
                self._log(f"[ERROR] Ошибка загрузки {f}: {str(e)}")
                continue

        self._log(f"[LOAD] Успешно загружено {len(result)} моделей из {len(pkl_files)} файлов")
        return result

    def _load_both_pops(self):
        a = self.dir_a.get().strip()
        b = self.dir_b.get().strip()
        if not a or not b:
            messagebox.showerror("Ошибка", "Выберите обе папки (A и B).")
            return

        self._log("Загрузка популяции A...")
        self.pop_a = self._load_population_from_dir(a)
        self._log("Загрузка популяции B...")
        self.pop_b = self._load_population_from_dir(b)

        self._log(f"Популяция A: {len(self.pop_a)} моделей; Популяция B: {len(self.pop_b)} моделей")

        # Диагностика
        if not self.pop_a:
            self._log(f"[DIAG] Папка A: {a}")
            self._log(f"[DIAG] Содержимое папки A: {os.listdir(a) if os.path.exists(a) else 'Папка не существует'}")
        if not self.pop_b:
            self._log(f"[DIAG] Папка B: {b}")
            self._log(f"[DIAG] Содержимое папки B: {os.listdir(b) if os.path.exists(b) else 'Папка не существует'}")

        if not self.pop_a or not self.pop_b:
            messagebox.showwarning("Предупреждение",
                                   f"Одна из популяций пуста или не содержит корректных pkl.\n"
                                   f"A: {len(self.pop_a)} моделей, B: {len(self.pop_b)} моделей")
        else:
            messagebox.showinfo("Успех", f"Загружено: A={len(self.pop_a)}, B={len(self.pop_b)}")

    def _run_crossbreeding(self):
        if not self.pop_a or not self.pop_b:
            messagebox.showerror("Ошибка", "Загрузите обе популяции.")
            return
        seed = int(self.random_seed.get())
        rng = random.Random(None if seed == 0 else seed)
        children_num = max(1, int(self.num_children.get()))
        mutation_rate = float(self.mutation_rate.get())

        outdir = os.path.join("crossed", f"cross_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(outdir, exist_ok=True)
        self.generated_children = []

        self._log(f"Создаём {children_num} детей (A×B) в {outdir} ...")

        for i in range(children_num):
            parent_a = pick_weighted(self.pop_a, rng)
            parent_b = pick_weighted(self.pop_b, rng)
            if parent_a is None or parent_b is None:
                self._log("[WARN] Не удалось выбрать пару родителей — пропуск.")
                continue

            attempts = 0
            while parent_a['filename'] == parent_b['filename'] and attempts < 8:
                parent_b = pick_weighted(self.pop_b, rng)
                attempts += 1

            child = self._create_child(parent_a, parent_b, rng, mutation_rate)
            if child:
                fname = os.path.join(outdir, f"child_{i+1:03d}.pkl")
                child['child_filename'] = fname
                with open(fname, 'wb') as fh:
                    pickle.dump(child, fh)
                self.generated_children.append(child)
                total_score = sum(child.get('scores', {}).values()) if child.get('scores') else 0.0
                self._log(f"[OK] child_{i+1:03d}.pkl (sum scores: {total_score:.4f})")
        self._log("Кроссинговер завершён.")
        messagebox.showinfo("Готово", f"Создано {len(self.generated_children)} детей. Папка: {outdir}")

    def _create_child(self, parent_a: dict, parent_b: dict, rng: random.Random, mutation_rate: float) -> dict:
        """
        Создаёт child dict, содержащий pipeline, parents (meta), scores, param_inherit_map, metric inheritance map.
        Использует равновероятный по каждому параметру кроссинговер.
        """
        try:
            pipe_a: Pipeline = parent_a['pipeline']
            pipe_b: Pipeline = parent_b['pipeline']
        except Exception:
            return {}

        if not isinstance(pipe_a, Pipeline) or not isinstance(pipe_b, Pipeline):
            self._log("[ERROR] Один из родителей не является Pipeline — пропуск child.")
            return {}
        child_steps = [(name, copy.deepcopy(step)) for name, step in pipe_a.steps]
        child_pipeline = Pipeline(child_steps)

        params_a = pipe_a.get_params()
        params_b = pipe_b.get_params()

        # РАВНОВЕРОЯТНЫЙ выбор для каждого параметра
        keys = sorted(set(list(params_a.keys()) + list(params_b.keys())))
        child_params = {}
        param_inherit_map = {}
        for k in keys:
            pick = rng.random()
            if pick < 0.5:
                child_params[k] = params_a.get(k, params_b.get(k))
                param_inherit_map[k] = 'A'
            else:
                child_params[k] = params_b.get(k, params_a.get(k))
                param_inherit_map[k] = 'B'

        # Мутация параметров
        child_params = mutate_params(child_params, rng, mutation_rate)

        # Санация перед set_params
        safe_params = sanitize_params_for_pipeline(child_pipeline, child_params)
        try:
            child_pipeline.set_params(**safe_params)
        except Exception:
            pass

        # # Randomize weights for MLP-like models (если есть) - чтобы дети реально отличались
        # randomize_mlp_weights_if_present(child_pipeline, rng)
        # ниже добавил, выше строку закомментировал
        # Применяем продвинутую мутацию ко всему пайплайну
        child_pipeline = advanced_mutation(child_pipeline, rng, mutation_rate)





        # Оценка метрик ребёнка (эвристически, опираясь на meta родителей если есть)
        scores, metric_inherit_map = self._evaluate_child(child_pipeline, parent_a, parent_b, param_inherit_map, rng)

        child = {
            'pipeline': child_pipeline,
            'parents': {
                'A': parent_a.get('filename'),
                'B': parent_b.get('filename')
            },
            'parent_meta': {
                'A': parent_a.get('meta', {}) or {},
                'B': parent_b.get('meta', {}) or {}
            },
            'param_inherit_map': param_inherit_map,
            'scores': scores,
            'inherit_map': metric_inherit_map,
            'created_at': datetime.datetime.now().isoformat(),

            # ниже добавил
            'mutation_rate_used': mutation_rate
        }
        return child

    def _evaluate_child(self, child_pipeline: Pipeline, parent_a: dict, parent_b: dict,
                        param_inherit_map: dict, rng: random.Random) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Оценка метрик ребёнка с использованием реальных параметров родителей
        """
        scores = {}
        inherit_map = {}
        metric_names = self.selector.metric_names

        meta_a = parent_a.get('meta', {}) or {}
        meta_b = parent_b.get('meta', {}) or {}

        # Получаем реальные параметры родителей (если доступны)
        parent_a_params = parent_a.get('pipeline', {}).get_params() if hasattr(parent_a.get('pipeline', {}),
                                                                               'get_params') else {}
        parent_b_params = parent_b.get('pipeline', {}).get_params() if hasattr(parent_b.get('pipeline', {}),
                                                                               'get_params') else {}

        for m in metric_names:
            # Определяем наследование на основе реальных параметров
            source = self._determine_metric_inheritance_by_params(m, param_inherit_map, rng)
            inherit_map[m] = source

            # Получаем оценки родителей
            sa = meta_a.get('metric_scores', {}).get(m, None)
            sb = meta_b.get('metric_scores', {}).get(m, None)

            # Выбираем оценку в зависимости от источника
            if source == 'A' and sa is not None:
                base_score = float(sa)
            elif source == 'B' and sb is not None:
                base_score = float(sb)
            else:
                # Fallback: среднее или случайное значение
                if sa is not None and sb is not None:
                    base_score = 0.5 * (float(sa) + float(sb))
                elif sa is not None:
                    base_score = float(sa)
                elif sb is not None:
                    base_score = float(sb)
                else:
                    base_score = rng.uniform(0.3, 0.9)

            # Добавляем небольшой шум
            perturb = rng.normalvariate(0, 0.02)
            val = max(0.0, min(1.0, base_score + perturb))
            scores[m] = float(val)

        return scores, inherit_map

    def _determine_metric_inheritance_by_params(self, metric_name: str, param_inherit_map: dict,
                                                rng: random.Random) -> str:
        """
        Честное определение: если большинство гиперпараметров ребёнок унаследовал от A — метрика от A.
        Если от B — от B.
        Если поровну — Mixed.
        """
        a_count = sum(1 for v in param_inherit_map.values() if v == 'A')
        b_count = sum(1 for v in param_inherit_map.values() if v == 'B')

        if a_count == b_count:
            return 'Mixed'
        return 'A' if a_count > b_count else 'B'

    def _fallback_inheritance_heuristic(self, metric_name: str, param_inherit_map: dict,
                                        rng: random.Random) -> str:
        """Улучшенная эвристика для случаев без четкого наследования"""
        total_params = len(param_inherit_map)
        if total_params == 0:
            return rng.choice(['A', 'B', 'Mixed'])

        a_count = sum(1 for source in param_inherit_map.values() if source == 'A')
        b_count = total_params - a_count

        ratio_a = a_count / total_params
        ratio_b = b_count / total_params

        # Только при значительном перевесе (70%) присваиваем конкретного родителя
        if ratio_a > 0.7:
            return 'A'
        elif ratio_b > 0.7:
            return 'B'
        else:
            # В остальных случаях - Mixed
            return 'Mixed'

    def _open_children_browser(self):
        if not self.generated_children:
            messagebox.showwarning("Нет детей", "Сначала выполните кроссинговер и создайте детей.")
            return
        win = tk.Toplevel(self)
        win.title("Дети — выбор для инспекции")
        win.geometry("520x420")
        win.transient(self)

        lb = tk.Listbox(win, width=80, height=20)
        lb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        for idx, c in enumerate(self.generated_children):
            fname = os.path.basename(c.get('child_filename', f'child_{idx + 1}.pkl'))
            total = sum(c.get('scores', {}).values()) if c.get('scores') else 0.0
            lb.insert(tk.END, f"{idx + 1:03d}. {fname}  (sum:{total:.3f})")

        def open_selected():
            sel = lb.curselection()
            if not sel:
                return
            idx = sel[0]
            child = self.generated_children[idx]
            InspectorWindow(self, child, self.selector)

        ttk.Button(win, text="Открыть", command=open_selected).pack(pady=(0, 8))

# -------------------------
# === InspectorWindow
# -------------------------
class InspectorWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, child: dict, selector: ModelSelector):
        super().__init__(parent)
        self.title("Inspector — Child")
        self.geometry("1200x700")
        self.child = child
        self.selector = selector

        # Создаем notebook для вкладок
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Вкладка 1: Метрики
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="Метрики")
        self._create_metrics_tab(metrics_frame)

        # Вкладка 2: Параметры
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Параметры")
        self._create_params_tab(params_frame)

        ttk.Button(self, text="Закрыть", command=self.destroy).pack(pady=6)

    def _create_metrics_tab(self, parent):
        """Вкладка с метриками и информацией о наследовании"""
        title = os.path.basename(self.child.get('child_filename', 'child.pkl'))
        ttk.Label(parent, text=f"Child: {title}", font=("Arial", 14, "bold")).pack(pady=(8,0))

        pa = self.child.get('parents', {}).get('A', 'A_unknown')
        pb = self.child.get('parents', {}).get('B', 'B_unknown')
        ttk.Label(parent, text=f"Parent A: {pa}    |    Parent B: {pb}", font=("Arial", 10)).pack(pady=(0, 8))

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        # Обновленные колонки - убираем A/B значения родителей, оставляем только источник
        columns = ("metric", "score", "level", "status", "inherited_from")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=15)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tree.heading("metric", text="Метрика")
        tree.heading("score", text="Оценка (0–1)")
        tree.heading("level", text="Уровень")
        tree.heading("status", text="Статус")
        tree.heading("inherited_from", text="Родитель-источник")

        tree.column("metric", width=220)
        tree.column("score", width=90, anchor='center')
        tree.column("level", width=150, anchor='center')
        tree.column("status", width=110, anchor='center')
        tree.column("inherited_from", width=120, anchor='center')

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        scores = self.child.get('scores', {})
        inherit_map = self.child.get('inherit_map', {})

        for m, s in scores.items():
            s_f = float(s)
            level = self._level_from_score(s_f)
            status = self._status_from_score(s_f)
            source = inherit_map.get(m, '?')

            tree.insert("", tk.END, values=(m, f"{s_f:.3f}", level, status, source))

        summary = ttk.Frame(parent, padding=8)
        summary.pack(fill=tk.X)
        ttk.Label(summary, text=self._make_summary_text(self.child), justify=tk.LEFT).pack(anchor=tk.W)

        inherit_map = self.child.get('inherit_map', {})
        inheritance_stats = {
            'A': sum(1 for v in inherit_map.values() if v == 'A'),
            'B': sum(1 for v in inherit_map.values() if v == 'B'),
            'Mixed': sum(1 for v in inherit_map.values() if v == 'Mixed')
        }

        summary_text = self._make_summary_text(self.child)
        summary_text += f"\nНаследование: A={inheritance_stats['A']}, B={inheritance_stats['B']}, Mixed={inheritance_stats['Mixed']}"

        ttk.Label(summary, text=summary_text, justify=tk.LEFT).pack(anchor=tk.W)

    def _create_params_tab(self, parent):
        """Новая вкладка с детальной информацией о наследовании параметров"""
        ttk.Label(parent, text="Детальное наследование параметров", font=("Arial", 12, "bold")).pack(pady=(8, 0))

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        columns = ("parameter", "value", "inherited_from")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=20)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tree.heading("parameter", text="Параметр")
        tree.heading("value", text="Значение")
        tree.heading("inherited_from", text="От родителя")

        tree.column("parameter", width=400)
        tree.column("value", width=200)
        tree.column("inherited_from", width=100, anchor='center')

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Заполняем реальными параметрами
        param_inherit_map = self.child.get('param_inherit_map', {})
        child_pipeline = self.child.get('pipeline')

        if child_pipeline and hasattr(child_pipeline, 'get_params'):
            child_params = child_pipeline.get_params()

            for param_name, source in param_inherit_map.items():
                if param_name in child_params:
                    value = child_params[param_name]
                    # Форматируем значение для отображения
                    value_str = self._format_param_value(value)

                    tree.insert("", tk.END, values=(
                        param_name,
                        value_str,
                        source
                    ))


    def _format_param_value(self, value):
        """Форматирует значение параметра для отображения"""
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) > 3:
                return f"{type(value).__name__}[{len(value)}]"
            else:
                return str(value)
        elif isinstance(value, str) and len(value) > 50:
            return value[:50] + "..."
        else:
            return str(value)

    def _level_from_score(self, s: float) -> str:
        if s >= 0.9:
            return "Очень хорошо"
        if s >= 0.7:
            return "Хорошо"
        if s >= 0.5:
            return "Удовлетворительно"
        if s >= 0.3:
            return "Плохо"
        return "Очень плохо"

    def _status_from_score(self, s: float) -> str:
        if s >= 0.7:
            return "Сильная"
        if s >= 0.5:
            return "Средняя"
        return "Слабая"

    def _make_summary_text(self, child: dict) -> str:
        total = sum(child.get('scores', {}).values()) if child.get('scores') else 0.0
        strong = sum(1 for v in child.get('scores', {}).values() if v >= 0.7)
        s = f"Создан: {child.get('created_at', 'N/A')} | Сумма оценок: {total:.3f} | Сильных метрик: {strong}\n"
        top_child = sorted(child.get('scores', {}).items(), key=lambda x: x[1], reverse=True)[:5]
        s += "Топ-метрики ребёнка: " + ", ".join([f"{k}:{v:.3f}" for k, v in top_child])
        return s

# ниже добавил
def test_mutation():
    """Тестируем мутацию на примере"""
    test_params = {
        'n_estimators': 100,  # int
        'learning_rate': 0.1,  # float
        'activation': 'relu',  # categorical
        'hidden_layer_sizes': (100, 50),  # tuple
        'early_stopping': True  # bool
    }

    rng = random.Random(42)

    print("До мутации:", test_params)
    for i in range(50):
        mutated = mutate_params(test_params.copy(), rng, mutation_rate=0.02)
        print(f"После мутации {i + 1}:", mutated)
        # Проверяем, что значения изменились
        for key in test_params:
            if test_params[key] != mutated[key]:
                print(f"  {key} изменился: {test_params[key]} -> {mutated[key]}")

# ----------------------
# --- Main
# ----------------------
def main():
    # ниже добавил для теста
    # test_mutation()

    app = CrossBreedingApp()
    app.mainloop()

if __name__ == "__main__":
    main()
