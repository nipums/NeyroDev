#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LastNewNeyro.py
Упрощённый и переработанный скрипт с GUI для создания "умной" нейросети и генерации популяции сетей.
Использует встроенные словари suspicious / exec weights и оставляет необходимые функции.
Запуск:
    python LastNewNeyro.py
"""
import datetime
import os
import pickle
import queue
import threading
import re
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, \
    classification_report
from sklearn.base import clone

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# БОЛЬШЕ РАНДОМА
from random import uniform

SUSPICIOUS_WORDS = {
    "secure": uniform(0,1.5),
    "login": uniform(0,1.5),
    "verify": uniform(0,1.5),
    "bank": uniform(0,1.5),
    "account": uniform(0,1.5),
    "update": uniform(0,1.5),
    "confirm": uniform(0,1.5),
    "paypal": uniform(0,1.5)
}

EXEC_WEIGHTS = {
    "exe": uniform(0,1.5),
    "bat": uniform(0,1.5),
    "cmd": uniform(0,1.5),
    "sh": uniform(0,1.5),
    "scr": uniform(0,1.5),
    "pif": uniform(0,1.5),
    "jar": uniform(0,1.5),
    "msi": uniform(0,1.5),
    "com": uniform(0,1.5),
    "vbs": uniform(0,1.5)
}

# ------------------------
# === ПРИЗНАКИ URL/ДОМЕНА ===
# ------------------------
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
            float(len_url),  # 0
            float(len_host),  # 1
            float(cnt_dots),  # 2
            float(cnt_hyphen),  # 3
            float(cnt_digits),  # 4
            float(digit_ratio),  # 5
            float(subdomains),  # 6
            float(is_ip),  # 7
            float(susp_count),  # 8
            float(susp_weighted),  # 9
            float(ext_flag),  # 10
            float(ext_weight),  # 11
            float(has_at),  # 12
            float(cnt_double_slash),  # 13
            float(entropy_host),  # 14
            float(ratio_special),  # 15
            float(cnt_slash),  # 16
            float(starts_http),  # 17
            float(num_letters),  # 18
            float(ratio_vowels)  # 19
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
        # Исправленная версия без предупреждения
        def sample_group(group, n_samples, total_len):
            sample_size = max(1, int(n_samples * len(group) / total_len))
            return group.sample(sample_size, random_state=42) if len(group) > 0 else group

        df_sampled = df.groupby('label', group_keys=False).apply(
            lambda x: sample_group(x, n_samples, len(df))
        )
        # Если получилось слишком много сэмплов, обрезаем до нужного количества
        if len(df_sampled) > n_samples:
            df_sampled = df_sampled.sample(n_samples, random_state=42)
        return df_sampled.reset_index(drop=True)
    except Exception:
        return df.sample(min(n_samples, len(df)), random_state=42).reset_index(drop=True)


# ------------------------
# === TRAIN & EVAL ===
# ------------------------
from sklearn.ensemble import RandomForestClassifier


def build_pipeline(hidden_layers=(128, 64), max_iter: int = 300, random_state: int = 42) -> Pipeline:
    # Для RF используем max_iter как n_estimators, но с большим разбросом
    n_estimators = max(10, min(max_iter, 500))

    # Базовые параметры RF
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,  # Будет варьироваться в популяции
            max_features='sqrt',
            random_state=random_state,
            bootstrap=True,
            oob_score=False
        ))
    ])
    return pipeline


def train_and_evaluate(X: np.ndarray, y: np.ndarray, hidden_layers=(128, 64), max_iter: int = 300,
                       random_state: int = 42):
    pipeline = build_pipeline(hidden_layers=hidden_layers, max_iter=max_iter, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0))
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    return pipeline, metrics, cm, report, (X_train, X_test, y_train, y_test)

# ------------------------
# === Генетическая популяция ===
# ------------------------
class GeneticPopulation:
    def __init__(self, base_pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray, pop_size: int = 10, random_state: int = 42):
        self.base_pipeline = base_pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.pop_size = int(pop_size)
        self.random_state = random_state
        self.population: List[Tuple[Pipeline, float, str]] = []
        self.model_type = self._detect_model_type()

    def _detect_model_type(self):
        if 'rf' in self.base_pipeline.named_steps:
            return 'rf'
        elif 'mlp' in self.base_pipeline.named_steps:
            return 'mlp'
        else:
            return 'unknown'

    def create_population(self):
        self.population = []

        # 1. Базовая модель (обучаем один раз)
        base_model = clone(self.base_pipeline)
        base_model.fit(self.X_train, self.y_train)
        f1_base = self._evaluate_f1(base_model)
        self.population.append((base_model, f1_base, "base"))

        # 2. Создаем разнообразные модели с прогресс-логом
        for i in range(self.pop_size - 1):
            if self.model_type == 'rf':
                model = self._create_fast_rf(i)
            elif self.model_type == 'mlp':
                model = self._create_fast_mlp(i)
            else:
                model = self._create_fast_generic(i)

            f1 = self._evaluate_f1(model)
            self.population.append((model, f1, f"var_{i + 1}"))

            # Логируем прогресс
            print(f"Создана модель {i + 1}/{self.pop_size - 1}, F1: {f1:.4f}")

        return self.population

    def _create_fast_rf(self, seed_offset: int):
        """Быстрое создание Random Forest"""
        # ОПТИМАЛЬНЫЕ ДИАПАЗОНЫ ДЛЯ СКОРОСТИ:
        n_estimators = np.random.choice([10, 20, 30])  # МАЛО деревьев
        max_depth = np.random.choice([1, 2, 3, 5,])  # ОГРАНИЧЕННАЯ глубина
        max_features = np.random.choice([3, 5, 'sqrt'])  # ОГРАНИЧЕННЫЕ признаки

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_split=2,  # ФИКСИРОВАННО - быстрее
                min_samples_leaf=1,  # ФИКСИРОВАННО - быстрее
                bootstrap=True,
                random_state=self.random_state + seed_offset,
                n_jobs=-1,  # ПАРАЛЛЕЛИЗМ - ВКЛЮЧАЕМ!
                verbose=0
            ))
        ])

        # ЖЕСТКОЕ ОГРАНИЧЕНИЕ ДАННЫХ
        max_training_samples = 10000  # МАКСИМУМ 1000 примеров
        if len(self.X_train) > max_training_samples:
            indices = np.random.choice(len(self.X_train), max_training_samples, replace=False)
            X_sample = self.X_train[indices]
            y_sample = self.y_train[indices]
        else:
            X_sample = self.X_train
            y_sample = self.y_train

        pipeline.fit(X_sample, y_sample)
        return pipeline

    def _create_fast_mlp(self, seed_offset: int):
        """Быстрое создание MLP"""
        # Простые архитектуры для скорости
        architectures = [(10,), (20,), (10, 5), (20, 10)]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=np.random.choice(architectures),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=100,  # МАЛО итераций
                random_state=self.random_state + seed_offset,
                early_stopping=False  # Выключаем для скорости
            ))
        ])

        # Ограничиваем данные
        max_training_samples = 1000
        if len(self.X_train) > max_training_samples:
            indices = np.random.choice(len(self.X_train), max_training_samples, replace=False)
            X_sample = self.X_train[indices]
            y_sample = self.y_train[indices]
        else:
            X_sample = self.X_train
            y_sample = self.y_train

        pipeline.fit(X_sample, y_sample)
        return pipeline

    def _create_fast_generic(self, seed_offset: int):
        """Быстрая универсальная модель"""
        model = clone(self.base_pipeline)

        # Ограничиваем данные
        max_training_samples = 1000
        if len(self.X_train) > max_training_samples:
            indices = np.random.choice(len(self.X_train), max_training_samples, replace=False)
            X_sample = self.X_train[indices]
            y_sample = self.y_train[indices]
        else:
            X_sample = self.X_train
            y_sample = self.y_train

        model.fit(X_sample, y_sample)
        return model

    def _evaluate_f1(self, pipeline: Pipeline) -> float:
        try:
            y_pred = pipeline.predict(self.X_test)
            return float(f1_score(self.y_test, y_pred, zero_division=0))
        except Exception:
            return 0.0

    def get_f1_list(self) -> List[float]:
        return [round(f1, 4) for (_, f1, _) in self.population]

    def get_labels(self) -> List[str]:
        return [label for (_, _, label) in self.population]


# ------------------------
# === GUI Приложение ===
# ------------------------
class NeyroApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Genetic-like NN Explorer")
        self.geometry("1000x700")
        self.resizable(True, True)

        # State
        self.model_pipeline: Pipeline | None = None  # последний обученный pipeline
        self.model_info = None  # metrics/report
        self.population_obj: GeneticPopulation | None = None
        self.save_dir = os.getcwd()

        # GUI vars
        self.data_path = tk.StringVar()
        self.n_samples = tk.IntVar(value=100000)
        self.hidden_layers = tk.StringVar(value="16,8")
        self.max_iter = tk.IntVar(value=1)
        self.pop_size = tk.IntVar(value=10)
        self.selected_pop_index = None  # индекс выбранной модели в популяции

        # log queue
        self.log_queue = queue.Queue()

        self._build_ui()
        self._after_process_logs()

    def _build_ui(self):
        # --- Top frame: data and params ---
        top = ttk.Frame(self, padding=(8, 8))
        top.pack(fill=tk.X)

        # Строка 1: CSV файл
        ttk.Label(top, text="CSV (url,label):").grid(row=0, column=0, sticky=tk.W)
        self.entry_data = ttk.Entry(top, textvariable=self.data_path, width=50)
        self.entry_data.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(top, text="Открыть CSV", command=self._choose_csv).grid(row=0, column=2, padx=5)

        # Строка 2: Параметры в одну линию
        ttk.Label(top, text="Кол-во примеров:").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(top, textvariable=self.n_samples, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=(8, 0))

        ttk.Label(top, text="Hidden layers:").grid(row=1, column=2, sticky=tk.W, pady=(8, 0))
        ttk.Entry(top, textvariable=self.hidden_layers, width=15).grid(row=1, column=3, sticky=tk.W, padx=(5, 15),
                                                                       pady=(8, 0))

        ttk.Label(top, text="Max iter:").grid(row=1, column=4, sticky=tk.W, pady=(8, 0))
        ttk.Entry(top, textvariable=self.max_iter, width=8).grid(row=1, column=5, sticky=tk.W, padx=5, pady=(8, 0))

        ttk.Label(top, text="Pop size:").grid(row=1, column=6, sticky=tk.W, pady=(8, 0))
        ttk.Entry(top, textvariable=self.pop_size, width=6).grid(row=1, column=7, sticky=tk.W, padx=5, pady=(8, 0))

        # --- Buttons ---
        btn_frame = ttk.Frame(self, padding=(8, 6))
        btn_frame.pack(fill=tk.X)
        self.btn_train = ttk.Button(btn_frame, text="📦 Обучить модель", command=self._on_train)
        self.btn_train.grid(row=0, column=0, padx=6)
        self.btn_population = ttk.Button(btn_frame, text="🧬 Создать популяцию", command=self._on_create_population)
        self.btn_population.grid(row=0, column=1, padx=6)
        self.btn_save = ttk.Button(btn_frame, text="💾 Сохранить выбранную модель", command=self._on_save_selected)
        self.btn_save.grid(row=0, column=2, padx=6)
        self.btn_load = ttk.Button(btn_frame, text="📂 Загрузить модель (.pkl)", command=self._on_load_model)
        self.btn_load.grid(row=0, column=3, padx=6)
        self.btn_reset = ttk.Button(btn_frame, text="♻️ Сбросить всё", command=self._on_reset)
        self.btn_reset.grid(row=0, column=4, padx=6)

        # --- Metrics / status ---
        metrics_frame = ttk.LabelFrame(self, text="Последние метрики", padding=(8, 8))
        metrics_frame.pack(fill=tk.X, padx=8, pady=6)
        self.metric_text = tk.StringVar(value="Модель не обучена.")
        ttk.Label(metrics_frame, textvariable=self.metric_text, justify=tk.LEFT).pack(anchor=tk.W)

        # --- Main area: chart and list ---
        main_frame = ttk.Frame(self, padding=(8, 8))
        main_frame.pack(fill=tk.BOTH, expand=True)

        # chart area - увеличен
        self.chart_frame = ttk.LabelFrame(main_frame, text="Популяция (F1-scores)", padding=(6, 6))
        self.chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # list area (models) - увеличен
        list_frame = ttk.LabelFrame(main_frame, text="Модели в популяции (клик для выбора)", padding=(6, 6), width=280)
        list_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_listbox = tk.Listbox(list_frame, width=35, height=25)
        self.model_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.model_listbox.bind('<<ListboxSelect>>', self._on_model_select)

        # --- Logs --- уменьшены
        log_frame = ttk.LabelFrame(self, text="Логи", padding=(6, 6))
        log_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=6)
        self.logbox = scrolledtext.ScrolledText(log_frame, height=4)  # уменьшена высота
        self.logbox.pack(fill=tk.BOTH, expand=True)

        # status bar
        self.status_var = tk.StringVar(value="Готов")
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM)

    # -------------------------
    # === UI Helpers ===
    # -------------------------
    def _choose_csv(self):
        p = filedialog.askopenfilename(title="Выберите CSV (url,label)",
                                       filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self.data_path.set(p)

    def _log(self, text: str):
        self.log_queue.put(text + "\n")

    def _after_process_logs(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                self.logbox.insert(tk.END, item)
                self.logbox.see(tk.END)
        except queue.Empty:
            pass
        self.after(200, self._after_process_logs)

    # -------------------------
    # === Train flow ===
    # -------------------------
    def _on_train(self):
        data_p = self.data_path.get().strip()
        if not data_p or not os.path.exists(data_p):
            messagebox.showerror("Ошибка", "Выберите корректный CSV файл с данными (url,label).")
            return
        try:
            n_samples = int(self.n_samples.get())
        except:
            n_samples = 10000
        try:
            hidden_layers = tuple(int(x.strip()) for x in self.hidden_layers.get().split(",") if x.strip())
            if len(hidden_layers) == 0:
                hidden_layers = (128, 64)
        except:
            hidden_layers = (128, 64)
        try:
            max_iter = int(self.max_iter.get())
        except:
            max_iter = 300

        # disable buttons
        self._set_buttons_state('disabled')
        self.status_var.set("Обучение: идёт подготовка...")
        t = threading.Thread(target=self._train_thread, args=(data_p, n_samples, hidden_layers, max_iter), daemon=True)
        t.start()

    def _train_thread(self, data_p, n_samples, hidden_layers, max_iter):
        try:
            self._log(f"[TRAIN] Загружаю dataset: {data_p}")
            df = load_dataset(data_p, n_samples=n_samples)
            self._log(f"[TRAIN] Всего примеров: {len(df)} (class1: {int(df['label'].sum())})")

            self._log("[TRAIN] Извлечение признаков...")
            X_list = []
            for i, u in enumerate(df['url'].tolist()):
                feats = extract_features(u)
                X_list.append(feats)
                if (i + 1) % 2000 == 0:
                    self._log(f"[TRAIN] processed {i + 1}/{len(df)}")
            X = np.array(X_list, dtype=float)
            y = df['label'].values.astype(int)

            self._log(f"[TRAIN] Обучаю модель (hidden={self.hidden_layers.get()}, max_iter={max_iter}) ...")
            model, metrics, cm, report, data_split = train_and_evaluate(X, y, hidden_layers=tuple(
                int(x.strip()) for x in self.hidden_layers.get().split(",") if x.strip()), max_iter=max_iter)
            self.model_pipeline = model
            self.model_info = (metrics, cm, report)

            # update UI (main thread)
            def ui_update():
                txt = (f"accuracy: {metrics['accuracy']:.4f}\n"
                       f"precision: {metrics['precision']:.4f}\n"
                       f"recall: {metrics['recall']:.4f}\n"
                       f"f1: {metrics['f1']:.4f}\n")
                self.metric_text.set(txt)
                self._log("[TRAIN] Confusion matrix:\n" + str(cm))
                self._log("[TRAIN] Classification report:\n" + report)
                self.status_var.set("Обучение завершено.")
                self._set_buttons_state('normal')

            # keep split data for later population generation
            self._last_data_split = data_split
            self.after(100, ui_update)
        except Exception as e:
            self._log(f"[ERROR] during training: {e}")

            def ui_err():
                self.status_var.set("Ошибка при обучении. Смотри лог.")
                self._set_buttons_state('normal')

            self.after(100, ui_err)

    def _set_buttons_state(self, state: str):
        for b in (self.btn_train, self.btn_population, self.btn_save, self.btn_load):
            try:
                b.config(state=state)
            except:
                pass

    # -------------------------
    # === Create population ===
    # -------------------------
    def _on_create_population(self):
        if self.model_pipeline is None:
            messagebox.showwarning("Нет модели", "Сначала обучите модель (или загрузите).")
            return
        if not hasattr(self, '_last_data_split') or self._last_data_split is None:
            messagebox.showwarning("Нет данных", "Сначала обучите модель, чтобы инициализировать данные для популяции.")
            return

        # read pop size
        try:
            pop_size = int(self.pop_size.get())
            if pop_size < 2:
                pop_size = 10
        except:
            pop_size = 10

        # disable
        self._set_buttons_state('disabled')
        self.status_var.set("Создание популяции...")
        t = threading.Thread(target=self._create_population_thread, args=(pop_size,), daemon=True)
        t.start()

    def _create_population_thread(self, pop_size):
        try:
            X_train, X_test, y_train, y_test = self._last_data_split
            gp = GeneticPopulation(self.model_pipeline, X_train, y_train, X_test, y_test, pop_size=pop_size)

            # Логируем начало
            self._log(f"[GENETIC] Создание {pop_size} моделей...")

            population = gp.create_population()
            self.population_obj = gp

            f1s = gp.get_f1_list()
            labels = gp.get_labels()
            self._log(f"[GENETIC] Популяция создана. F1: {f1s}")

            def ui_update():
                self._render_population_chart(f1s)
                self._populate_listbox(labels, f1s)
                self.status_var.set("Популяция готова.")
                self._set_buttons_state('normal')

            self.after(100, ui_update)
        except Exception as e:
            self._log(f"[ERROR] при создании популяции: {e}")

            def ui_err():
                self.status_var.set("Ошибка при создании популяции.")
                self._set_buttons_state('normal')

            self.after(100, ui_err)

    def _on_reset(self):
        """Полный сброс состояния"""
        self.model_pipeline = None
        self.model_info = None
        self.population_obj = None
        self.selected_pop_index = None
        self._last_data_split = None
        # очистка GUI
        self.metric_text.set("Модель не обучена.")
        self.status_var.set("Сброшено.")
        self.model_listbox.delete(0, tk.END)
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        self.logbox.delete(1.0, tk.END)
        self._log("[RESET] Всё очищено.")

    # -------------------------
    # === Chart & List Helpers ===
    # -------------------------
    def _render_population_chart(self, f1_scores: List[float]):
        # очистка рамки
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        # создаём график
        fig, ax = plt.subplots(figsize=(6, 4))
        indices = list(range(1, len(f1_scores) + 1))
        bars = ax.bar(indices, f1_scores)
        ax.set_xlabel("№ сети")
        ax.set_ylabel("F1-score")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Популяция нейросетей — F1")
        for i, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{f1_scores[i]:.3f}", ha='center',
                    va='bottom', fontsize=9)
        fig.tight_layout()

        # Встраиваем в tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)

    def _populate_listbox(self, labels: List[str], f1_scores: List[float]):
        self.model_listbox.delete(0, tk.END)
        for i, (lab, f1) in enumerate(zip(labels, f1_scores)):
            display = f"{i + 1:02d}. [{lab}] F1={f1:.4f}"
            self.model_listbox.insert(tk.END, display)
        self.selected_pop_index = None

    def _on_model_select(self, event):
        sel = event.widget.curselection()
        if not sel:
            self.selected_pop_index = None
            return
        idx = sel[0]
        self.selected_pop_index = idx
        self._log(f"[UI] Выбрана модель #{idx + 1}")

    # -------------------------
    # === Save / Load модель ===
    # -------------------------
    def _on_save_selected(self):
        if self.population_obj is None:
            messagebox.showwarning("Нет популяции", "Сначала создайте популяцию.")
            return

        # Спрашиваем папку для сохранения всей популяции
        folder_path = filedialog.askdirectory(title="Выберите папку для сохранения всей популяции")
        if not folder_path:
            return

        try:
            # Создаем подпапку с временной меткой для организации
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pop_folder = os.path.join(folder_path, f"population_{timestamp}")
            os.makedirs(pop_folder, exist_ok=True)

            total_models = len(self.population_obj.population)
            saved_count = 0

            # Сохраняем каждую модель популяции
            for i, (pipeline, f1, label) in enumerate(self.population_obj.population):
                try:
                    # Создаем имя файла для каждой модели
                    filename = f"model_{i + 1:02d}_{label}_f1_{f1:.4f}.pkl"
                    filepath = os.path.join(pop_folder, filename)

                    # Сохраняем модель
                    with open(filepath, 'wb') as f:
                        pickle.dump({
                            'pipeline': pipeline,
                            'f1': f1,
                            'label': label,
                            'index': i,
                            'timestamp': timestamp
                        }, f)

                    saved_count += 1
                    self._log(f"[SAVE] Сохранена модель {i + 1}/{total_models}: {filename}")

                except Exception as e:
                    self._log(f"[ERROR] Ошибка при сохранении модели {i + 1}: {e}")

            # Сохраняем общую информацию о популяции
            pop_info = {
                'total_models': total_models,
                'saved_models': saved_count,
                'timestamp': timestamp,
                'f1_scores': self.population_obj.get_f1_list(),
                'labels': self.population_obj.get_labels(),
                'population_size': self.population_obj.pop_size
            }

            info_filepath = os.path.join(pop_folder, "population_info.pkl")
            with open(info_filepath, 'wb') as f:
                pickle.dump(pop_info, f)

            # Также сохраняем информацию в текстовом формате для удобства
            info_txt_filepath = os.path.join(pop_folder, "population_info.txt")
            with open(info_txt_filepath, 'w', encoding='utf-8') as f:
                f.write(f"Информация о популяции нейросетей\n")
                f.write(f"================================\n")
                f.write(f"Время создания: {timestamp}\n")
                f.write(f"Всего моделей: {total_models}\n")
                f.write(f"Успешно сохранено: {saved_count}\n")
                f.write(f"Размер популяции: {self.population_obj.pop_size}\n\n")
                f.write("Список моделей:\n")
                f.write("№ | Метка | F1-score\n")
                f.write("--|-------|----------\n")
                for i, (label, f1) in enumerate(
                        zip(self.population_obj.get_labels(), self.population_obj.get_f1_list())):
                    f.write(f"{i + 1:2d} | {label:6} | {f1:.4f}\n")

            self._log(f"[SAVE] Вся популяция сохранена в папку: {pop_folder}")
            self._log(f"[SAVE] Сохранено {saved_count}/{total_models} моделей")

            messagebox.showinfo("Сохранено",
                                f"Вся популяция сохранена!\n"
                                f"Папка: {pop_folder}\n"
                                f"Моделей: {saved_count}/{total_models}\n"
                                f"Файлы:\n"
                                f"- model_XX_*.pkl - отдельные модели\n"
                                f"- population_info.pkl - общая информация\n"
                                f"- population_info.txt - читаемая информация")

        except Exception as e:
            self._log(f"[ERROR] Ошибка при сохранении популяции: {e}")
            messagebox.showerror("Ошибка", f"Не удалось сохранить популяцию: {e}")

    def _on_load_model(self):
        p = filedialog.askopenfilename(title="Загрузить модель (.pkl)", filetypes=[("Pickle", "*.pkl"), ("All", "*.*")])
        if not p:
            return
        try:
            with open(p, 'rb') as f:
                data = pickle.load(f)
            pipeline = data.get('pipeline') if isinstance(data, dict) else data
            label = data.get('label', 'loaded') if isinstance(data, dict) else 'loaded'
            f1 = data.get('f1', None) if isinstance(data, dict) else None
            # Добавляем в текущую популяцию (если нет — создаём пустую)
            if self.population_obj is None:
                # нужен X_train/X_test — если их нет, не можем оценить F1, но добавим в listbox как загруженную
                if hasattr(self, '_last_data_split') and self._last_data_split is not None:
                    X_train, X_test, y_train, y_test = self._last_data_split
                    # создаём временный gp для хранения
                    self.population_obj = GeneticPopulation(pipeline, X_train, y_train, X_test, y_test, pop_size=1)
                    # вручную поместим pipeline в population
                    evaluated_f1 = None
                    try:
                        evaluated_f1 = self.population_obj._evaluate_f1(pipeline)
                    except:
                        evaluated_f1 = f1 if f1 is not None else 0.0
                    self.population_obj.population = [(pipeline, float(evaluated_f1), label)]
                    self._populate_listbox(self.population_obj.get_labels(), self.population_obj.get_f1_list())
                    self._render_population_chart(self.population_obj.get_f1_list())
                    self._log(f"[MODEL] Загружена модель и добавлена в популяцию (F1={evaluated_f1})")
                    messagebox.showinfo("Загружено", "Модель загружена и добавлена в популяцию.")
                    return
                else:
                    # нет данных — создаём временную структуру и добавляем элемент в listbox (без F1)
                    self.population_obj = GeneticPopulation(pipeline, np.zeros((1, URL_FEATURES_COUNT)), np.array([0]),
                                                            np.zeros((1, URL_FEATURES_COUNT)), np.array([0]),
                                                            pop_size=1)
                    self.population_obj.population = [(pipeline, float(f1) if f1 is not None else 0.0, label)]
                    self._populate_listbox(self.population_obj.get_labels(), self.population_obj.get_f1_list())
                    self._render_population_chart(self.population_obj.get_f1_list())
                    self._log(f"[MODEL] Загружена модель (без оценки).")
                    messagebox.showinfo("Загружено", "Модель загружена (без оценки, нет данных).")
                    return
            # если есть популяция, просто добавляем
            evaluated_f1 = None
            try:
                evaluated_f1 = self.population_obj._evaluate_f1(pipeline)
            except:
                evaluated_f1 = float(f1) if f1 is not None else 0.0
            self.population_obj.population.append((pipeline, float(evaluated_f1), label))
            self._populate_listbox(self.population_obj.get_labels(), self.population_obj.get_f1_list())
            self._render_population_chart(self.population_obj.get_f1_list())
            self._log(f"[MODEL] Загружена модель {p} и добавлена в популяцию (F1={evaluated_f1})")
            messagebox.showinfo("Загружено", f"Модель загружена и добавлена (F1={evaluated_f1}).")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")


# ------------------------
# === MAIN ===
# ------------------------
def main():
    app = NeyroApp()
    app.mainloop()


if __name__ == '__main__':
    main()
