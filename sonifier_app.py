"""
Sonifier v3.0 — PyQt5 Application

A real-time market sonification tool built for active analysis.
Designed for use, not training.

Usage:
    python sonifier_app.py

Requirements:
    pip install numpy pandas yfinance pyqt5 sounddevice scipy
"""

import sys
import os
import time
import threading
import json
from datetime import datetime

import numpy as np
import pandas as pd
import sounddevice as sd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QGridLayout, QSlider, QCheckBox, QTextEdit, QProgressBar,
    QSplitter, QFrame, QFileDialog, QMessageBox, QTabWidget, QScrollArea,
    QSizePolicy, QToolButton,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

# Import our engine
from audio_engine import AudioGenerator, AlertEvent, _normalize
from data_fetcher import fetch_ohlcv

import config


# ═══════════════════════════════════════════════════════════════════
#  WORKER THREAD — data fetch + synthesis in background
# ═══════════════════════════════════════════════════════════════════

class GenerationWorker(QThread):
    progress = pyqtSignal(int, int)     # current, total
    finished = pyqtSignal(np.ndarray, list, dict, str)
    error    = pyqtSignal(str)
    status   = pyqtSignal(str)

    def __init__(self, ticker, period, interval, preset_cfg):
        super().__init__()
        self.ticker       = ticker
        self.period       = period
        self.interval     = interval
        self.preset_cfg   = preset_cfg

    def run(self):
        try:
            self.status.emit("Fetching data…")
            data = fetch_ohlcv(self.ticker, self.period, self.interval)
            if data.empty:
                self.error.emit("No data returned.")
                return

            self.progress.emit(0, len(data))

            gen = AudioGenerator(
                scale               = self.preset_cfg['scale'],
                tonic_freq          = self.preset_cfg['tonic_freq'],
                freq_min            = self.preset_cfg['freq_min'],
                freq_max            = self.preset_cfg['freq_max'],
                duration_per_note   = self.preset_cfg['duration_ms'] / 1000.0,
                sr                  = 44100,
                smooth              = self.preset_cfg['smooth'],
                waveform_name       = self.preset_cfg['waveform'],
                effect              = self.preset_cfg['effect'],
                vibrato_cents       = self.preset_cfg['vibrato_max_cents'],
                sustain_min         = self.preset_cfg['sustain_min'],
                sustain_max         = self.preset_cfg['sustain_max'],
                silence_body        = self.preset_cfg['silence_threshold_body'],
                silence_range       = self.preset_cfg['silence_threshold_range'],
                heartbeat           = self.preset_cfg['heartbeat_enabled'],
                bass                = self.preset_cfg['bass_enabled'],
                bass_lo             = self.preset_cfg['bass_freq_min'],
                bass_hi             = self.preset_cfg['bass_freq_max'],
                bass_every          = self.preset_cfg['bass_note_every'],
                ref_mode            = self.preset_cfg['reference_mode'],
                ref_window          = self.preset_cfg['reference_window'],
                speed               = self.preset_cfg['speed'],
                alerts              = {
                    'breakout':     self.preset_cfg.get('alert_on_breakout', False),
                    'divergence':   self.preset_cfg.get('alert_on_divergence', False),
                    'volume_spike': self.preset_cfg.get('alert_on_volume_spike', False),
                    'spike_std':    self.preset_cfg.get('volume_spike_threshold', 2.0),
                    'reversal':     self.preset_cfg.get('alert_on_reversal', False),
                },
            )

            self.status.emit("Generating audio…")
            audio, events = gen.generate(data, target_speed=self.preset_cfg.get('speed', 1.0))

            event_summary = {}
            for ev in events:
                event_summary[ev.candle_index] = ev.description

            self.finished.emit(
                audio, events, data.to_dict('index'), gen.__class__.__name__
            )

        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════

class SonifierMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sonifier v3.0 — Market Sonification Tool")
        self.setMinimumSize(720, 860)
        self.resize(900, 920)
        self._dark_theme()

        self.current_audio = np.array([], dtype=np.float32)
        self.events: list = []
        self.play_start_idx = 0
        self.is_playing = False
        self.stream = None

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)

        # ── Top bar: Ticker + Timeframe + Generate ───────────
        top = QHBoxLayout()

        QLabel("Ticker:", self).setToolTip("Yahoo Finance symbol, e.g. BTC-USD AAPL ^GSPC ETH-USD")
        self.ticker_input = QLineEdit("BTC-USD")
        self.ticker_input.setFixedWidth(120)
        self.ticker_input.setFont(QFont("Consolas", 11))
        top.addWidget(self.ticker_input)

        QLabel("Period:", self)
        self.period_combo = QComboBox(self)
        for label in config.TIMEFRAME_OPTIONS:
            self.period_combo.addItem(label)
        self.period_combo.setCurrentText("3M")
        top.addWidget(self.period_combo)

        QLabel("Preset:", self)
        self.preset_combo = QComboBox(self)
        for name, cfg in config.PRESETS.items():
            desc = cfg.get('description', name)
            self.preset_combo.addItem(f"{name}")
            self.preset_combo.setItemData(self.preset_combo.count() - 1, desc, Qt.ToolTipRole)
        self.preset_combo.setCurrentText(config.DEFAULT_PRESET)
        top.addWidget(self.preset_combo)

        self.generate_btn = QPushButton("▶ Generate & Play")
        self.generate_btn.setFont(QFont("Sans-serif", 10, QFont.Bold))
        self.generate_btn.clicked.connect(self._on_generate)
        top.addWidget(self.generate_btn)

        self.export_btn = QPushButton("💾 Export WAV")
        self.export_btn.clicked.connect(self._on_export)
        top.addWidget(self.export_btn)

        root.addLayout(top)

        # ── Status / progress ────────────────────────────────
        self.status_label = QLabel("Enter a ticker and press Generate.")
        self.status_label.setStyleSheet("color: #888; font-style: italic;")
        root.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        # ── Playback controls ─────────────────────────────────
        pb = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._on_play)
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self._on_pause)
        pb.addWidget(self.play_btn)
        pb.addWidget(self.pause_btn)
        pb.addWidget(self.stop_btn)

        QLabel("Speed:", self)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(25, 400)
        self.speed_slider.setValue(100)
        self.speed_slider.setFixedWidth(140)
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_label.setText(f"{v/100:.2f}x")
        )
        self.speed_label = QLabel("1.00x")
        pb.addWidget(self.speed_slider)
        pb.addWidget(self.speed_label)

        root.addLayout(pb)

        # ── Info log ──────────────────────────────────────────
        self.info_log = QTextEdit()
        self.info_log.setReadOnly(True)
        self.info_log.setMaximumHeight(150)
        self.info_log.setFont(QFont("Consolas", 9))
        self.info_log.setPlaceholderText(
            "Event log will appear here — detected events, generation status, etc."
        )
        root.addWidget(self.info_log)

        # ── Separator ─────────────────────────────────────────
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #555;")
        root.addWidget(line)

        # ── Advanced Parameters (collapsible) ─────────────────
        self.adv_toggle = QToolButton()
        self.adv_toggle.setText("▸ Advanced Parameters")
        self.adv_toggle.setCheckable(True)
        self.adv_toggle.setChecked(False)
        self.adv_toggle.clicked.connect(self._toggle_advanced)
        root.addWidget(self.adv_toggle)

        self.adv_frame = QWidget()
        self.adv_frame.setVisible(False)
        adv_layout = QGridLayout(self.adv_frame)

        row = 0
        adv_layout.addWidget(QLabel("Scale:"), row, 0)
        self.scale_combo = QComboBox()
        for s in config.SCALE_SEMITONES:
            self.scale_combo.addItem(s)
        self.scale_combo.setCurrentText("Dorian")
        adv_layout.addWidget(self.scale_combo, row, 1)

        row += 1
        adv_layout.addWidget(QLabel("Tonic (Hz):"), row, 0)
        self.tonic_spin = QDoubleSpinBox()
        self.tonic_spin.setRange(55, 880)
        self.tonic_spin.setValue(293)
        self.tonic_spin.setSuffix(" Hz")
        adv_layout.addWidget(self.tonic_spin, row, 1)

        row += 1
        adv_layout.addWidget(QLabel("Pitch Range:"), row, 0)
        self.freq_min_spin = QDoubleSpinBox()
        self.freq_min_spin.setRange(80, 2000)
        self.freq_min_spin.setValue(220)
        self.freq_min_spin.setSuffix(" Hz")
        adv_layout.addWidget(QLabel("Min"), row, 1)
        self.freq_min_spin.setFixedWidth(80)
        adv_layout.addWidget(self.freq_min_spin, row, 2)

        self.freq_max_spin = QDoubleSpinBox()
        self.freq_max_spin.setRange(400, 5000)
        self.freq_max_spin.setValue(1760)
        self.freq_max_spin.setSuffix(" Hz")
        adv_layout.addWidget(QLabel("Max"), row, 3)
        self.freq_max_spin.setFixedWidth(80)
        adv_layout.addWidget(self.freq_max_spin, row, 4)

        row += 1
        adv_layout.addWidget(QLabel("Note Duration:"), row, 0)
        self.dur_spin = QDoubleSpinBox()
        self.dur_spin.setRange(100, 1500)
        self.dur_spin.setValue(400)
        self.dur_spin.setSuffix(" ms")
        self.dur_spin.setSingleStep(50)
        adv_layout.addWidget(self.dur_spin, row, 1)

        row += 1
        adv_layout.addWidget(QLabel("Waveform:"), row, 0)
        self.waveform_combo = QComboBox()
        for w in ['triangle', 'sine', 'sawtooth', 'square']:
            self.waveform_combo.addItem(w)
        self.waveform_combo.setCurrentText("triangle")
        adv_layout.addWidget(self.waveform_combo, row, 1)

        row += 1
        adv_layout.addWidget(QLabel("Effect:"), row, 0)
        self.effect_combo = QComboBox()
        for e in ['None', 'Chorus', 'Reverb']:
            self.effect_combo.addItem(e)
        self.effect_combo.setCurrentText("None")
        adv_layout.addWidget(self.effect_combo, row, 1)

        row += 1
        adv_layout.addWidget(QLabel("Smooth (portamento):"), row, 0)
        self.smooth_cb = QCheckBox()
        self.smooth_cb.setChecked(True)
        adv_layout.addWidget(self.smooth_cb, row, 1)

        row += 1
        adv_layout.addWidget(QLabel(f"Vibrato (cents ±):"), row, 0)
        self.vibrato_spin = QDoubleSpinBox()
        self.vibrato_spin.setRange(0, 200)
        self.vibrato_spin.setValue(100)
        self.vibrato_spin.setSuffix(" cents")
        adv_layout.addWidget(self.vibrato_spin, row, 1)

        row += 1
        adv_layout.addWidget(QLabel("Sustain Range:"), row, 0)
        self.sustain_min_spin = QDoubleSpinBox()
        self.sustain_min_spin.setRange(0.1, 1.0)
        self.sustain_min_spin.setValue(0.3)
        self.sustain_min_spin.setSingleStep(0.05)
        adv_layout.addWidget(QLabel("Min"), row, 1)
        adv_layout.addWidget(self.sustain_min_spin, row, 2)

        self.sustain_max_spin = QDoubleSpinBox()
        self.sustain_max_spin.setRange(1.0, 4.0)
        self.sustain_max_spin.setValue(2.0)
        self.sustain_max_spin.setSingleStep(0.1)
        adv_layout.addWidget(QLabel("Max"), row, 3)
        adv_layout.addWidget(self.sustain_max_spin, row, 4)

        row += 1
        adv_layout.addWidget(QLabel("Silence Threshold:"), row, 0)
        self.silence_body_spin = QDoubleSpinBox()
        self.silence_body_spin.setRange(0.0, 0.1)
        self.silence_body_spin.setValue(0.005)
        self.silence_body_spin.setSingleStep(0.001)
        self.silence_body_spin.setDecimals(4)
        self.silence_body_spin.setSuffix(" (body)")
        adv_layout.addWidget(self.silence_body_spin, row, 1)

        self.silence_range_spin = QDoubleSpinBox()
        self.silence_range_spin.setRange(0.0, 0.1)
        self.silence_range_spin.setValue(0.008)
        self.silence_range_spin.setSingleStep(0.001)
        self.silence_range_spin.setDecimals(4)
        self.silence_range_spin.setSuffix(" (wick)")
        adv_layout.addWidget(self.silence_range_spin, row, 2)

        row += 1
        adv_layout.addWidget(QLabel("Reference:"), row, 0)
        self.ref_combo = QComboBox()
        for m in ['ema20', 'sma20', 'vwap', 'session_open']:
            self.ref_combo.addItem(m.replace('_', ' ').title())
        self.ref_combo.setCurrentText("Ema20")
        adv_layout.addWidget(self.ref_combo, row, 1)

        row += 1
        self.bass_cb = QCheckBox("Bass voice (trend)")
        self.bass_cb.setChecked(True)
        adv_layout.addWidget(self.bass_cb, row, 0)

        self.hb_cb = QCheckBox("Heartbeat during silence")
        self.hb_cb.setChecked(True)
        adv_layout.addWidget(self.hb_cb, row, 1)

        row += 1
        self.alert_bo_cb = QCheckBox("Alert: Breakout")
        self.alert_bo_cb.setChecked(True)
        adv_layout.addWidget(self.alert_bo_cb, row, 0)

        self.alert_dv_cb = QCheckBox("Alert: Divergence")
        self.alert_dv_cb.setChecked(True)
        adv_layout.addWidget(self.alert_dv_cb, row, 1)

        self.alert_vs_cb = QCheckBox("Alert: Vol Spike")
        self.alert_vs_cb.setChecked(True)
        adv_layout.addWidget(self.alert_vs_cb, row, 2)

        self.alert_rv_cb = QCheckBox("Alert: Reversal")
        self.alert_rv_cb.setChecked(False)
        adv_layout.addWidget(self.alert_rv_cb, row, 3)

        root.addWidget(self.adv_frame)

        # ── Legend / shortcuts ────────────────────────────────
        legend = QLabel(
            "<b>Keyboard shortcuts:</b> Space = play/stop | "
            "<b>Ticker format:</b> BTC-USD AAPL ^GSPC ETH-USD"
        )
        legend.setStyleSheet("color: #666; font-size: 10px;")
        root.addWidget(legend)

        # ── Keyboard shortcuts ───────────────────────────────
        from PyQt5.QtCore import Qt
        # (Handled via focus — simple approach)

        # ── Apply preset on load ─────────────────────────────
        self._apply_preset(config.DEFAULT_PRESET)

    # ── Theme ──────────────────────────────────────────────────────
    def _dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #1e1e2e; }
            QWidget     { background: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', sans-serif; }
            QPushButton {
                background: #45475a; border: 1px solid #585b70;
                color: #cdd6f4; padding: 6px 14px; border-radius: 4px;
            }
            QPushButton:hover  { background: #585b70; }
            QPushButton:pressed { background: #6c7086; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #313244; border: 1px solid #45475a;
                color: #cdd6f4; padding: 4px; border-radius: 3px;
            }
            QGroupBox  { border: 1px solid #45475a; border-radius: 4px;
                         margin-top: 10px; padding-top: 14px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QTextEdit  { background: #181825; border: 1px solid #313244;
                         color: #a6adc8; border-radius: 4px; padding: 4px; }
            QProgressBar { border: 1px solid #45475a; border-radius: 3px;
                           text-align: center; background: #181825; }
            QProgressBar::chunk { background: #a6e3a1; }
            QSlider::groove:horizontal { background: #313244; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #89b4fa; width: 14px;
                margin:-4px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #a6e3a1; border-radius: 3px; }
            QCheckBox   { spacing: 6px; }
        """)

    # ── Preset system ─────────────────────────────────────────────
    def _apply_preset(self, name: str):
        cfg = config.PRESETS.get(name, config.PRESETS[config.DEFAULT_PRESET])
        self.scale_combo.setCurrentText(cfg['scale'])
        self.tonic_spin.setValue(cfg['tonic_freq'])
        self.freq_min_spin.setValue(cfg['freq_min'])
        self.freq_max_spin.setValue(cfg['freq_max'])
        self.dur_spin.setValue(cfg['duration_ms'])
        self.waveform_combo.setCurrentText(cfg['waveform'])
        self.effect_combo.setCurrentText(cfg['effect'])
        self.smooth_cb.setChecked(cfg['smooth'])
        self.vibrato_spin.setValue(cfg['vibrato_max_cents'])
        self.sustain_min_spin.setValue(cfg['sustain_min'])
        self.sustain_max_spin.setValue(cfg['sustain_max'])
        self.silence_body_spin.setValue(cfg['silence_threshold_body'])
        self.silence_range_spin.setValue(cfg['silence_threshold_range'])
        idx = self.ref_combo.findText(
            cfg['reference_mode'].replace('_', ' ').title(),
        )
        if idx >= 0:
            self.ref_combo.setCurrentIndex(idx)
        self.bass_cb.setChecked(cfg['bass_enabled'])
        self.hb_cb.setChecked(cfg['heartbeat_enabled'])
        self.alert_bo_cb.setChecked(cfg.get('alert_on_breakout', False))
        self.alert_dv_cb.setChecked(cfg.get('alert_on_divergence', False))
        self.alert_vs_cb.setChecked(cfg.get('alert_on_volume_spike', False))
        self.alert_rv_cb.setChecked(cfg.get('alert_on_reversal', False))

    def _current_preset(self) -> dict:
        ref_idx = self.ref_combo.currentIndex()
        ref_key = self.ref_combo.itemData(ref_idx, Qt.DisplayRole)
        ref_map = {
            'Ema20': 'ema20', 'Sma20': 'sma', 'Vwap': 'vwap',
            'Session Open': 'session_open',
        }
        return {
            'scale':                self.scale_combo.currentText(),
            'tonic_freq':           self.tonic_spin.value(),
            'freq_min':             self.freq_min_spin.value(),
            'freq_max':             self.freq_max_spin.value(),
            'duration_ms':          self.dur_spin.value(),
            'smooth':               self.smooth_cb.isChecked(),
            'waveform':             self.waveform_combo.currentText(),
            'effect':               self.effect_combo.currentText(),
            'vibrato_max_cents':    self.vibrato_spin.value(),
            'sustain_min':          self.sustain_min_spin.value(),
            'sustain_max':          self.sustain_max_spin.value(),
            'silence_threshold_body': self.silence_body_spin.value(),
            'silence_threshold_range': self.silence_range_spin.value(),
            'reference_mode':       ref_map.get(ref_key, 'ema20'),
            'bass_enabled':         self.bass_cb.isChecked(),
            'heartbeat_enabled':    self.hb_cb.isChecked(),
            'alert_on_breakout':    self.alert_bo_cb.isChecked(),
            'alert_on_divergence':  self.alert_dv_cb.isChecked(),
            'alert_on_volume_spike': self.alert_vs_cb.isChecked(),
            'alert_on_reversal':    self.alert_rv_cb.isChecked(),
            'volume_spike_threshold': 2.0,
            'bass_freq_min': 55.0,
            'bass_freq_max': 110.0,
            'bass_note_every': 3,
            'reference_window': 20,
        }

    def _toggle_advanced(self):
        visible = self.adv_toggle.isChecked()
        self.adv_frame.setVisible(visible)
        self.adv_toggle.setText(
            "▾ Advanced Parameters" if visible else "▸ Advanced Parameters"
        )

    # ── Data fetch + generate ─────────────────────────────────────
    def _on_generate(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            self.status_label.setText("⚠ Enter a ticker symbol.")
            return

        period_map = {
            '7D': '7d', '1M': '1mo', '3M': '3mo',
            '6M': '6mo', '1Y': '1y', '5Y': '5y',
        }
        period = period_map.get(self.period_combo.currentText(), '3mo')
        tf_map = {
            '7D': '1h', '1M': '1d', '3M': '1d',
            '6M': '1wk', '1Y': '1wk', '5Y': '1mo',
        }
        interval = tf_map.get(self.period_combo.currentText(), '1d')

        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Fetching {ticker} ({period}, {interval})…")
        self.info_log.clear()

        cfg = self._current_preset()

        self._worker = GenerationWorker(ticker, period, interval, cfg)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.status.connect(self._on_status)
        self._worker.start()

    def _on_progress(self, current, total):
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(current)

    def _on_finished(self, audio, events, raw_data, tag):
        self.current_audio = audio
        self.events = events
        self._raw_data = raw_data
        self._current_ticker = self.ticker_input.text().strip().upper()
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)

        n = len(events)
        msg = f"✓ Generated {len(audio) / 44100:.1f}s of audio "
        msg += f"({self._current_ticker}). "
        if n:
            msg += f"{n} event{'s' if n != 1 else ''} detected."
            self.info_log.append(f"== {n} events detected ==")
            for ev in events:
                icon = {'breakout': '🚀', 'divergence': '⚠',
                        'volume_spike': '📊', 'reversal': '🔄'}.get(ev.kind, '•')
                self.info_log.append(f"  {icon} Candle #{ev.candle_index}: {ev.description}")
        else:
            msg += "No significant events detected."
        self.status_label.setText(msg)
        self.info_log.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] {msg}")

        # Auto-play
        self._on_play()

    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.status_label.setText(f"✗ Error: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def _on_status(self, text):
        self.status_label.setText(text)

    # ── Playback ──────────────────────────────────────────────────
    def _on_play(self):
        if len(self.current_audio) == 0:
            self.status_label.setText("Nothing to play. Generate audio first.")
            return

        if self.is_playing:
            self._stop_stream()

        self.is_playing = True
        self.play_btn.setText("▶ Playing…")

        speed = self.speed_slider.value() / 100.0
        if speed != 1.0:
            # Simple resample for speed adjustment
            new_len = max(1, int(len(self.current_audio) / speed))
            indices = np.linspace(0, len(self.current_audio) - 1, new_len)
            audio_resampled = np.interp(
                indices,
                np.arange(len(self.current_audio)),
                self.current_audio.astype(np.float64),
            ).astype(np.float32)
        else:
            audio_resampled = self.current_audio

        self._play_start_time = time.time()
        self._playback_thread = _PlaybackThread(audio_resampled)
        self._playback_thread.finished.connect(self._on_playback_done)
        self._playback_thread.start()

    def _on_pause(self):
        if self.is_playing:
            self._playback_thread.pause()

    def _on_stop(self):
        self._stop_stream()

    def _stop_stream(self):
        if hasattr(self, '_playback_thread') and self._playback_thread.isRunning():
            self._playback_thread.stop()
            self._playback_thread.wait(2000)
        self.is_playing = False
        self.play_btn.setText("▶ Play")

    def _on_playback_done(self):
        self.is_playing = False
        self.play_btn.setText("▶ Play")
        self.status_label.setText("Playback complete.")

    # ── Export ────────────────────────────────────────────────────
    def _on_export(self):
        if len(self.current_audio) == 0:
            QMessageBox.warning(self, "Nothing to export", "Generate audio first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save WAV file",
            f"sonifier_{self.ticker_input.text().strip()}.wav",
            "WAV files (*.wav)",
        )
        if path:
            from scipy.io import wavfile
            audio_int = np.int16(self.current_audio * 32767)
            wavfile.write(path, 44100, audio_int)
            self.status_label.setText(f"✓ Exported to {os.path.basename(path)}")
            self.info_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Exported: {path}")

    # ── Keyboard shortcuts ────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.is_playing:
                self._on_stop()
            else:
                self._on_play()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self._on_generate()
        super().keyPressEvent(event)


# ═══════════════════════════════════════════════════════════════════
#  BACKGROUND PLAYBACK THREAD
# ═══════════════════════════════════════════════════════════════════

class _PlaybackThread(QThread):
    finished = pyqtSignal()

    def __init__(self, audio: np.ndarray):
        super().__init__()
        self.audio = audio
        self._paused = False
        self._stop = False
        self._cond = threading.Condition()

    def run(self):
        try:
            with sd.OutputStream(
                samplerate=44100,
                channels=1,
                dtype='float32',
                blocksize=1024,
                callback=self._audio_callback,
            ):
                while not self._stop:
                    with self._cond:
                        while self._paused and not self._stop:
                            self._cond.wait(0.1)
                        if self._stop:
                            break

                    time.sleep(0.01)

        except Exception:
            pass
        finally:
            self.finished.emit()

    def _audio_callback(self, outdata, frames, time_info, status):
        if self._stop:
            outdata.fill(0)
            raise sd.CallbackStop

        chunk = self.audio[:frames]
        outdata[:len(chunk)] = chunk.reshape(-1, 1)
        outdata[len(chunk):] = 0
        self.audio = self.audio[frames:]

        if len(self.audio) == 0:
            raise sd.CallbackStop

    def pause(self):
        self._paused = not self._paused

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = SonifierMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
