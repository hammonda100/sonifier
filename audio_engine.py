"""
Sonifier v3.0 — Core Audio Synthesis Engine

Converts OHLCV market data into a polyphonic auditory stream.
Designed for maximum information preservation with zero interpretation.
The ear is the processor; this is the cable.

Architecture:
    Voice 1 (Melody, C4-C6): Full composite timbre per candle
    Voice 2 (Bass, C2-C3):   SMA/EMA trend drone, sparse updates
    No harmony voice — avoids register collision during volatile periods.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict


# ── Utility Functions ──────────────────────────────────────────────

def midi_to_freq(midi_note: float) -> float:
    """Convert MIDI note number to frequency."""
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10 ** (db / 20)


# ── Waveform Generators (anti-aliased) ─────────────────────────────

def _waveform_sine(phase: np.ndarray) -> np.ndarray:
    return np.sin(phase)


def _waveform_triangle(phase: np.ndarray) -> np.ndarray:
    """Band-limited triangle (arcsin method)."""
    return (2.0 / np.pi) * np.arcsin(np.sin(phase))


def _waveform_sawtooth(phase: np.ndarray, harmonics: int = 20) -> np.ndarray:
    """Band-limited sawtooth via additive synthesis."""
    result = np.zeros_like(phase)
    for h in range(1, harmonics + 1):
        amplitude = 1.0 / h
        result += amplitude * np.sin(phase * h)
    return -(2.0 / np.pi) * result


def _waveform_square(phase: np.ndarray, harmonics: int = 16) -> np.ndarray:
    """Band-limited square wave."""
    result = np.zeros_like(phase)
    for h in range(1, harmonics + 1, 2):
        amplitude = 1.0 / h
        result += amplitude * np.sin(phase * h)
    return (4.0 / np.pi) * result


WAVEFORM_MAP = {
    'sine':     _waveform_sine,
    'triangle': _waveform_triangle,
    'sawtooth': _waveform_sawtooth,
    'square':   _waveform_square,
}


# ── ADSR Envelope ──────────────────────────────────────────────────

def _adsr_envelope(
    num_samples: int,
    sample_rate: int,
    attack_time: float,
    decay_time: float,
    sustain_level: float,
    release_time: float,
) -> np.ndarray:
    """Generate ADSR envelope. All times in seconds."""
    t = np.arange(num_samples) / sample_rate
    total = num_samples / sample_rate
    envelope = np.zeros(num_samples, dtype=np.float64)

    atk_end = min(attack_time, total)
    dec_end = min(atk_end + decay_time, total)
    rel_start = max(total - release_time, dec_end)

    # Attack: 0 → 1
    mask = t < atk_end
    if np.any(mask):
        d = max(atk_end, 1e-12)
        envelope[mask] = 0.5 * (1 - np.cos(np.pi * t[mask] / d))

    # Decay: 1 → sustain_level
    mask = (t >= atk_end) & (t < dec_end)
    if np.any(mask):
        d = max(dec_end - atk_end, 1e-12)
        envelope[mask] = 1.0 + (sustain_level - 1.0) * (t[mask] - atk_end) / d

    # Sustain
    mask = (t >= dec_end) & (t < rel_start)
    if np.any(mask):
        envelope[mask] = sustain_level

    # Release: sustain_level → 0
    mask = t >= rel_start
    if np.any(mask):
        d = max(total - rel_start, 1e-12)
        envelope[mask] = sustain_level * 0.5 * (
            1 + np.cos(np.pi * (t[mask] - rel_start) / d)
        )

    # Final fade to zero (anti-click)
    fade_len = min(200, num_samples)
    envelope[-fade_len:] *= np.linspace(1, 0, fade_len)

    return envelope


# ── Signal Processing ──────────────────────────────────────────────

def _normalize(audio: np.ndarray, headroom_db: float = 3.0) -> np.ndarray:
    """Normalize to peak with headroom."""
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-12:
        return audio
    target = db_to_linear(-headroom_db)
    return audio * (target / peak)


def _apply_reverb(
    audio: np.ndarray,
    wet: float = 0.2,
    delay_ms: float = 30.0,
    feedback: float = 0.35,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Simple feedback delay reverb."""
    delay_samples = int(delay_ms * sample_rate / 1000.0)
    if delay_samples < 1:
        return audio

    output = audio.copy()
    for i in range(delay_samples, len(audio)):
        output[i] += output[i - delay_samples] * feedback * wet

    return _normalize(output)


def _apply_chorus(
    audio: np.ndarray,
    depth_cents: float = 8.0,
    rate_hz: float = 1.2,
    mix: float = 0.3,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Simple chorus effect (pitch modulation + mixing)."""
    t = np.arange(len(audio), dtype=np.float64) / sample_rate
    mod = depth_cents * np.sin(2 * np.pi * rate_hz * t)
    mod_ratio = 2 ** (mod / 1200.0)

    # Fractional delay via linear interpolation
    base_idx = np.arange(len(audio))
    delay_idx = np.cumsum(mod_ratio)
    idx_floor = np.floor(delay_idx).astype(int)
    frac = delay_idx - idx_floor

    # Clamp indices
    idx_floor = np.clip(idx_floor, 0, len(audio) - 2)
    delayed = audio[idx_floor] * (1 - frac) + audio[idx_floor + 1] * frac

    return audio * (1 - mix) + delayed * mix


def _apply_filter(
    audio: np.ndarray,
    cutoff_hz: float,
    sample_rate: int = 44100,
    filter_type: str = 'lowpass',
) -> np.ndarray:
    """Simple first-order IIR filter."""
    dt = 1.0 / sample_rate
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    alpha = dt / (rc + dt)

    filtered = np.zeros_like(audio)
    if filter_type == 'lowpass':
        filtered[0] = audio[0]
        for i in range(1, len(audio)):
            filtered[i] = filtered[i - 1] + alpha * (audio[i] - filtered[i - 1])
    elif filter_type == 'highpass':
        filtered[0] = audio[0]
        for i in range(1, len(audio)):
            filtered[i] = alpha * (filtered[i - 1] + audio[i] - audio[i - 1])

    return filtered


# ── Event System ───────────────────────────────────────────────────

class SonificationEvent:
    """A significant event detected during sonification."""

    BREAKOUT = 'breakout'
    DIVERGENCE = 'divergence'
    VOLUME_SPIKE = 'volume_spike'
    REVERSAL = 'reversal'
    SUPPORT_TEST = 'support_test'
    RESISTANCE_TEST = 'resistance_test'

    def __init__(
        self,
        kind: str,
        candle_index: int,
        timestamp: str,
        description: str,
        magnitude: float,
    ):
        self.kind = kind
        self.candle_index = candle_index
        self.timestamp = timestamp
        self.description = description
        self.magnitude = magnitude

    def __repr__(self):
        return f"Event({self.kind} at candle {self.candle_index}: {self.description})"


# ── Main Generator ─────────────────────────────────────────────────

class AudioGenerator:
    """
    Generates audio from OHLCV market data.

    The design preserves maximum information density. Each candle
    becomes a single complex tone along with contributions from
    contextual voices (bass). No data is summarized or filtered.
    """

    def __init__(
        self,
        scale: str = 'Dorian',
        tonic_freq: float = 293.0,
        freq_min: float = 220.0,
        freq_max: float = 1760.0,
        duration_per_note: float = 0.4,
        sample_rate: int = 44100,
        smooth: bool = True,
        waveform: str = 'triangle',
        effect: str = 'None',
        vibrato_max_cents: float = 100.0,
        sustain_min: float = 0.3,
        sustain_max: float = 2.0,
        silence_threshold_body: float = 0.005,
        silence_threshold_range: float = 0.008,
        heartbeat_enabled: bool = True,
        bass_enabled: bool = True,
        bass_freq_min: float = 55.0,
        bass_freq_max: float = 110.0,
        bass_note_every: int = 3,
        reference_mode: str = 'ema20',
        reference_window: int = 20,
        alert_breakout: bool = True,
        alert_divergence: bool = True,
        alert_volume_spike: bool = True,
        volume_spike_threshold: float = 2.0,
    ):
        self.scale = scale
        self.tonic_freq = tonic_freq
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.duration_per_note = duration_per_note
        self.sample_rate = sample_rate
        self.smooth = smooth
        self.waveform_name = waveform
        self.waveform = WAVEFORM_MAP.get(waveform, _waveform_triangle)
        self.effect = effect
        self.vibrato_max_cents = vibrato_max_cents
        self.sustain_min = sustain_min
        self.sustain_max = sustain_max
        self.silence_threshold_body = silence_threshold_body
        self.silence_threshold_range = silence_threshold_range
        self.heartbeat_enabled = heartbeat_enabled
        self.bass_enabled = bass_enabled
        self.bass_freq_min = bass_freq_min
        self.bass_freq_max = bass_freq_max
        self.bass_note_every = max(1, bass_note_every)
        self.reference_mode = reference_mode
        self.reference_window = reference_window
        self.alert_breakout = alert_breakout
        self.alert_divergence = alert_divergence
        self.alert_volume_spike = alert_volume_spike
        self.volume_spike_threshold = volume_spike_threshold

        # Populated by generate()
        self.events: List[SonificationEvent] = []

    # ── Scale Quantization ─────────────────────────────────────────

    def _get_scale_frequencies(self) -> np.ndarray:
        """All frequencies in the selected scale within audible range."""
        semitones = SCALE_SEMITONES.get(self.scale, SCALE_SEMITONES['Dorian'])
        freqs = []
        for octave in range(0, 10):
            for st in semitones:
                f = self.tonic_freq * (2 ** ((octave * 12 + st) / 12))
                if self.freq_min * 0.5 <= f <= self.freq_max * 2:
                    freqs.append(f)
        return np.array(sorted(freqs), dtype=np.float64)

    def _quantize_to_scale(self, freq: float) -> float:
        """Snap frequency to nearest scale degree."""
        scale_freqs = self._get_scale_frequencies()
        if len(scale_freqs) == 0:
            return freq
        idx = int(np.argmin(np.abs(np.log2(scale_freqs / freq))))
        idx = max(0, min(idx, len(scale_freqs) - 1))
        return float(scale_freqs[idx])

    # ── Reference Computation ──────────────────────────────────────

    def _compute_reference(
        self, prices: np.ndarray, volumes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute the dynamic reference frame (tonic per candle)."""
        n = len(prices)
        refs = np.zeros(n, dtype=np.float64)

        if self.reference_mode == 'session_open':
            refs[:] = prices[0]
        elif self.reference_mode == 'ema20':
            series = pd.Series(prices)
            ema = series.ewm(span=self.reference_window, adjust=False).mean()
            refs[:] = ema.values
            # Fill initial NaN with first valid EMA or session open
            first_valid = ema.first_valid_index() or 0
            refs[:first_valid] = prices[0]
        elif self.reference_mode == 'sma20':
            series = pd.Series(prices)
            sma = series.rolling(window=self.reference_window, min_periods=1).mean()
            refs[:] = sma.values
        elif self.reference_mode == 'vwap':
            if volumes is not None:
                cvol = np.cumsum(prices * volumes)
                cvolsum = np.cumsum(volumes)
                vwap = cvol / np.maximum(cvolsum, 1e-12)
                refs[:] = vwap
            else:
                refs[:] = prices[0]
        else:
            refs[:] = prices[0]

        return refs

    # ── Pitch Computation ──────────────────────────────────────────

    def _compute_pitches(
        self,
        prices: np.ndarray,
        refs: np.ndarray,
    ) -> np.ndarray:
        """Log-ratio pitch mapping with scale quantization."""
        n = len(prices)
        frequencies = np.zeros(n, dtype=np.float64)

        for i in range(n):
            ref = refs[i] if refs[i] > 1e-12 else prices[0]
            price = prices[i]

            if price < 1e-12 or ref < 1e-12:
                frequencies[i] = self.freq_min
                continue

            # Log ratio from reference
            log_ratio = np.log(price / ref)
            # Map to frequency: semitones from reference
            semitones = log_ratio * (12.0 / np.log(2))
            raw_freq = ref * (2 ** (semitones / 12))

            # Clamp to range
            raw_freq = max(self.freq_min, min(self.freq_max, raw_freq))

            # Quantize
            frequencies[i] = self._quantize_to_scale(raw_freq)

        return frequencies

    # ── Bass Voice ─────────────────────────────────────────────────

    def _compute_bass(
        self, refs: np.ndarray
    ) -> List[Tuple[int, float]]:
        """
        Compute bass note events (sparse, only when pitch changes
        significantly).

        Returns list of (candle_index, frequency).
        """
        if not self.bass_enabled:
            return []

        bass_notes = []
        last_bass_freq = None

        for i in range(len(refs)):
            # Bass maps to same reference but 2 octaves lower
            # Only update when reference shifts enough
            bass_freq = refs[i] * (2 ** (-2))  # 2 octaves below
            bass_freq = max(self.bass_freq_min, min(self.bass_freq_max, bass_freq))
            bass_freq = self._quantize_to_scale(bass_freq)

            # Only emit a new bass note every bass_note_every candles
            # or when pitch changes by a semitone
            is_interval = (i % self.bass_note_every == 0)
            is_shift = (
                last_bass_freq is None or
                abs(bass_freq - last_bass_freq) > last_bass_freq * 0.05
            )

            if is_interval or is_shift:
                bass_notes.append((i, bass_freq))
                last_bass_freq = bass_freq

        return bass_notes

    # ── Alert Detection ────────────────────────────────────────────

    def _detect_alerts(
        self,
        prices: np.ndarray,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        body_sizes: np.ndarray,
        wick_ratios: np.ndarray,
        refs: np.ndarray,
    ) -> List[SonificationEvent]:
        """Detect significant market events for alert sounds."""
        events = []
        n = len(prices)
        if n < 3:
            return events

        # Volume spike detection
        if self.alert_volume_spike and n > 10:
            mean_vol = np.mean(volumes)
            std_vol = np.std(volumes)
            if std_vol > 1e-12:
                for i in range(1, n):
                    z = (volumes[i] - mean_vol) / std_vol
                    if z > self.volume_spike_threshold:
                        events.append(SonificationEvent(
                            kind=SonificationEvent.VOLUME_SPIKE,
                            candle_index=i,
                            timestamp=f"candle_{i}",
                            description=f"Volume spike: {z:.1f}σ above mean",
                            magnitude=float(z),
                        ))

        # Breakout detection: price closes above/below prior N-period high/low
        if self.alert_breakout and n > 20:
            for i in range(20, n):
                prior_high = np.max(highs[max(0, i - 20):i])
                prior_low = np.min(lows[max(0, i - 20):i])
                if closes[i] > prior_high and body_sizes[i] > 0.01:
                    events.append(SonificationEvent(
                        kind=SonificationEvent.BREAKOUT,
                        candle_index=i,
                        timestamp=f"candle_{i}",
                        description="Breakout above prior 20-period high",
                        magnitude=float(closes[i] - prior_high) / prior_high,
                    ))
                elif closes[i] < prior_low and body_sizes[i] > 0.01:
                    events.append(SonificationEvent(
                        kind=SonificationEvent.BREAKOUT,
                        candle_index=i,
                        timestamp=f"candle_{i}",
                        description="Breakdown below prior 20-period low",
                        magnitude=float(prior_low - closes[i]) / prior_low,
                    ))

        # Divergence: price makes new high/low but direction candle weakens
        if self.alert_divergence and n > 5:
            for i in range(5, n):
                # Price higher, but body shrinking (weakness)
                if (prices[i] > prices[i - 3] and
                        body_sizes[i] < body_sizes[i - 3] * 0.5 and
                        body_sizes[i - 3] > 0.02):
                    events.append(SonificationEvent(
                        kind=SonificationEvent.DIVERGENCE,
                        candle_index=i,
                        timestamp=f"candle_{i}",
                        description="Price divergence: higher price, weaker body",
                        magnitude=float(prices[i] - prices[i - 3]) / prices[i - 3],
                    ))

        # Reversal detection: 3 consecutive direction changes
        if self.alert_reversal:
            for i in range(3, n):
                d1 = 1 if closes[i] >= opens[i] else -1
                d2 = 1 if closes[i - 1] >= opens[i - 1] else -1
                d3 = 1 if closes[i - 2] >= opens[i - 2] else -1
                if d1 != d2 and d2 != d3:
                    events.append(SonificationEvent(
                        kind=SonificationEvent.REVERSAL,
                        candle_index=i,
                        timestamp=f"candle_{i}",
                        description="Three-candle reversal pattern",
                        magnitude=1.0,
                    ))

        self.events = events
        return events

    # ── Note Synthesis ─────────────────────────────────────────────

    def _generate_note(
        self,
        freq: float,
        velocity: float,
        note_samples: int,
        body_direction: float = 0.0,
        wick_ratio: float = 0.0,
        attack_sharpness: float = 0.5,
        harmonic_bloom: int = 0,
    ) -> np.ndarray:
        """
        Generate a single note with full timbral complexity.

        Args:
            freq: Base frequency in Hz.
            velocity: Amplitude [0, 1].
            note_samples: Number of samples.
            body_direction: +1 (bull), -1 (bear), 0 (doji).
            wick_ratio: Normalized wick range [0, ~1].
            attack_sharpness: 0 = slow attack, 1 = instant attack.
            harmonic_bloom: Number of additional partials (0-6).
        """
        actual_samples = max(note_samples, 100)
        sr = self.sample_rate

        # ── Vibrato ────────────────────────────────────────────
        v_depth = min(wick_ratio * self.vibrato_max_cents, self.vibrato_max_cents)
        v_rate = 4.0 + 4.0 * min(wick_ratio, 1.0)

        t = np.arange(actual_samples, dtype=np.float64) / sr
        base_phase = 2 * np.pi * freq * t

        # Apply vibrato via phase modulation
        vibrato_mod = v_depth * np.sin(2 * np.pi * v_rate * t)
        vibrato_ratio = 2 ** (vibrato_mod / 1200.0)
        phase = base_phase * vibrato_ratio

        # Pitch sweep: subtle lean in candle direction
        pitch_sweep = np.linspace(0, body_direction * 0.01, actual_samples)
        sweep_addition = np.cumsum(
            2 ** (pitch_sweep / 12)
        ) * freq * 2 * np.pi / sr
        phase = 0.7 * phase + 0.3 * sweep_addition

        # ── Waveform ────────────────────────────────────────────
        wave = self.waveform(phase) * velocity

        # ── Harmonic Bloom ──────────────────────────────────────
        for p in range(2, 2 + min(harmonic_bloom, 7)):
            amp = 0.25 / (p - 0.3)
            wave += np.sin(phase * p) * amp * velocity

        # ── Envelope ────────────────────────────────────────────
        atk_time = 0.005 + 0.045 * (1 - attack_sharpness)
        dec_time = 0.1 + 0.2 * velocity
        sus_level = 0.25 + 0.6 * velocity
        rel_time = 0.1 + 0.2 * velocity

        envelope = _adsr_envelope(
            actual_samples, sr,
            atk_time, dec_time, sus_level, rel_time,
        )

        return (wave * envelope).astype(np.float32)

    def _generate_portamento_note(
        self,
        target_freq: float,
        prev_freq: float,
        velocity: float,
        portamento_sec: float,
        note_samples: int,
        body_direction: float = 0.0,
        wick_ratio: float = 0.0,
        attack_sharpness: float = 0.5,
        harmonic_bloom: int = 0,
    ) -> np.ndarray:
        """Generate note with smooth pitch slide from previous note."""
        actual_samples = max(note_samples, 100)
        sr = self.sample_rate

        port_samples = min(
            int(portamento_sec * sr), actual_samples // 2
        )
        sustain_samples = actual_samples - port_samples

        # Portamento phase: frequency sweep from prev to target
        port_freqs = np.linspace(prev_freq, target_freq, port_samples + 1)
        port_phase = np.zeros(port_samples)
        for j in range(port_samples):
            port_phase[j] = 2 * np.pi * port_freqs[j] * (1.0 / sr)
        port_phase = np.cumsum(port_phase)

        # Vibrato on portamento too
        v_depth = min(wick_ratio * self.vibrato_max_cents * 0.5, self.vibrato_max_cents)
        v_rate = 4.0 + 4.0 * min(wick_ratio, 1.0)
        t_port = np.arange(port_samples) / sr
        vibrato = v_depth * np.sin(2 * np.pi * v_rate * t_port)
        vibrato_ratio = 2 ** (vibrato / 1200.0)
        port_phase = port_phase * vibrato_ratio

        wave_port = self.waveform(port_phase) * velocity

        # Portamento envelope (fade out tail)
        port_env = np.linspace(1, 0.3, port_samples) ** 0.5
        wave_port *= port_env

        # Sustain phase
        if sustain_samples > 20:
            sustain_note = self._generate_note(
                target_freq, velocity, sustain_samples,
                body_direction, wick_ratio,
                attack_sharpness, harmonic_bloom,
            )
            # Crossfade first 50 samples of sustain with portamento tail
            cf_len = min(50, sustain_samples)
            crossfade_in = (1 - np.cos(np.pi * np.arange(cf_len) / cf_len)) / 2
            sustain_note[:cf_len] *= crossfade_in
            wave_port[-cf_len:] *= (1 - crossfade_in)
            output = np.concatenate([wave_port, sustain_note])
        else:
            output = wave_port

        output = output[:actual_samples]

        # Final envelope
        atk_time = 0.005 + 0.045 * (1 - attack_sharpness)
        envelope = _adsr_envelope(
            len(output), sr, atk_time, 0.15, 0.5, 0.15
        )

        return (output * envelope).astype(np.float32)

    # ── Heartbeat ─────────────────────────────────────────────────

    def _generate_heartbeat(
        self, num_samples: int
    ) -> np.ndarray:
        """Sub-audible pulse to prevent abandonment during silence."""
        t = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        pulse = HEARTBEAT_AMP * np.sin(
            2 * np.pi * HEARTBEAT_FREQ * t
        )
        return pulse.astype(np.float32)

    # ── Alert Sound ───────────────────────────────────────────────

    def _generate_alert(
        self,
        kind: str,
        magnitude: float,
    ) -> np.ndarray:
        """Short distinctive sound for significant events."""
        sr = self.sample_rate
        duration = int(0.15 * sr)
        t = np.arange(duration) / sr

        if kind == SonificationEvent.BREAKOUT:
            # Rising chirp: ascending pitch burst
            f_start = 800
            f_end = 2000
            phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2)
            wave = 0.6 * np.sin(phase)
            envelope = np.sin(np.pi * t / (duration / sr))

        elif kind == SonificationEvent.DIVERGENCE:
            # Dissonant pair: two close frequencies beating
            f1 = 600
            f2 = 640
            wave = 0.3 * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t))
            envelope = np.exp(-3 * t)

        elif kind == SonificationEvent.VOLUME_SPIKE:
            # Sharp percussive hit
            freq = 1200
            wave = 0.4 * np.sin(2 * np.pi * freq * t) * np.exp(-8 * t)
            envelope = np.ones(duration)

        else:
            # Default: short click
            wave = 0.3 * np.random.randn(duration) * np.exp(-10 * t)
            envelope = np.ones(duration)

        return (wave * envelope).astype(np.float32)

    # ── Core Generate Method ──────────────────────────────────────

    def generate(
        self,
        data: pd.DataFrame,
        target_speed: float = 1.0,
    ) -> Tuple[np.ndarray, List[SonificationEvent]]:
        """
        Generate audio from market data.

        Args:
            data: DataFrame with columns: Open, High, Low, Close, Volume
            target_speed: Playback speed multiplier (1.0 = real-time equivalent)

        Returns:
            tuple: (audio_array, list_of_events)
        """
        self.events = []

        # ── Preprocess: Heikin-Ashi ──────────────────────────────
        ha = self._heikin_ashi(data)

        # ── Extract arrays ───────────────────────────────────────
        n = len(ha)
        closes = ha['Close'].values.astype(np.float64)
        opens = ha['Open'].values.astype(np.float64)
        highs = ha['High'].values.astype(np.float64)
        lows = ha['Low'].values.astype(np.float64)
        volumes = data['Volume'].values[-n:].astype(np.float64)

        # ── Compute features ─────────────────────────────────────
        body_sizes = np.abs(closes - opens) / (np.abs(opens) + 1e-12)
        wick_ranges = (highs - lows) / (np.abs(closes) + 1e-12)
        directions = np.where(closes >= opens, 1.0, -1.0)

        # ── Compute reference frame ──────────────────────────────
        refs = self._compute_reference(closes, volumes if 'Volume' in data.columns else None)

        # ── Detect alerts ────────────────────────────────────────
        self._detect_alerts(closes, opens, highs, lows, volumes, body_sizes, wick_ranges, refs)

        # ── Compute pitches ──────────────────────────────────────
        frequencies = self._compute_pitches(closes, refs)

        # ── Compute bass notes ───────────────────────────────────
        bass_notes = self._compute_bass(refs)

        # ── Generate note parameters ─────────────────────────────
        note_base_samples = int(self.duration_per_note * self.sample_rate / target_speed)

        # Sustain from volume (normalized per session)
        vol_mean = np.mean(volumes) if np.mean(volumes) > 0 else 1e-12
        vol_normalized = volumes / vol_mean
        sustain_multipliers = self.sustain_min + np.clip(
            vol_normalized, 0, 5
        ) * (self.sustain_max - self.sustain_min) / 5.0

        # Volume spikes for accent
        if len(volumes) > 10:
            vol_std = np.std(volumes)
            vol_spike_mask = (volumes - vol_mean) > self.volume_spike_threshold * vol_std
        else:
            vol_spike_mask = np.zeros(n, dtype=bool)

        # Attack sharpness from body direction consistency
        attack_sharpness = np.clip(body_sizes * 10, 0.1, 1.0)

        # Harmonic bloom from volume
        harmonic_blooms = np.clip((vol_normalized * 4).astype(int), 0, 7)

        # ── Market state detection ───────────────────────────────
        market_state = self._detect_market_state(closes)

        # Crossfade based on market state
        expected_interval = int(self.sample_rate * self.duration_per_note / target_speed)
        if market_state == 'trending':
            crossfade_samples = int(0.2 * expected_interval)
        elif market_state == 'choppy':
            crossfade_samples = 0
        else:
            crossfade_samples = int(0.08 * expected_interval)

        # ── Generate audio ───────────────────────────────────────
        notes: List[np.ndarray] = []
        prev_freq = None
        silence_threshold_body = self.silence_threshold_body
        silence_threshold_range = self.silence_threshold_range

        # Adjust silence threshold based on market
        if market_state == 'choppy':
            silence_threshold_body *= 0.5
            silence_threshold_range *= 0.5

        event_cursor = 0  # Track which event comes next
        sorted_events = sorted(self.events, key=lambda e: e.candle_index)

        for i in range(n):
            # Check for alerts at this candle
            while (event_cursor < len(sorted_events) and
                   sorted_events[event_cursor].candle_index == i):
                evt = sorted_events[event_cursor]
                alert_sound = self._generate_alert(evt.kind, evt.magnitude)
                notes.append(alert_sound)
                event_cursor += 1

            # Silence check
            if (body_sizes[i] < silence_threshold_body and
                    wick_ranges[i] < silence_threshold_range):
                note = np.zeros(expected_interval, dtype=np.float32)
                if self.heartbeat_enabled:
                    heartbeat = self._generate_heartbeat(expected_interval)
                    note = note + heartbeat
                notes.append(note)
                prev_freq = None
                continue

            # Regular note generation
            freq = frequencies[i]
            direction = directions[i]
            wick = wick_ranges[i]
            sustain_mult = sustain_multipliers[i]
            actual_samples = max(int(note_base_samples * sustain_mult), 100)

            if (self.smooth and prev_freq is not None and actual_samples > 100):
                portamento_sec = 0.03 if market_state == 'trending' else 0.01
                note = self._generate_portamento_note(
                    target_freq=freq,
                    prev_freq=prev_freq,
                    velocity=0.7,
                    portamento_sec=portamento_sec,
                    note_samples=actual_samples,
                    body_direction=direction,
                    wick_ratio=wick,
                    attack_sharpness=attack_sharpness[i],
                    harmonic_bloom=harmonic_blooms[i],
                )
            else:
                note = self._generate_note(
                    freq=freq,
                    velocity=0.7,
                    note_samples=actual_samples,
                    body_direction=direction,
                    wick_ratio=wick,
                    attack_sharpness=attack_sharpness[i],
                    harmonic_bloom=harmonic_blooms[i],
                )

            # Crossfade with previous
            if i > 0 and crossfade_samples > 0 and notes:
                prev_note = notes.pop()
                note = self._crossfade(prev_note, note, crossfade_samples)

            # Pad to expected interval
            if len(note) < expected_interval:
                note = np.pad(note, (0, expected_interval - len(note)))
            else:
                note = note[:expected_interval]

            notes.append(note.astype(np.float32))
            prev_freq = freq

        # ── Interleave bass ─────────────────────────────────────
        audio = np.concatenate(notes) if notes else np.array([], dtype=np.float32)
        audio = self._layer_bass(audio, bass_notes, target_speed)

        # ── Apply effects ───────────────────────────────────────
        if self.effect == 'Chorus':
            audio = _apply_chorus(audio)
        elif self.effect == 'Reverb':
            audio = _apply_reverb(audio)

        # ── Normalize ───────────────────────────────────────────
        audio = _normalize(audio, headroom_db=3.0)

        return audio.astype(np.float32), self.events

    # ── Internal Helpers ───────────────────────────────────────────

    @staticmethod
    def _heikin_ashi(data: pd.DataFrame) -> pd.DataFrame:
        """Apply Heikin-Ashi transformation to smooth candles."""
        ha = data.copy()
        ha['HA_Close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4

        ha_open_list = []
        for i in range(len(data)):
            if i == 0:
                ha_open_list.append((data['Open'].iloc[0] + data['Close'].iloc[0]) / 2)
            else:
                ha_open_list.append((ha_open_list[i - 1] + ha.iloc[i - 1]['HA_Close']) / 2)
        ha['HA_Open'] = ha_open_list
        ha['HA_High'] = data[['High', 'HA_Open', 'HA_Close']].max(axis=1)
        ha['HA_Low'] = data[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

        return ha.rename(columns={
            'HA_Open': 'Open', 'HA_High': 'High',
            'HA_Low': 'Low', 'HA_Close': 'Close',
        })

    def _detect_market_state(self, closes: np.ndarray) -> str:
        """Classify current market state."""
        if len(closes) < 5:
            return 'ranging'

        recent = closes[-5:]
        diffs = np.diff(recent)

        if np.all(diffs > 0) or np.all(diffs < 0):
            return 'trending'

        recent_returns = np.abs(np.diff(recent) / recent[:-1])
        if np.mean(recent_returns) > 0.01:
            return 'choppy'

        return 'ranging'

    @staticmethod
    def _crossfade(
        note_a: np.ndarray,
        note_b: np.ndarray,
        samples: int,
    ) -> np.ndarray:
        """Cosine crossfade between end of note_a and start of note_b."""
        cf = min(samples, len(note_a), len(note_b))
        if cf <= 0:
            return np.concatenate([note_a, note_b])

        t = np.arange(cf) / cf
        fade_out = np.cos(np.pi * t / 2)
        fade_in = np.sin(np.pi * t / 2)

        cross = note_a[-cf:] * fade_out + note_b[:cf] * fade_in
        return np.concatenate([note_a[:-cf], cross, note_b[cf:]])

    def _layer_bass(
        self,
        melody: np.ndarray,
        bass_notes: List[Tuple[int, float]],
        speed: float,
    ) -> np.ndarray:
        """Layer bass drone beneath melody."""
        if not bass_notes:
            return melody

        sr = self.sample_rate
        bass_audio = np.zeros_like(melody)
        scaled_interval = int(self.duration_per_note * sr / speed)

        for candle_idx, bass_freq in bass_notes:
            start_sample = candle_idx * scaled_interval
            if start_sample >= len(bass_audio):
                break

            # Generate bass note (long, smooth)
            duration = scaled_interval * 4  # Bass notes are longer
            end = min(start_sample + duration, len(bass_audio))
            length = end - start_sample
            if length < 10:
                continue

            t = np.arange(length, dtype=np.float64) / sr

            # Smooth sine bass with slight attack
            bass_wave = np.sin(2 * np.pi * bass_freq * t)
            attack = min(0.1, 100 / length)
            env = np.ones(length)
            ramp = int(attack * length)
            if ramp > 0:
                env[:ramp] = np.linspace(0, 1, ramp)
            if length - ramp > 10:
                env[-10:] = np.linspace(env[-1], 0, 10)

            bass_audio[start_sample:end] += bass_wave * env * 0.3

        return _normalize(melody + bass_audio, headroom_db=3.0)
