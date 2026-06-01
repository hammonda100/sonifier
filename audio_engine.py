"""
Sonifier v3.0 — Core Audio Synthesis Engine

Converts OHLCV market data into a polyphonic auditory stream optimized
for active trading analysis, not training.

Voice architecture:
  Melody (C4-C6): Full composite timbre per candle
  Bass   (C2-C3): EMA/SMA trend drone, sparse updates
  Alert  (burst): Short distinctive sounds for significant events
  No harmony pad — avoids register collision during volatile periods.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


# ═══════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════

def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


# ═══════════════════════════════════════════════════════════════════
#  WAVEFORM GENERATORS (band-limited)
# ═══════════════════════════════════════════════════════════════════

def _waveform_sine(phase: np.ndarray) -> np.ndarray:
    return np.sin(phase)


def _waveform_triangle(phase: np.ndarray) -> np.ndarray:
    """Band-limited triangle via arcsin."""
    return (2.0 / np.pi) * np.arcsin(np.sin(phase))


def _waveform_sawtooth(phase: np.ndarray, harmonics: int = 20) -> np.ndarray:
    """Band-limited sawtooth via additive synthesis."""
    result = np.zeros_like(phase)
    for h in range(1, harmonics + 1):
        result += (1.0 / h) * np.sin(phase * h)
    return -(2.0 / np.pi) * result


def _waveform_square(phase: np.ndarray, harmonics: int = 16) -> np.ndarray:
    """Band-limited square wave."""
    result = np.zeros_like(phase)
    for h in range(1, harmonics + 2, 2):
        result += (1.0 / h) * np.sin(phase * h)
    return (4.0 / np.pi) * result


WAVEFORM_MAP = {
    'sine':     _waveform_sine,
    'triangle': _waveform_triangle,
    'sawtooth': _waveform_sawtooth,
    'square':   _waveform_square,
}


# ═══════════════════════════════════════════════════════════════════
#  ADSR ENVELOPE
# ═══════════════════════════════════════════════════════════════════

def _adsr_envelope(
    num_samples: int,
    sample_rate: int,
    attack_time: float,
    decay_time: float,
    sustain_level: float,
    release_time: float,
) -> np.ndarray:
    t = np.arange(num_samples) / sample_rate
    total = num_samples / sample_rate
    env = np.zeros(num_samples, dtype=np.float64)

    atk_end = min(attack_time, total)
    dec_end = min(atk_end + decay_time, total)
    rel_start = max(total - release_time, dec_end)

    # Attack: 0 → 1 (cosine quarter-wave)
    m = t < atk_end
    if np.any(m):
        d = max(atk_end, 1e-12)
        env[m] = 0.5 * (1.0 - np.cos(np.pi * t[m] / d))

    # Decay: 1 → sustain_level
    m = (t >= atk_end) & (t < dec_end)
    if np.any(m):
        d = max(dec_end - atk_end, 1e-12)
        env[m] = 1.0 + (sustain_level - 1.0) * (t[m] - atk_end) / d

    # Sustain
    m = (t >= dec_end) & (t < rel_start)
    if np.any(m):
        env[m] = sustain_level

    # Release: sustain_level → 0
    m = t >= rel_start
    if np.any(m):
        d = max(total - rel_start, 1e-12)
        env[m] = sustain_level * 0.5 * (1.0 + np.cos(np.pi * (t[m] - rel_start) / d))

    # Final sample fade-out to prevent click
    fade_len = min(64, num_samples)
    env[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

    return env


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════════════════

def _normalize(audio: np.ndarray, headroom_db: float = HEADROOM_DB) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-12:
        return audio
    return audio * (db_to_linear(-headroom_db) / peak)


def _apply_reverb(
    audio: np.ndarray,
    wet: float = 0.2,
    delay_ms: float = 30.0,
    feedback: float = 0.35,
    sr: int = 44100,
) -> np.ndarray:
    delay_samples = int(delay_ms * sr / 1000.0)
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
    mix: float = 0.30,
    sr: int = 44100,
) -> np.ndarray:
    """Simple chorus via fractional-delay modulation."""
    t = np.arange(len(audio), dtype=np.float64) / sr
    mod = depth_cents * np.sin(2.0 * np.pi * rate_hz * t)
    mod_ratio = 2.0 ** (mod / 1200.0)

    idx = np.arange(len(audio))
    delay_idx = np.cumsum(mod_ratio)
    idx_floor = np.clip(np.floor(delay_idx).astype(int), 0, len(audio) - 2)
    frac = delay_idx - idx_floor

    delayed = audio[idx_floor] * (1.0 - frac) + audio[idx_floor + 1] * frac
    return audio * (1.0 - mix) + delayed * mix


def _apply_filter_lowpass(audio: np.ndarray, cutoff_hz: float, sr: int = 44100) -> np.ndarray:
    """First-order low-pass IIR."""
    alpha = (1.0 / sr) / ((1.0 / (2.0 * np.pi * cutoff_hz)) + (1.0 / sr))
    out = np.zeros_like(audio)
    out[0] = audio[0]
    for i in range(1, len(audio)):
        out[i] = out[i - 1] + alpha * (audio[i] - out[i - 1])
    return out


def _apply_filter_highpass(audio: np.ndarray, cutoff_hz: float, sr: int = 44100) -> np.ndarray:
    """First-order high-pass IIR."""
    alpha = (1.0 / sr) / ((1.0 / (2.0 * np.pi * cutoff_hz)) + (1.0 / sr))
    out = np.zeros_like(audio)
    out[0] = audio[0]
    for i in range(1, len(audio)):
        out[i] = alpha * (out[i - 1] + audio[i] - audio[i - 1])
    return out


# ═══════════════════════════════════════════════════════════════════
#  SCALE QUANTIZATION
# ═══════════════════════════════════════════════════════════════════

def _get_scale_frequencies(scale: str, tonic_freq: float,
                           f_min: float, f_max: float) -> np.ndarray:
    semitones = SCALE_SEMITONES.get(scale, SCALE_SEMITONES['Dorian'])
    freqs = []
    for octave in range(0, 10):
        for st in semitones:
            f = tonic_freq * (2.0 ** ((octave * 12 + st) / 12.0))
            if f_min * 0.5 <= f <= f_max * 2.0:
                freqs.append(f)
    return np.array(sorted(freqs), dtype=np.float64)


def _quantize_to_scale(freq: float, scale_freqs: np.ndarray) -> float:
    if len(scale_freqs) == 0:
        return freq
    idx = int(np.argmin(np.abs(np.log2(scale_freqs / freq))))
    idx = max(0, min(idx, len(scale_freqs) - 1))
    return float(scale_freqs[idx])


# ═══════════════════════════════════════════════════════════════════
#  PRIMARY SYNTHESES
# ═══════════════════════════════════════════════════════════════════

def _synth_note(
    freq: float,
    velocity: float,
    num_samples: int,
    sr: int,
    waveform,
    body_direction: float,
    wick_ratio: float,
    vibrato_cents: float,
    harmonic_bloom: int,
    attack_sharpness: float,
) -> np.ndarray:
    """
    Synthesize one candle's note.

    Parameters map 1:1 onto the data dimensions described in the
    architecture document.  Nothing is summarised.
    """
    actual = max(num_samples, 100)
    t = np.arange(actual, dtype=np.float64) / sr

    # ── base phase ─────────────────────────────────────────────
    base_phase = 2.0 * np.pi * freq * t

    # ── vibrato ────────────────────────────────────────────────
    v_depth = min(wick_ratio * vibrato_cents, vibrato_cents)
    v_rate  = 4.0 + 4.0 * min(wick_ratio, 1.0)
    vibrato = vibrato_cents * np.sin(2.0 * np.pi * v_rate * t)
    ratio   = 2.0 ** (vibrato / 1200.0)
    phase   = base_phase * ratio

    # ── pitch sweep (subtle lean in candle direction) ──────────
    sweep = np.linspace(0, body_direction * 0.012, actual)
    sweep_phase = np.cumsum(2.0 ** (sweep / 12.0)) * freq * 2.0 * np.pi / sr
    phase = 0.70 * phase + 0.30 * sweep_phase

    # ── waveform + harmonics ───────────────────────────────────
    wave = waveform(phase) * velocity
    for p in range(2, 2 + min(harmonic_bloom, 7)):
        wave += np.sin(phase * p) * (0.25 / (p - 0.3)) * velocity

    # ── ADSR envelope ──────────────────────────────────────────
    atk  = 0.005 + 0.045 * (1.0 - attack_sharpness)
    dec  = 0.10 + 0.20 * velocity
    sus  = 0.25 + 0.60 * velocity
    rel  = 0.10 + 0.20 * velocity
    env  = _adsr_envelope(actual, sr, atk, dec, sus, rel)

    return (wave * env).astype(np.float32)


def _synth_portamento_note(
    target_freq: float,
    prev_freq: float,
    velocity: float,
    port_sec: float,
    num_samples: int,
    sr: int,
    waveform,
    body_direction: float,
    wick_ratio: float,
    vibrato_cents: float,
    attack_sharpness: float,
    harmonic_bloom: int,
) -> np.ndarray:
    """Note with smooth pitch glide from previous frequency."""
    actual = max(num_samples, 100)
    port_smpl = min(int(port_sec * sr), actual // 2)
    sust_smpl = actual - port_smpl

    # Portamento frequency sweep
    port_freqs = np.linspace(prev_freq, target_freq, port_smpl + 1)
    # Phase via instantaneous frequency integration
    port_phases = 2.0 * np.pi * port_freqs / sr
    port_phase  = np.cumsum(port_phases)

    # Vibrato on portamento (halved depth)
    v_depth = min(wick_ratio * vibrato_cents * 0.5, vibrato_cents)
    v_rate  = 4.0 + 4.0 * min(wick_ratio, 1.0)
    t_port  = np.arange(port_smpl, dtype=np.float64) / sr
    vibrato = v_depth * np.sin(2.0 * np.pi * v_rate * t_port)
    ratio   = 2.0 ** (vibrato / 1200.0)
    port_phase *= ratio

    wave_port = waveform(port_phase) * velocity

    # Portamento tail envelope
    wave_port *= np.linspace(1.0, 0.3, port_smpl) ** 0.5

    # Sustain portion
    if sust_smpl > 20:
        sust = _synth_note(
            target_freq, velocity, sust_smpl, sr,
            waveform, body_direction, wick_ratio,
            vibrato_cents, harmonic_bloom, attack_sharpness,
        )
        # Crossfade join
        cf = min(50, sust_smpl, port_smpl)
        cf_in  = (1.0 - np.cos(np.pi * np.arange(cf) / cf)) / 2.0
        cf_out = 1.0 - cf_in
        wave_port[-cf:]    *= cf_out
        sust[:cf]          *= cf_in
        out = np.concatenate([wave_port, sust])
    else:
        out = wave_port

    out = out[:actual]
    env = _adsr_envelope(len(out), sr, 0.005 + 0.045 * (1 - attack_sharpness), 0.15, 0.5, 0.15)
    return (out * env).astype(np.float32)


def _synth_bass_note(
    freq: float,
    duration_samples: int,
    sr: int,
) -> np.ndarray:
    """Smooth sub-bass drone note."""
    t = np.arange(duration_samples, dtype=np.float64) / sr
    wave = np.sin(2.0 * np.pi * freq * t)

    # Gentle attack / release
    ramp = min(int(0.02 * sr), duration_samples // 4)
    if ramp > 0:
        env = np.ones(duration_samples)
        env[:ramp]             = np.linspace(0.0, 1.0, ramp)
        env[-min(20, duration_samples):] = np.linspace(env[-1], 0.0, min(20, duration_samples))
        wave *= env

    return (wave * 0.35).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  ALERT SOUNDS
# ═══════════════════════════════════════════════════════════════════

class AlertEvent:
    """Represents a detected significant market event."""
    BREAKOUT      = 'breakout'
    DIVERGENCE    = 'divergence'
    VOLUME_SPIKE  = 'volume_spike'
    REVERSAL      = 'reversal'

    def __init__(self, kind, candle_index, description, magnitude):
        self.kind        = kind
        self.candle_index = candle_index
        self.description = description
        self.magnitude   = magnitude

    def __repr__(self):
        return f"Alert({self.kind}, idx={self.candle_index}, '{self.description}')"


def _synth_alert(kind: str, magnitude: float, sr: int = 44100) -> np.ndarray:
    """Short distinctive sound for significant events."""
    dur = int(0.15 * sr)
    t   = np.arange(dur, dtype=np.float64) / sr

    if kind == AlertEvent.BREAKOUT:
        # Rising chirp burst
        f0, f1 = 800.0, 2400.0
        phase  = 2.0 * np.pi * (f0 * t + (f1 - f0) * t * t / dur)
        wave   = 0.55 * np.sin(phase)
        env    = np.sin(np.pi * np.arange(dur) / dur)

    elif kind == AlertEvent.DIVERGENCE:
        # Two close frequencies beating (dissonance)
        f1, f2 = 620.0, 670.0
        wave   = 0.25 * (np.sin(2.0 * np.pi * f1 * t)
                        + np.sin(2.0 * np.pi * f2 * t))
        env    = np.exp(-4.0 * t)

    elif kind == AlertEvent.VOLUME_SPIKE:
        # Sharp percussive hit
        freq = 1400.0
        wave = 0.45 * np.sin(2.0 * np.pi * freq * t) * np.exp(-10.0 * t)
        env  = np.ones(dur)

    elif kind == AlertEvent.REVERSAL:
        # Descending minor-second slap
        f0, f1 = 700.0, 630.0
        wave   = 0.4 * (np.sin(2.0 * np.pi * f0 * t)
                        * np.exp(-6.0 * t)
                       + np.sin(2.0 * np.pi * f1 * t)
                        * np.exp(-6.0 * (t + 0.02)))
        env = np.ones(dur)

    else:
        wave = 0.2 * np.random.randn(dur) * np.exp(-8.0 * t)
        env  = np.ones(dur)

    return (wave * env).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  EVENT DETECTION
# ═══════════════════════════════════════════════════════════════════

def _detect_events(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    body_sizes: np.ndarray,
    wick_ratios: np.ndarray,
    refs: np.ndarray,
    vp_breakout: float = 20,
    vp_spike: float = 2.0,
    vp_reversal: bool = True,
    vp_divergence: bool = True,
) -> List[AlertEvent]:
    """Scan candle data for actionable events."""
    events = []
    n = len(closes)
    if n < 5:
        return events

    # ── Volume spikes ──────────────────────────────────────────
    if n > 10:
        mn = float(np.mean(volumes))
        sd = float(np.std(volumes))
        if sd > 1e-12:
            for i in range(1, n):
                z = (volumes[i] - mn) / sd
                if z > vp_spike:
                    events.append(AlertEvent(
                        kind=AlertEvent.VOLUME_SPIKE,
                        candle_index=i,
                        description=f'Volume spike {z:.1f}σ',
                        magnitude=float(z),
                    ))

    # ── Breakouts / breakdowns (close outside prior N-period range) ──
    if n > vp_breakout:
        for i in range(vp_breakout, n):
            prior_hi  = np.max(highs[max(0, i - vp_breakout):i])
            prior_lo  = np.min(lows[max(0, i - vp_breakout):i])
            if closes[i] > prior_hi and body_sizes[i] > 0.005:
                events.append(AlertEvent(
                    kind=AlertEvent.BREAKOUT,
                    candle_index=i,
                    description='Breakout above prior high',
                    magnitude=float(closes[i] - prior_hi) / prior_hi,
                ))
            elif closes[i] < prior_lo and body_sizes[i] > 0.005:
                events.append(AlertEvent(
                    kind=AlertEvent.BREAKOUT,
                    candle_index=i,
                    description='Breakdown below prior low',
                    magnitude=float(prior_lo - closes[i]) / prior_lo,
                ))

    # ── Divergence (price up, body momentum weakening) ────────
    if vp_divergence:
        for i in range(5, n):
            if (closes[i] > closes[i - 4] and
                    body_sizes[i] < body_sizes[i - 4] * 0.4 and
                    body_sizes[i - 4] > 0.015):
                events.append(AlertEvent(
                    kind=AlertEvent.DIVERGENCE,
                    candle_index=i,
                    description='Price divergence — weaker body at higher price',
                    magnitude=float(closes[i] - closes[i - 4]) / closes[i - 4],
                ))

    # ── Three-candle reversals ────────────────────────────────
    if vp_reversal:
        for i in range(2, n):
            d0 = 1 if closes[i] >= opens[i] else -1
            d1 = 1 if closes[i - 1] >= opens[i - 1] else -1
            d2 = 1 if closes[i - 2] >= opens[i - 2] else -1
            if d0 != d1 != d2:
                events.append(AlertEvent(
                    kind=AlertEvent.REVERSAL,
                    candle_index=i,
                    description='Three-candle direction reversal',
                    magnitude=1.0,
                ))

    return sorted(events, key=lambda e: e.candle_index)


# ═══════════════════════════════════════════════════════════════════
#  MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate(
    data: pd.DataFrame,
    # Audio parameters
    scale: str = 'Dorian',
    tonic_freq: float = 293.0,
    freq_min: float = 220.0,
    freq_max: float = 1760.0,
    duration_per_note: float = 0.40,     # seconds per candle at 1x
    sr: int = 44100,
    smooth: bool = True,
    waveform_name: str = 'triangle',
    effect: str = 'None',
    # Sonification parameters
    vibrato_cents: float = 100.0,
    sustain_min: float = 0.3,
    sustain_max: float = 2.0,
    silence_body: float = 0.005,
    silence_range: float = 0.008,
    heartbeat: bool = True,
    # Bass parameters
    bass: bool = True,
    bass_lo: float = 55.0,
    bass_hi: float = 110.0,
    bass_every: int = 3,
    # Reference
    ref_mode: str = 'ema20',
    ref_window: int = 20,
    # Speed
    speed: float = 1.0,
    # Alerts
    alerts: dict | None = None,
) -> Tuple[np.ndarray, List[AlertEvent]]:
    """
    Generate audio from OHLCV data.

    Parameters
    ----------
    data: DataFrame with columns Open, High, Low, Close, Volume.
    All other parameters control sonification behaviour.

    Returns
    -------
    (audio_samples, list_of_detected_events)
    """
    if alerts is None:
        alerts = {}

    # ── Validate ────────────────────────────────────────────
    required = ['Open', 'High', 'Low', 'Close']
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Data missing columns: {missing}")

    n = len(data)
    if n < 2:
        return np.zeros(0, dtype=np.float32), []

    waveform = WAVEFORM_MAP.get(waveform_name, _waveform_triangle)

    # ── Heikin-Ashi transform ────────────────────────────────
    ha = data.copy()
    ha_close = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4.0
    ha_open  = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            ha_open[i] = (data['Open'].iloc[0] + data['Close'].iloc[0]) / 2.0
        else:
            ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_open_s  = pd.Series(ha_open,  index=data.index)
    ha_close_s = pd.Series(ha_close, index=data.index)
    ha_high  = pd.concat([data['High'], ha_open_s, ha_close_s], axis=1).max(axis=1)
    ha_low   = pd.concat([data['Low'],  ha_open_s, ha_close_s], axis=1).min(axis=1)

    closes = ha_close.values.astype(np.float64)
    opens  = ha_open.astype(np.float64)
    highs  = ha_high.values.astype(np.float64)
    lows   = ha_low.values.astype(np.float64)
    volumes = np.zeros(n, dtype=np.float64)
    if 'Volume' in data.columns:
        volumes = data['Volume'].values[-n:].astype(np.float64)

    # ── Derived features ──────────────────────────────────────
    body_sizes  = np.abs(closes - opens) / (np.abs(opens) + 1e-12)
    wick_ranges = (highs - lows) / (np.abs(closes) + 1e-12)
    directions  = np.where(closes >= opens, 1.0, -1.0)

    # ── Reference frame (dynamic tonic) ───────────────────────
    if ref_mode == 'session_open':
        refs = np.full(n, closes[0], dtype=np.float64)
    elif ref_mode == 'sma':
        series = pd.Series(closes)
        sma = series.rolling(window=ref_window, min_periods=1).mean()
        refs = sma.values.astype(np.float64)
    elif ref_mode == 'vwap' and 'Volume' in data.columns:
        cvol  = np.cumsum(closes * volumes)
        cvs   = np.cumsum(volumes)
        refs  = (cvol / np.maximum(cvs, 1e-12)).astype(np.float64)
    else:  # default: EMA
        series = pd.Series(closes)
        ema = series.ewm(span=ref_window, adjust=False).mean()
        refs = ema.values.astype(np.float64)
        # Fill leading NaN
        first = ema.first_valid_index() or 0
        if first > 0:
            refs[:first] = closes[0]

    # ── Scale frequencies ─────────────────────────────────────
    scale_freqs = _get_scale_frequencies(scale, tonic_freq, freq_min, freq_max)

    # ── Pitch computation (log-ratio, UNSCALED, then quantize) ─
    frequencies = np.empty(n, dtype=np.float64)
    for i in range(n):
        ref = refs[i] if refs[i] > 1e-12 else closes[0]
        price = closes[i]
        if price < 1e-12 or ref < 1e-12:
            frequencies[i] = freq_min
            continue
        log_ratio = np.log(price / ref)
        semitones = log_ratio * (12.0 / np.log(2))
        raw_freq = ref * (2.0 ** (semitones / 12.0))
        raw_freq = max(freq_min, min(freq_max, raw_freq))
        frequencies[i] = _quantize_to_scale(raw_freq, scale_freqs)

    # ── Bass notes (sparse) ───────────────────────────────────
    bass_notes = []
    if bass:
        last_bass = None
        for i in range(0, n, max(1, bass_every)):
            bass_freq = refs[i] * (2.0 ** (-2))  # 2 octaves below tonic
            bass_freq = max(bass_lo, min(bass_hi, bass_freq))
            bass_freq = _quantize_to_scale(bass_freq, scale_freqs)
            if last_bass is None or abs(bass_freq - last_bass) > last_bass * 0.04:
                bass_notes.append((i, bass_freq))
                last_bass = bass_freq

    # ── Event detection ───────────────────────────────────────
    vp_breakout = alerts.get('breakout_window', 20)
    vp_spike    = alerts.get('spike_std', 2.0)
    do_reversal = alerts.get('reversal', True)
    do_diverge  = alerts.get('divergence', True)
    do_breakout = alerts.get('breakout', True)
    do_volspike = alerts.get('volume_spike', True)

    events = []
    if do_reversal or do_diverge or do_breakout or do_volspike:
        events = _detect_events(
            closes, opens, highs, lows, volumes,
            body_sizes, wick_ranges, refs,
            vp_breakout=vp_breakout,
            vp_spike=vp_spike,
            vp_reversal=do_reversal,
            vp_divergence=do_diverge,
        )
        # Filter by alert flags
        keep = []
        for ev in events:
            if ev.kind == AlertEvent.BREAKOUT      and not do_breakout:     continue
            if ev.kind == AlertEvent.DIVERGENCE    and not do_diverge:      continue
            if ev.kind == AlertEvent.VOLUME_SPIKE  and not do_volspike:     continue
            if ev.kind == AlertEvent.REVERSAL      and not do_reversal:     continue
            keep.append(ev)
        events = keep

    # ── Market state ──────────────────────────────────────────
    if n >= 5:
        recent_diffs = np.diff(closes[-5:])
        if np.all(recent_diffs > 0) or np.all(recent_diffs < 0):
            market_state = 'trending'
        else:
            ret = np.abs(np.diff(closes[-10:])) / np.abs(closes[-10:-1] + 1e-12)
            market_mean_ret = np.mean(ret) if len(ret) > 0 else 0
            market_state = 'choppy' if market_mean_ret > 0.008 else 'ranging'
    else:
        market_state = 'ranging'

    # Crossfade samples
    base_interval = int(sr * duration_per_note / speed)
    if market_state == 'trending':
        crossfade = int(0.20 * base_interval)
    elif market_state == 'choppy':
        crossfade = 0
    else:
        crossfade = int(0.08 * base_interval)

    # Silence thresholds adapt to market state
    s_body = silence_body
    s_range = silence_range
    if market_state == 'choppy':
        s_body *= 0.5
        s_range *= 0.5

    # ── Render notes ──────────────────────────────────────────
    events_by_idx = {}
    for ev in events:
        events_by_idx.setdefault(ev.candle_index, []).append(ev)

    notes: List[np.ndarray] = []
    prev_freq = None
    heart_beat = heartbeat  # local alias

    for i in range(n):
        # Insert alert sounds
        if i in events_by_idx:
            for ev in events_by_idx[i]:
                alert = _synth_alert(ev.kind, ev.magnitude, sr)
                notes.append(alert)

        # Silence check
        if body_sizes[i] < s_body and wick_ranges[i] < s_range:
            gap = np.zeros(base_interval, dtype=np.float32)
            if heart_beat:
                # Sub-bass pulse at ~38 Hz
                t_beat = np.arange(base_interval, dtype=np.float64) / sr
                pulse = 0.008 * np.sin(2.0 * np.pi * 38.0 * t_beat)
                gap = (gap + pulse).astype(np.float32)
            notes.append(gap)
            prev_freq = None
            continue

        freq       = frequencies[i]
        direction  = directions[i]
        wick       = wick_ranges[i]
        vol_mean   = np.mean(volumes) if np.mean(volumes) > 1e-12 else 1e-12
        vol_norm   = volumes[i] / vol_mean
        sustain_m  = sustain_min + np.clip(vol_norm, 0, 5.0) * (sustain_max - sustain_min) / 5.0
        actual     = max(int(base_interval * sustain_m), 100)
        atk_sharp  = float(np.clip(body_sizes[i] * 10.0, 0.1, 1.0))
        bloom      = int(np.clip((vol_norm * 4.0), 0, 7))

        if smooth and prev_freq is not None and actual > 100 and abs(freq - prev_freq) / max(prev_freq, 1) > 0.01:
            port_sec = 0.03 if market_state == 'trending' else 0.01
            note = _synth_portamento_note(
                target_freq=freq, prev_freq=prev_freq,
                velocity=0.7, port_sec=port_sec,
                num_samples=actual, sr=sr,
                waveform=waveform, body_direction=direction,
                wick_ratio=wick, vibrato_cents=vibrato_cents,
                attack_sharpness=atk_sharp, harmonic_bloom=bloom,
            )
        else:
            note = _synth_note(
                freq=freq, velocity=0.7,
                num_samples=actual, sr=sr,
                waveform=waveform, body_direction=direction,
                wick_ratio=wick, vibrato_cents=vibrato_cents,
                harmonic_bloom=bloom, attack_sharpness=atk_sharp,
            )

        # Crossfade
        if i > 0 and crossfade > 0 and notes:
            prev = notes.pop()
            note = _crossfade(prev, note, crossfade)

        # Pad / trim
        if len(note) < base_interval:
            note = np.pad(note, (0, base_interval - len(note)))
        else:
            note = note[:base_interval]

        notes.append(note.astype(np.float32))
        prev_freq = freq

    # ── Concatenate ───────────────────────────────────────────
    if not notes:
        return np.zeros(0, dtype=np.float32), events

    audio = np.concatenate(notes)

    # ── Bass layer ────────────────────────────────────────────
    if bass and bass_notes:
        audio = _layer_bass(audio, bass_notes, base_interval, sr,
                            bass_lo, bass_hi, scale_freqs)

    # ── Effects ───────────────────────────────────────────────
    if effect == 'Chorus':
        audio = _apply_chorus(audio)
    elif effect == 'Reverb':
        audio = _apply_reverb(audio)

    # ── Normalize ─────────────────────────────────────────────
    audio = _normalize(audio)

    return audio.astype(np.float32), events


def _crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Cosine crossfade: end of a into start of b."""
    n = min(n, len(a), len(b))
    if n <= 0:
        return np.concatenate([a, b])
    t = np.arange(n, dtype=np.float64) / n
    fade_out = np.cos(np.pi * t / 2.0)
    fade_in  = np.sin(np.pi * t / 2.0)
    return np.concatenate([a[:-n], a[-n:] * fade_out + b[:n] * fade_in, b[n:]])


def _layer_bass(
    melody: np.ndarray,
    bass_notes: List[Tuple[int, float]],
    interval_samples: int,
    sr: int,
    bass_lo: float,
    bass_hi: float,
    scale_freqs: np.ndarray,
) -> np.ndarray:
    """Overlay bass drone track onto melody."""
    bass = np.zeros(len(melody), dtype=np.float32)
    for candle_idx, bass_freq in bass_notes:
        start = candle_idx * interval_samples
        if start >= len(bass):
            break
        dur = interval_samples * 5
        end  = min(start + dur, len(bass))
        length = end - start
        if length < 20:
            continue

        t = np.arange(length, dtype=np.float64) / sr
        wave = np.sin(2.0 * np.pi * bass_freq * t)

        # Envelope: gentle attack, long sustain, fade at end
        env = np.ones(length, dtype=np.float64)
        atk = min(int(0.04 * sr), length // 3)
        if atk > 0:
            env[:atk] = np.linspace(0.0, 1.0, atk)
        fade_len = min(200, length)
        env[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

        bass[start:end] += (wave * env * 0.30).astype(np.float32)

    return _normalize(melody + bass)
