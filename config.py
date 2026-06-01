"""
Sonifier v3.0 — Configuration & Presets
Designed for active use, not training.
"""

SAMPLE_RATE = 44100
BIT_DEPTH = np.float32

# ── Scale Definitions ──────────────────────────────────────────────
SCALE_SEMITONES = {
    'Major':          [0, 2, 4, 5, 7, 9, 11],
    'Natural Minor':  [0, 2, 3, 5, 7, 8, 10],
    'Harmonic Minor': [0, 2, 3, 5, 7, 8, 11],
    'Pentatonic':     [0, 2, 4, 7, 9],
    'Blues':          [0, 3, 5, 6, 7, 10],
    'Dorian':         [0, 2, 3, 5, 7, 9, 10],
    'Mixolydian':     [0, 2, 4, 5, 7, 9, 10],
    'Chromatic':      list(range(12)),
}

# ── Presets ────────────────────────────────────────────────────────
# Each preset is tuned for a specific analytical task.
PRESETS = {
    'Momentum Scan': {
        'scale': 'Major',
        'tonic_freq': 330.0,       # E4 — bright, forward
        'freq_min': 330.0,
        'freq_max': 1319.0,        # E5
        'duration_ms': 280,
        'smooth': True,
        'effect': 'None',
        'speed': 1.0,
        'waveform': 'triangle',
        'vibrato_max_cents': 80,
        'sustain_min': 0.25,
        'sustain_max': 1.5,
        'silence_threshold_body': 0.003,
        'silence_threshold_range': 0.006,
        'heartbeat_enabled': True,
        'bass_enabled': True,
        'bass_freq_min': 55.0,
        'bass_freq_max': 110.0,
        'bass_note_every': 3,      # Update bass every N candles
        'reference_mode': 'ema20',# EMA-20 for responsive trend tracking
        'alert_on_breakout': True,
        'alert_on_divergence': True,
        'alert_on_volume_spike': True,
        'volume_spike_threshold': 2.0,  # Std deviations
    },
    'Reversal Hunt': {
        'scale': 'Harmonic Minor',
        'tonic_freq': 220.0,       # A3 — darker, more tension
        'freq_min': 165.0,
        'freq_max': 990.0,
        'duration_ms': 400,
        'smooth': True,
        'effect': 'None',
        'speed': 0.7,              # Slower for careful listening
        'waveform': 'sine',
        'vibrato_max_cents': 120,  # More vibrato = more visible indecision
        'sustain_min': 0.3,
        'sustain_max': 2.0,
        'silence_threshold_body': 0.002,
        'silence_threshold_range': 0.005,
        'heartbeat_enabled': True,
        'bass_enabled': True,
        'bass_freq_min': 41.0,     # Lower register for deeper bass
        'bass_freq_max': 82.0,
        'bass_note_every': 2,
        'reference_mode': 'vwap', # VWAP — institutional reference
        'alert_on_breakout': True,
        'alert_on_divergence': True,
        'alert_on_volume_spike': True,
        'volume_spike_threshold': 1.5,
    },
    'Volume Analysis': {
        'scale': 'Minor',
        'tonic_freq': 261.0,       # C4 — middle register
        'freq_min': 261.0,
        'freq_max': 1047.0,
        'duration_ms': 350,
        'smooth': False,           # Staccato reveals volume gaps
        'effect': 'None',
        'speed': 1.2,
        'waveform': 'triangle',
        'vibrato_max_cents': 60,
        'sustain_min': 0.2,        # Very short by default
        'sustain_max': 3.0,        # But very long at high volume
        'silence_threshold_body': 0.001,
        'silence_threshold_range': 0.004,
        'heartbeat_enabled': True,
        'bass_enabled': True,
        'bass_freq_min': 55.0,
        'bass_freq_max': 130.0,
        'bass_note_every': 2,
        'reference_mode': 'ema20',
        'alert_on_breakout': False,
        'alert_on_divergence': False,
        'alert_on_volume_spike': True,
        'volume_spike_threshold': 1.8,
    },
    'Full Picture': {
        'scale': 'Dorian',
        'tonic_freq': 293.0,       # D4 — balanced
        'freq_min': 220.0,
        'freq_max': 1760.0,
        'duration_ms': 400,
        'smooth': True,
        'effect': 'Chorus',
        'speed': 1.0,
        'waveform': 'triangle',
        'vibrato_max_cents': 100,
        'sustain_min': 0.3,
        'sustain_max': 2.0,
        'silence_threshold_body': 0.005,
        'silence_threshold_range': 0.008,
        'heartbeat_enabled': True,
        'bass_enabled': True,
        'bass_freq_min': 55.0,
        'bass_freq_max': 110.0,
        'bass_note_every': 3,
        'reference_mode': 'ema20',
        'alert_on_breakout': True,
        'alert_on_divergence': True,
        'alert_on_volume_spike': True,
        'volume_spike_threshold': 2.0,
    },
}

DEFAULT_PRESET = 'Full Picture'

# ── Rendering ──────────────────────────────────────────────────────
TIMEFRAME_OPTIONS = {
    '7D':  ('7d', '1h'),
    '1M':  ('1mo', '1d'),
    '3M':  ('3mo', '1d'),
    '6M':  ('6mo', '1wk'),
    '1Y':  ('1y', '1wk'),
    '5Y':  ('5y', '1mo'),
}

# ── Physical Constants ─────────────────────────────────────────────
CENTS_PER_SEMITONE = 100
SEMITONES_PER_OCTAVE = 12

# ── Pattern Labels ─────────────────────────────────────────────────
PATTERN_NAMES = {
    'doji':            'Doji',
    'hammer':          'Hammer',
    'shooting_star':   'Shooting Star',
    'engulfing_bull':  'Bullish Engulfing',
    'engulfing_bear':  'Bearish Engulfing',
    'morning_star':    'Morning Star',
    'evening_star':    'Evening Star',
    'harami_bull':     'Bullish Harami',
    'harami_bear':     'Bearish Harami',
    'piercing':        'Piercing Line',
    'dark_cloud':      'Dark Cloud Cover',
    'three_white':     'Three White Soldiers',
    'three_black':     'Three Black Crows',
}
