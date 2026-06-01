How to Run

pip install numpy pandas yfinance PyQt5 sounddevice scipy
cd sonifier_v3
python sonifier_app.py

Quick-Start Guide
First Run (30 seconds)

    Type a ticker — BTC-USD, AAPL, ^GSPC, ETH-USD, GC=F (gold)
    Select a period — 3M is a good default
    Pick a preset — start with Full Picture
    Click Generate & Play — audio plays automatically when ready
    Close your eyes — listen for 2 minutes before looking at any chart

Keyboard Shortcuts
Key	Action
Space	Play / Stop
Enter	Generate new audio
Ctrl+S	Export WAV (via button)
Preset Selection Guide
Preset	When to Use
Momentum Scan	Quick triage — is anything moving? Fast, bright, exposes direction changes immediately
Reversal Hunt	Deep analysis — high vibrato sensitivity reveals indecision; slower playback for careful listening
Volume Analysis	When volume is the question — wide dynamic range makes accumulation/distribution obvious
Full Picture	Default for everything — all features balanced
What to Listen For
Sound	What It Means
Melodic line rising in pitch	Price moving up relative to the trend reference
Triangular waveform notes	Bullish candles (buyers won)
Sawtooth waveform notes	Bearish candles (sellers won)
Thin, hollow sine notes	Doji / indecision
Wobbling notes (vibrato)	Long wicks = market contested this price
Long, sustained notes	High volume — the market cared about this price
Short, staccato plucks	Low volume — market shrugged
Silence	Nothing happened — truly nothing, consolidation
Low bass pulse (~38 Hz)	Silence is intentional; tool is alive
Rising chirp burst	Breakout alert — price cleared a prior high/low
Beating dissonance	Divergence alert — price and momentum disagree
Sharp percussive hit	Volume spike — sudden surge of participation
Smooth portamento slides	Trending market — orderly transitions
Abrupt gaps between notes	Choppy market — fragmentation, no consensus
Reference Mode (Advanced)

In the Advanced panel, choose how the "home key" is computed:
Reference	Character	Best For
EMA-20	Responsive, tracks recent price	Intraday / swing trading
SMA-20	Slightly lagging, stable	Confirming trends
VWAP	Institutional anchor	Intraday — where smart money trades
Session Open	Fixed reference	End-of-day analysis of the full session
Architecture: What This Tool Does and Why

┌──────────────────────────────────────────────────────────────┐
│                        DATA SOURCE                           │
│  yfinance → OHLCV DataFrame                                  │
│  ↓ Apply Heikin-Ashi (noise filter)                          │
│  Cleaned candlestick data                                    │
└───────────────┬──────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│                  FEATURE EXTRACTION                          │
│                                                              │
│  Close → Pitch         Log-ratio from EMA-20/VWAP           │
│  Open vs Close → Timbre Triangle(bull)/Saw(bear)/Sine(doji) │
│  High-Low → Vibrato    Depth + rate from wick ratio          │
│  Body Size → Attack    Sharp(fast)/gradual(slow)             │
│  Volume → Sustain      Duration + harmonic bloom             │
│  EMA-20 → Bass Voice   Sub-bass drone (sparse updates)       │
│  Gaps → Silence        Actual temporal structure preserved   │
│  Patterns → Alerts     Short earcon per detected event       │
└───────────────┬──────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│                  SYNTHESIS ENGINE                            │
│                                                              │
│  Melody Voice (C4-C6)                                       │
│    • Composite timbre per candle                             │
│    • Log-ratio pitch + scale quantization                    │
│    • Vibrato from wick ratio                                 │
│    • ADSR envelope from candle aggression                    │
│    • Harmonic bloom from volume                              │
│                                                              │
│  Bass Voice (C2-C3)                                         │
│    • EMA-20 trend as sub-bass drone                          │
│    • Updates every N candles                                 │
│                                                              │
│  Alert Sounds                                               │
│    • Breakout: rising chirp                                  │
│    • Divergence: beating dissonance                          │
│    • Volume spike: percussive hit                            │
│    • Reversal: descending slap                               │
└───────────────┬──────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│                  OUTPUT                                      │
│                                                              │
│  sounddevice → real-time playback through headphones/speakers│
│  scipy.io.wavfile → WAV export for sharing                   │
│  Event log → text display in UI                             │
│                                                              │
│  No quantization unless user enables it (optional toggle)    │
│  No smoothing — raw structure preserved                      │
│  No interpretation — data delivered, not summarized          │
└──────────────────────────────────────────────────────────────┘

Usage Scenarios
Scenario 1: Quick Market Check (2 minutes)

1. Open app
2. Type BTC-USD → select 7D → preset: Momentum Scan
3. Click Generate
4. Listen for 60 seconds with eyes closed
5. Open eyes → play again while watching price
6. Done

Scenario 2: Deep Weekend Analysis (30 minutes)

1. Load BTC-USD 3M (daily candles)
2. Listen at 0.5x speed with Full Picture preset
3. Note moments where vibrato intensifies → mark on chart
4. Note bass divergence moments → check if trend reversed
5. Export WAV → listen during commute
6. Replay specific sections by regenerating with subset of data

Scenario 3: Compare Instruments

1. Generate BTC-USD 4H with Full Picture
2. Export WAV
3. Generate ETH-USD 4H with same preset
4. Export WAV
5. Listen back-to-back → hear structural similarities/differences
6. Note: which instrument "sounds" more volatile? Which has clearer trends?

Scenario 4: Pre-Trade Routine

1. Before market open, load SPX 1M → 1D timeframe
2. Listen with Reversal Hunt preset
3. Focus on silence → where is the market NOT moving?
4. Focus on alerts → are there recent breakouts or divergences?
5. Form hypothesis → listen for confirmation as price moves

Known Limitations & Next Steps
Limitation	Workaround
yfinance data has occasional gaps for crypto	Use Binance API for production crypto data
Speed adjustment uses simple resampling (pitch shift)	Implement granular synthesis for pitch-independent speed
No live WebSocket mode	Add Binance/Kraken WebSocket adapter (Phase 2)
Single-threaded synthesis blocks UI during generation	Progress bar + thread already implemented; optimize for >10k candles
No chart visualization included	Pair with TradingView in separate window; consider embedding mplfinance
Stereo flow / VPVR / order flow data not implemented	Requires paid data feeds — defer to Phase 2
