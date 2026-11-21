# GTA VI Trophy Rarity Forecast — E/N paired-scenario forecaster

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/durov001/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/durov001/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Built for the [make.fun gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) — a Python program that forecasts every GTA VI trophy's earn rate on Exophase at 2026-11-26T00:00Z as five quantiles, scored by pinball loss on the logit scale.

Referral code: `zsk0hsg3` — https://make.fun/r/zsk0hsg3

## How it works

- **Rate = E / N, modelled separately.** E is a trophy's earners, N is players tracked. New players dilute the rate *and* add earners, so the two are never forecast independently.
- **Launch-window slopes are shrunk, not trusted.** The first 48 hours are the explosive-N phase; the 4.5-day horizon is the slowing phase. Every fitted slope is shrunk ×0.3 (empirical Bayes), fits use only the last 18 hours with exponential weights (12 h half-life), and the N lever arm is damped so N growth saturates over the horizon.
- **Paired scenarios, not averaged quantiles.** 3 N-growth scenarios × 3 progression speeds = 9 paths; each N path carries a consistent E path (new players earn at a factor of the existing rate — near 1 for common trophies, ~0.35 for rare ones). Persistence is a 10th path, an anchor rather than the centre. The median of the 10 paths on the logit scale is the forecast centre.
- **Honest spread from a variance sum.** Scenario dispersion + (lever arm × slope SE)² + refresh variance measured from the logit jumps when N surged in the history + count variance 1/√(E+1) + a floor. Quantiles use t(4) (median ∓ 2.13σ at 5/95%, ∓ 0.74σ at 25/75%). Late story/collection trophies (0.1–5%) get an asymmetric band with a wider upper tail; platinum gets a wide floor.
- **Discipline clamps.** Every quantile sits inside [0.85×, 1.5×] the trophy's last rate (platinum up to 3×); early-story trophies cap q95 at 95%.

## Run it

```bash
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `.out` per case. Standard library only, no network, deterministic (two runs are byte-identical), well under the 8-second cap per case.
