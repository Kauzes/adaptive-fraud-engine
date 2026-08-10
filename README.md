# Adaptive Fraud Engine

A fraud detection engine with **no presets and no hand-tuned weights**. Every customer
starts blank. The engine learns what is normal for each person from their own transactions,
and tunes its own decision thresholds from the rulings analysts give it.

This is the direct answer to the limitation of a rule-based engine: *"weights are hand-tuned,
not learned."* Here nothing is hand-tuned. The score is derived from the model, and the
thresholds move on their own.

It stays explainable. Every decision comes with the specific reasons that produced it, and
the score is exactly the sum of its parts — no hidden term, no black box.

```
$ python -m adaptive_fraud watch --archetype student

 txn     amount   expected   conf   score  decision
   1         44        250   0.00    0.00  approve     <- knows nothing, borrows the population
   3        110         61   0.14    0.00  approve
  24        388         65   0.66    7.81  review
  60        128         63   0.83    1.18  approve     <- learned this person spends ~60

Probe: 720 at 03:00 from an unknown device in RU
  against this customer's learned profile -> REJECT (score 21.12)
      - 720 is 4.4 sd above what their own history predicts
      - 03:00 is outside this customer's usual hours
      - device unknown-device has never been seen for this customer
      - first transaction from RU
```

The engine is never told that customer is a student, or that students spend about 60. It
works it out.

## Measured results

Eight independent simulations, pooled: **83,880 transactions, 396 fraudulent (0.47%)**,
across account-takeover, card-testing and slow-escalation attacks.

| approach | precision | recall | F1 | false-positive rate | false alarms |
|---|---|---|---|---|---|
| One global limit (>3,000) | 0.9% | 29.8% | 0.017 | 16.43% | 13,720 |
| Learned baseline, fixed cutoffs | 49.1% | 59.8% | 0.539 | 0.29% | 246 |
| **Learned baseline, adaptive cutoffs** | **53.6%** | **69.2%** | **0.604** | **0.28%** | **237** |

Two separate effects, deliberately measured apart:

- **Personalisation** is what does the heavy lifting. Judging each customer against their own
  baseline instead of one global limit cuts false alarms by **98%** (13,720 → 237) while
  *raising* recall from 29.8% to 69.2%. A single limit is hopeless because it flags every
  business owner constantly and still misses fraud against small spenders.
- **Adaptive thresholds** add a smaller, real gain on top: recall 59.8% → 69.2% with slightly
  fewer false alarms. Not a trade-off — better on both axes.

Reproduce with `python -m adaptive_fraud evaluate`.

## How it works

### Scoring in bits of surprise

Signals are measured in **bits of surprise** — the negative log probability of what was
observed under the current model. Bits are additive across independent signals, so a total
score needs no invented weights; the numbers come out of the model itself.

| Signal | Source |
|---|---|
| Amount | Gaussian tail probability in log space, against a learned mean and spread |
| Hour | Smoothed circular histogram of when this customer actually transacts |
| Device / country | Add-alpha smoothed categorical, scaled by how well the customer is known |
| Velocity | Attempts inside a 10-minute window |
| Impossible travel | Country change inside 5 minutes |

Amounts are modelled in **log space** because spending is roughly log-normal: a customer who
usually pays 50 and occasionally 500 is unremarkable on a log scale and a wild outlier on a
linear one.

### Cold start: empirical-Bayes shrinkage

A blank customer has no baseline, and that window is exactly when account-takeover fraud is
easiest. Instead of a preset or an arbitrary grace period, a new customer borrows the
population and is pulled toward their own behaviour as evidence arrives:

```
w = n / (n + 12)
estimate = w * personal + (1 - w) * population
```

At n=12 the estimate is a 50/50 blend; by n=60 it is 83% personal. No cliff edge, no day on
which behaviour suddenly changes.

### Online estimation

Everything updates in O(1) time and memory — an unbounded stream cannot be re-scanned.

- **Welford's algorithm** for mean and variance. The naive sum-of-squares formula loses
  catastrophic precision when variance is small relative to the mean, which is exactly the
  case for a customer who reliably spends about the same.
- **EWMA** for the amount baseline, so the model forgets. Someone who gets a raise should not
  be judged for years against who they used to be.
- **Circular smoothing** for hours, because 23:00 and 00:00 are adjacent and a plain
  histogram treats a midnight-crossing habit as two unrelated peaks.

### The feedback loop

Analysts already rule on flagged transactions, so the labels are free — no annotation project
required. Each ruling moves the thresholds:

| Event | Meaning | Response |
|---|---|---|
| Analyst approves a flagged payment | false positive | raise the bar |
| Analyst rejects one | true positive | hold, creep down |
| Fraud surfaces after auto-approval | false negative | lower the bar, weighted 1.5x |

False negatives are weighted harder because the costs are not symmetric: an annoyed customer
is cheaper than a drained account. This is a proportional controller, not a learned model —
deliberately, so the thresholds stay two inspectable numbers with an audit trail.

### Resisting poisoning

A learning system invites a specific attack: make small odd payments, let the baseline follow
you, then take everything. Three defences:

1. **Flagged transactions are never learned from** until a human clears them.
2. **Updates are clipped** to ±2.5 sd, so no single transaction can drag the baseline.
3. **Analyst-approved transactions skip the clip** — once a person has vouched for it, the
   unattended-baseline threat model no longer applies, and clipping would only block the
   engine from absorbing a genuine change like a pay rise.

The simulation includes an explicit `escalation` attack that tries exactly this.

## Try it

No dependencies. Developed and tested on Python 3.11; the syntax requires 3.10 or newer.

```bash
python -m adaptive_fraud watch --archetype business   # watch a baseline form from nothing
python -m adaptive_fraud evaluate                     # reproduce the results table
python -m unittest discover -s tests -t .             # 41 tests
```

## Honest limitations

The numbers above come from **simulated data, not real fraud**. That buys ground-truth labels
you cannot get otherwise, and it means the results measure the engine against my own model of
how fraud looks. Specifically:

- **The simulated analyst is always right.** Real reviewers are wrong sometimes, and a
  feedback loop fed bad labels will confidently learn the wrong thresholds. This is the single
  biggest gap between the evaluation and reality.
- **Single seeds swing wildly.** Fraud is rare, so one simulation yields a few dozen positives
  and its recall moves by tens of points between seeds. Everything quoted here is pooled over
  eight runs; there is a test (`test_single_seeds_vary_enough_to_mislead`) whose only job is to
  keep this caveat honest.
- **The population prior only catches egregious cold-start fraud.** A brand-new customer's
  720 payment scores 1.56 bits — below the review threshold — because the population mixes
  students with business owners. It catches a 90,000 transfer, not a merely suspicious one.
  A production system would segment the prior rather than pooling everyone.
- **A step change stays flagged until someone reviews it.** Adaptation to legitimate change
  needs the analyst; the engine cannot do it alone. Asserted directly in
  `test_a_step_change_stays_flagged_while_nobody_reviews_it`.
- **Impossible travel** compares country changes against a clock, not real distance or
  feasible travel speed.

## Layout

```
adaptive_fraud/
  estimators.py   Welford, EWMA, circular-hour and categorical estimators
  population.py   the population prior a blank customer borrows from
  profile.py      per-customer learned state, shrinkage, update clipping
  engine.py       scoring in bits, decisions, the learn/resolve loop
  thresholds.py   the feedback-driven tuner (and a static control)
  simulation.py   synthetic customers and injected fraud, with labels
  evaluate.py     precision/recall against two controls
tests/            41 tests
```

`estimators.py` and `thresholds.py` have no dependency on the rest, which is why the maths can
be tested directly rather than only through the engine.

## Related

- [fraud-detection-engine](https://github.com/Kauzes/fraud-detection-engine) — rule-based, per-user statistics, Python + REST
- [sentinel-pay-android](https://github.com/Kauzes/sentinel-pay-android) — per-profile thresholds, native Android

## License

MIT — see [LICENSE](LICENSE).
