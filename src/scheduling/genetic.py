"""Flight rescheduling via a genetic algorithm -- STUB, not implemented.

Deferred per the project brief: Phase 8 is out of scope for this build.
This module exists only to reserve the interface; it contains no working
optimizer.

Intended approach, to be implemented in a future phase:

- Scope: one day of flights at a time (`data[data['FL_DATE'] == <day>]`),
  since delay propagation and airport/runway contention are local to a
  single day's schedule.
- Chromosome: a candidate schedule is a vector of adjusted CRS_DEP_TIME
  values, one per flight in that day, each perturbed from its original
  scheduled departure time.
- Population: POPULATION_SIZE candidate schedules, randomly initialized
  around the day's actual departure times.
- Selection: tournament selection -- sample a small subset of the
  population, keep the fittest as a parent.
- Crossover: single-point crossover between two parent schedules to produce
  offspring (swap the departure-time vector at a random cut point).
- Mutation: perturb a departure time by up to +/-30 minutes, applied with
  some per-gene mutation probability.
- Fitness: total predicted departure delay across the day's schedule (lower
  is better) -- evaluated using one of this project's trained delay models
  (Phase 5/6) to score a candidate schedule's flights, rather than the
  ground-truth delay, since the whole point is to explore schedules that
  were never actually flown.
- Termination: fixed generation count or fitness-plateau early stopping,
  returning the best schedule found.

Notes for the future implementer, from reading reference/FINAL.ipynb's GA
section (cells 111-118) -- two design flaws to fix rather than port:
  - Its fitness function reassigns CRS_DEP_TIME to the candidate schedule
    but then sums the *original, unmodified* DEP_DELAY column -- DEP_DELAY
    is never recomputed from the perturbed schedule, so every candidate in
    every generation scores identically and the GA optimizes over a flat
    fitness landscape. A real implementation must score candidates with a
    trained delay model (Phase 5/6), not a stale historical column.
  - It perturbs CRS_DEP_TIME (an HHMM-encoded integer, e.g. 1345) with
    plain integer arithmetic (`+ np.random.randint(-30, 30)` at init,
    `+= np.random.randint(-63, 63)` at mutation) instead of real
    minute-of-day arithmetic, so a perturbed value can land outside valid
    HHMM range (e.g. 1345 + 30 = 1375, not a real time) mid-run; it only
    normalizes the final best schedule (`adjust_time_format`), not every
    intermediate candidate. A real implementation should perturb in
    minutes-past-midnight (as src.features.build.convert_hhmm_to_mins
    already does) and convert back to HHMM once, at the end.

TODO(phase-8): implement population initialization, tournament selection,
single-point crossover, mutation, fitness evaluation against a trained
model, and the generational loop. Wire up config-driven paths and a CLI
entry point (`python -m src.scheduling.genetic`) once implemented.
"""

from __future__ import annotations

import pandas as pd


def run_rescheduler(day_flights: pd.DataFrame, population_size: int, n_generations: int) -> pd.DataFrame:
    """Run the genetic algorithm over one day's flights and return the best schedule found.

    Not implemented -- Phase 8 is deferred. See the module docstring for the
    intended approach.
    """
    raise NotImplementedError("Flight rescheduling (Phase 8) is deferred and not yet implemented.")
