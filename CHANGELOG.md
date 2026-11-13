# Changelog

## 0.1.0 - 2026-11-13

First release.

- Success-by-budget curve as the cumulative incidence of success, with runs that ended unsolved on
  their own as a competing event and runs stopped by the limit as censored; pointwise intervals
  (delta method, or a bootstrap over items when items have repeated runs); the plain share beside
  it.
- `at`: the score under a smaller limit by truncation, bounds when runs were stopped earlier, and
  a labelled extrapolation beyond the observed range.
- `compare`: difference at equal budgets with a paired bootstrap over items, a log-rank test and
  the budgets at which the lead changes.
- `marginal`: successes gained per additional unit of limit and the budget from which the gain
  stays under a threshold.
- `cost`: budget spent per solved item, failures included.
- Chart of the curves (optional matplotlib extra).
- Readers for CSV and JSON lines; Markdown and JSON output.
