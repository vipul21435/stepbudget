# stepbudget

Estimates an agent's success rate as a function of the step (or token, cost, time) limit it is
given, treating runs that were stopped by the limit as censored.

A run stopped by the limit says only that solving the item would have needed more than the limit.
Dropping such runs or counting them as failures both bias the curve when runs were made under
different limits.

## Status

Early version. What works so far:

- reading one run per row (item, system, outcome, budget used, exit reason) from CSV or JSON lines;
- the success-by-budget curve as the Aalen-Johansen cumulative incidence of success, with giving
  up as a competing event and limit-reached runs as censored, with pointwise intervals;
- the plain share of runs solved within each budget beside it, and a statement of where the two
  part ways;
- items with repeated runs are weighted once and their intervals come from a bootstrap over items;
- `stepbudget curve` with Markdown and JSON output.

Not there yet: truncation to a smaller limit, comparison of two systems, marginal return, cost of
a success, a chart, and a run on public data.

## Install

    pip install git+https://github.com/vipul21435/stepbudget

## Example

`examples/runs.csv` holds synthetic runs of two made-up systems (written by
`examples/make_runs.py`).

    stepbudget curve examples/runs.csv --system alpha --budgets 20 40 60 80

```
## alpha

120 runs on 120 items: 75 succeeded, 26 gave up, 19 were stopped by the limit.

| budget | solved within budget | 95 % interval | plain share | still running |
| --- | --- | --- | --- | --- |
| 20 | 23.3 % | 16.2 to 31.2 % | 23.3 % | 65.8 % |
| 40 | 54.2 % | 44.8 to 62.6 % | 54.2 % | 25.8 % |
| 60 | 66.4 % | 56.3 to 74.7 % | 61.7 % | 12.2 % |
| 80 | 67.8 % | 57.7 to 75.9 % | 62.5 % | 9.5 % |
```

## Tests

    uv run --extra dev pytest -q

The estimators are checked against hand-worked tables and against the lifelines package.

## Licence

MIT.

Built by Vipul Raj Jha · https://vipul21435.github.io
