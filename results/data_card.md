# Data Card - UCI Adult

## Source
UCI Machine Learning Repository, "Adult" dataset, version 2,
accessed via sklearn.datasets.fetch_openml(name='adult', version=2).

## License
CC BY 4.0 (per UCI repository metadata).

## Size
48818 rows x 14 columns.
SHA256 of processed sample: `8201959436d65c04f30eb529f9bbf07c33c3c0f7d21ac7fb5ad9d549a8e3849a`

## Known biases
- Class imbalance: `<=50K` is dominant.
- Sex, race, and native-country are imbalanced and correlated with income.
- `workclass` and `occupation` include significant `'?'` missingness.
- The label is a proxy for real-world income and reflects historical
  inequality; a model trained on it can encode those biases.

## Privacy concerns
- Contains self-reported demographic attributes (sex, race, native-country).
- No direct identifiers (names, SSNs) in the released version.
- Still sensitive enough to warrant restricted redistribution and
  anonymization considerations in downstream use.

## Intended use
Educational ML lab: binary classification benchmark.
