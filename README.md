# math2osc

TODO:
- audio optimization / cleaning pass
  - prune terms whose freq is > nyquist (for those which are statically provable, e.g., sin(kx))
  - auto-normalization option (save users from having to guess scaling)
- support variable declarations
- add pre-defined macros, e.g., saw(), square(), tri(), etc.