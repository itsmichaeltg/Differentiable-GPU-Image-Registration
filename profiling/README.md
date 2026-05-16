# Profiling

This directory is intentionally kept light in Git. Nsight output files (`*.ncu-rep`, `*.nsys-rep`) are ignored because they are large and machine-specific.

Recommended run:

```bash
./scripts/profile.sh 1024 20
```

The script builds the project, runs `registration_bench`, and attempts an Nsight Compute capture comparing texture-object warping against the global-memory baseline.
