# Limitations

This document states what SimpleCFD currently does and does not implement.

## Implemented Scope

SimpleCFD currently supports:

- one-dimensional staggered pressure-velocity coupling problems;
- variable-area nozzle/channel-style registered cases;
- SIMPLE, SIMPLEC, and SIMPLER strategies on the 1D formulation;
- upwind and central-difference convection policies;
- 1D momentum source terms for diffusion, linear friction, and localized loss;
- tridiagonal linear solves with TDMA;
- analytic 1D Poiseuille and Couette verification benchmarks;
- CSV, Markdown, and PNG artifact generation;
- method-comparison and mesh-refinement utilities for registered 1D cases.

## Not Implemented

The project does not implement:

- 2D or 3D flow solvers;
- lid-driven cavity simulation;
- unstructured or body-fitted meshes;
- transient time integration;
- turbulence modeling;
- compressible flow;
- energy equation or species transport;
- general sparse matrix assembly;
- iterative sparse linear solvers;
- wall functions;
- inlet/outlet libraries beyond the current 1D inlet stagnation pressure and
  outlet fixed pressure objects;
- pressure boundary-condition families for multidimensional CFD.

## PISO Status

Reports may include a PISO row as a controlled failed row when comparing method
families. PISO is not implemented as a coupling strategy in `COUPLING_STRATEGIES`.
This is intentional so reports can show missing methods without crashing.

## Geometry Restrictions

`Geometry` is one-dimensional. It stores pressure-node areas and staggered
velocity-node areas. It does not store cell volumes in 2D/3D, face normals,
cell connectivity, or vector-valued coordinates.

## Field Restrictions

`Field` contains:

- pressure `p`;
- one velocity component `u`;
- pressure correction `p_prime`.

It does not contain a second velocity component, vector fields, tensor fields,
or cell-centered/face-centered field families for multidimensional problems.

## Solver Restrictions

The SIMPLE-family solver assumes tridiagonal 1D systems. Extending it to 2D
would require new mesh, field, assembly, boundary-condition, and linear-solver
abstractions. The current design intentionally keeps those concerns out of the
1D package core.

## Verification Restrictions

The analytic benchmarks verify 1D transverse diffusion problems and integrated
metrics derived from their profiles. They do not validate the registered
SIMPLE solver against a full Navier-Stokes analytic solution.

The Versteeg 6.2 benchmark is a numerical regression benchmark, not an
independent analytic solution.
