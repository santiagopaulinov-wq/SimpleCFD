# Numerical Methods

This document describes the numerical methods that are actually implemented in
SimpleCFD. It does not describe general CFD capabilities that are outside the
current codebase.

## 1D Staggered Finite Volume Formulation

The registered solver cases use a one-dimensional staggered mesh:

```text
p0    p1    p2    p3
  u0    u1    u2
```

Pressure and pressure correction live on pressure nodes. Velocity lives on
staggered velocity control volumes between pressure nodes. `Geometry` stores
pressure-node areas, velocity-node areas, and grid spacing. The code supports
uniform scalar `dx` or nonuniform spacing with one interval per pressure-node
pair.

The main momentum equation is assembled in coefficient form:

```text
aP uP = aW uW + aE uE + Su
```

For internal velocity nodes, `MomentumAssembler` computes:

- west and east mass fluxes from neighboring staggered velocities;
- convection coefficients from the selected scheme;
- pressure source `(p_left - p_right) * velocity_area`;
- pressure-correction coefficient `d = A / aP`.

The inlet and outlet velocity nodes are handled by boundary-condition objects
rather than by the generic internal-node formula.

## Boundary Conditions

The implemented 1D solver boundary objects are:

- `InletStagnationPressure`: applies the Versteeg-style stagnation-pressure
  inlet relation with inlet-plane velocity correction.
- `OutletFixedPressure`: applies a fixed outlet static pressure.

There are no wall, periodic, symmetry, or multidimensional boundary-condition
classes in the main SIMPLE solver.

## Convection Schemes

Convection schemes implement `ConvectionScheme`:

```python
interpolate(west_value, east_value, mass_flux)
west_coefficient(mass_flux)
east_coefficient(mass_flux)
```

Implemented schemes:

- `Upwind`: upwind interpolation and positive/negative flux splitting.
- `CentralDifference`: arithmetic interpolation and centered coefficient
  contributions.

These schemes are one-dimensional policies used by the 1D momentum assembler.

## Pressure Correction

`PressureCorrectionAssembler` builds a 1D tridiagonal pressure-correction
system:

```text
aP pP' = aW pW' + aE pE' + b
```

where:

```text
aW = rho * d_w * A_w
aE = rho * d_e * A_e
aP = aW + aE
b  = Fw* - Fe*
```

The pressure-correction boundary rows are identity rows fixing pressure
correction at the first and last pressure nodes.

The same right-hand side vector is used as the continuity residual vector in
the global convergence criterion.

## SIMPLE

`SIMPLECouplingStrategy` performs one pressure-velocity stage per global
iteration:

1. Assemble and solve the momentum equation for starred velocity.
2. Apply velocity under-relaxation.
3. Assemble and solve pressure correction.
4. Correct pressure with pressure relaxation.
5. Correct velocity using `u_i = u_i* + d_i(p_i' - p_{i+1}')`.

## SIMPLEC

`SIMPLECCouplingStrategy` modifies the pressure-correction velocity
coefficient:

```text
d_i = A_i / (aP_i / alpha_u - aW_i - aE_i)
```

It also modifies the momentum-system coefficients for velocity relaxation. If
the SIMPLEC denominator is numerically singular, the code raises a
`ZeroDivisionError` with the affected velocity-node indices.

## SIMPLER

`SIMPLERCouplingStrategy` uses the `AbsolutePressureAssembler` to solve an
absolute-pressure equation based on pseudo-velocities, then predicts momentum,
solves pressure correction, and corrects velocity. The implementation is still
restricted to the same 1D staggered geometry and tridiagonal systems.

## Linear Solver

The only implemented linear solver is `tdma` in `simplecfd.linalg`, a Thomas
algorithm for tridiagonal systems represented by:

```python
LinearSystem(lower, diagonal, upper, rhs)
```

Inputs are copied before elimination so the original `LinearSystem` is not
mutated. Shape validation and zero-pivot checks are explicit.

## Analytic Benchmark Discretizations

Poiseuille and Couette benchmarks are 1D transverse diffusion problems:

- Poiseuille solves a second-order finite-difference system for a quadratic
  velocity profile between stationary plates.
- Couette solves a second-order finite-difference Laplace problem for a linear
  velocity profile between moving plates.

Both use `LinearSystem` and `tdma`, but they do not use the SIMPLE case
registry.
