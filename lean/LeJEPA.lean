import LeJEPA.ThmHermite
import LeJEPA.ThmDirichlet
import LeJEPA.PropApprox

/-!
# LeJEPA Identifiability: Formal Verification in Lean 4

Comprehensive formalization of the three main theoretical results.

## Files

- **`LeJEPA.ThmHermite`**: Main theorem (Theorem 4.1) via Hermite
  polynomial spectral decomposition and the correlation bound.

- **`LeJEPA.ThmDirichlet`**: Alternative proof (Appendix C) via
  Dirichlet energy, the mean value theorem, and Mazur–Ulam.

- **`LeJEPA.PropApprox`**: Approximate identifiability bound
  (Proposition 4.3) with D + (ε + D)² recovery error.

## Verification Summary

  | Component                        | Status      |
  |----------------------------------|-------------|
  | Hermite basis & completeness     | axiomatized |
  | Contraction lemma (ρᵈ decay)     | axiomatized |
  | Mehler's formula                 | axiomatized |
  | Correlation bound ≤ ρ            | VERIFIED    |
  | Equality ⟺ w₁ = 1 (linearity)  | VERIFIED    |
  | Loss lower bound 2(1−ρ)n        | VERIFIED    |
  | Hermite theorem assembly h=Uz   | VERIFIED    |
  | AM-GM / Jensen                   | axiomatized |
  | Mazur–Ulam                       | axiomatized |
  | Orthogonal Jacobian → Lipschitz  | VERIFIED    |
  | Bilipschitz → global isometry    | VERIFIED    |
  | Dirichlet theorem assembly h=Uz | VERIFIED    |
  | Polar decomposition              | axiomatized |
  | Cross-degree Hermite orthogonality| axiomatized |
  | Spectral gap → W_nl ≤ D         | VERIFIED    |
  | ‖M − Q‖²_F bound                | VERIFIED    |
  | Pythagorean decomposition        | axiomatized |
  | Bound monotonicity               | VERIFIED    |
  | Approximate bound assembly       | VERIFIED    |
  | Exact recovery (δ=ε=0)          | VERIFIED    |
-/
