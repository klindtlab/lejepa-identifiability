import LeJEPA.Hermite
import LeJEPA.Uniqueness
import LeJEPA.Approx
import LeJEPA.Dirichlet
import LeJEPA.Planning

/-!
# LeJEPA Identifiability: Formal Verification in Lean 4

Comprehensive formalization of the theoretical results in the paper.

## Files

- **`LeJEPA.Hermite`**: Main theorem (Theorem 4.1) via Hermite
  polynomial spectral decomposition and the correlation bound.

- **`LeJEPA.Uniqueness`**: Converse direction (Theorem 4.2), that
  the Gaussian is the unique latent distribution yielding linear
  identifiability under the Sturm–Liouville operator.

- **`LeJEPA.Approx`**: Approximate identifiability bound
  (Theorem 4.3) with D + (ε + D)² recovery error.

- **`LeJEPA.Dirichlet`**: Alternative proof (Appendix D) via
  Dirichlet energy, AM-GM / Jensen, and Mazur–Ulam.

- **`LeJEPA.Planning`**: Planning equivalence corollary
  (Corollary 4.5): under orthogonal identifiability, expected
  costs, optimal values, and optimal plans coincide between the
  learned latent and the true latent for any O(n)-invariant cost.

## Verification Summary

  | Component                            | Status      |
  |--------------------------------------|-------------|
  | Hermite basis & completeness         | axiomatized |
  | Contraction lemma (ρᵈ decay)         | axiomatized |
  | Mehler's formula                     | axiomatized |
  | Correlation bound ≤ ρ                | VERIFIED    |
  | Equality ⟺ w₁ = 1 (linearity)       | VERIFIED    |
  | Loss lower bound 2(1−ρ)n             | VERIFIED    |
  | Hermite theorem assembly h = Qz      | VERIFIED    |
  | Affine eigenfunction → affine score  | VERIFIED    |
  | Affine score → Gaussian density      | axiomatized |
  | Gaussian → Hermite eigenfunctions    | axiomatized |
  | Gaussian uniqueness biconditional    | VERIFIED    |
  | Polar decomposition                  | axiomatized |
  | Cross-degree Hermite orthogonality   | axiomatized |
  | Spectral gap → W_nl ≤ D              | VERIFIED    |
  | ‖M − Q‖²_F bound                     | VERIFIED    |
  | Pythagorean decomposition            | axiomatized |
  | Bound monotonicity                   | VERIFIED    |
  | Approximate bound assembly           | VERIFIED    |
  | Exact recovery (δ = ε = 0)           | VERIFIED    |
  | AM-GM / Jensen                       | axiomatized |
  | Mazur–Ulam                           | axiomatized |
  | Orthogonal Jacobian → Lipschitz      | VERIFIED    |
  | Bilipschitz → global isometry        | VERIFIED    |
  | Dirichlet theorem assembly h = Qz    | VERIFIED    |
  | Trajectory pushforward (stage/term)  | axiomatized |
  | Per-step stage / terminal equiv.     | VERIFIED    |
  | Planning equivalence (main step)     | VERIFIED    |
  | Minimizer equivalence                | VERIFIED    |
  | Value equivalence                    | VERIFIED    |
-/
