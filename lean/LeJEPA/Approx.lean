import Mathlib

/-!
# Part C — Approximate Identifiability (Proposition 4.3)

  Under approximate alignment (gap δ) and approximate covariance
  (error ε), the recovery error satisfies:

    𝔼[‖h(z) − Qz‖²] ≤ D + (ε + D)²

  where D = δ/(2ρ(1−ρ)) is the alignment gap normalized by the
  spectral gap between Hermite degrees 1 and 2.

  When δ = ε = 0 this recovers Theorem 4.1: h(z) = Qz a.e.

  ## Verification status

  | Component                          | Status      |
  |------------------------------------|-------------|
  | Spectral gap positivity            | VERIFIED    |
  | W_nl ≤ D from gap inequality       | VERIFIED    |
  | Polar decomposition ‖M−Q‖ bound    | hypothesis  |
  | Linear deviation ‖M−Q‖² bound      | VERIFIED    |
  | Pythagorean decomposition           | hypothesis  |
  | Bound monotonicity in W_nl          | VERIFIED    |
  | Full bound assembly                 | VERIFIED    |
  | Exact recovery (δ=ε=0 ⟹ error=0)  | VERIFIED    |
-/

noncomputable section


-- ═══════════════════════════════════════════════════════════════
-- STEP 1: SPECTRAL GAP CONTROLS NONLINEAR ENERGY
-- ═══════════════════════════════════════════════════════════════

/-- The spectral gap ρ(1−ρ) is positive for 0 < ρ < 1. -/
theorem spectral_gap_pos (ρ : ℝ) (hρ0 : 0 < ρ) (hρ1 : ρ < 1) :
    0 < ρ * (1 - ρ) := by
  apply mul_pos hρ0; linarith

/-- 2ρ(1−ρ) is positive. -/
theorem two_spectral_gap_pos (ρ : ℝ) (hρ0 : 0 < ρ) (hρ1 : ρ < 1) :
    0 < 2 * ρ * (1 - ρ) := by
  have : 0 < ρ * (1 - ρ) := spectral_gap_pos ρ hρ0 hρ1
  linarith

/-- **Nonlinear energy bound** (VERIFIED): from the spectral gap
    inequality δ ≥ 2ρ(1−ρ) W_nl, we get W_nl ≤ D = δ/(2ρ(1−ρ)). -/
theorem nonlinear_energy_le_D
    (ρ δ W_nl : ℝ) (hρ0 : 0 < ρ) (hρ1 : ρ < 1)
    (_hδ_nonneg : 0 ≤ δ) (_hW_nonneg : 0 ≤ W_nl)
    (hgap : δ ≥ 2 * ρ * (1 - ρ) * W_nl) :
    W_nl ≤ δ / (2 * ρ * (1 - ρ)) := by
  have hsgap : (0 : ℝ) < 2 * ρ * (1 - ρ) := two_spectral_gap_pos ρ hρ0 hρ1
  rw [le_div_iff₀ hsgap]
  linarith


-- ═══════════════════════════════════════════════════════════════
-- STEP 2: LINEAR PART DEVIATION
-- ═══════════════════════════════════════════════════════════════

/-- **Polar decomposition bound** (tautological restatement): the real
    content (‖M − Q‖_F ≤ ε + W_nl, from polar decomposition, |σᵢ−1| ≤
    |σᵢ²−1|, the covariance split Cov(h) = MMᵀ + N, and the triangle
    inequality) enters `approximate_identifiability` as the hypothesis
    `hpolar`. The statement here is `P → P`, proved by `id`. -/
theorem polar_bound_axiom
    (M_Q_norm ε W_nl : ℝ)
    (_hε : 0 ≤ ε) (_hW : 0 ≤ W_nl) :
    M_Q_norm ≤ ε + W_nl →
    M_Q_norm ≤ ε + W_nl := id

/-- **Linear deviation squared** (VERIFIED): ‖M−Q‖ ≤ ε+W_nl implies
    ‖M−Q‖² ≤ (ε+W_nl)². -/
theorem linear_deviation_sq_bound
    (M_Q_norm ε W_nl : ℝ)
    (hMQ_nonneg : 0 ≤ M_Q_norm)
    (hε : 0 ≤ ε) (hW : 0 ≤ W_nl)
    (hbound : M_Q_norm ≤ ε + W_nl) :
    M_Q_norm ^ 2 ≤ (ε + W_nl) ^ 2 := by
  have h1 : 0 ≤ ε + W_nl := by linarith
  nlinarith [sq_nonneg (ε + W_nl - M_Q_norm)]


-- ═══════════════════════════════════════════════════════════════
-- STEP 3: PYTHAGOREAN DECOMPOSITION
-- ═══════════════════════════════════════════════════════════════

/-- **Pythagorean decomposition** (tautological restatement): the real
    content (the recovery error splits into linear deviation and
    nonlinear energy, via Hermite orthogonality and z ~ N(0,I)) enters
    `approximate_identifiability` as the hypothesis `hpythag`. The
    statement here is `P → P`, proved by `id`. -/
theorem pythagorean_axiom
    (total_error M_Q_norm_sq W_nl : ℝ) :
    total_error = M_Q_norm_sq + W_nl →
    total_error = M_Q_norm_sq + W_nl := id


-- ═══════════════════════════════════════════════════════════════
-- STEP 4: MONOTONICITY
-- ═══════════════════════════════════════════════════════════════

/-- **Monotonicity** (VERIFIED): f(t) = (ε + t)² + t is increasing
    for t ≥ 0. So W_nl ≤ D implies (ε+W_nl)²+W_nl ≤ (ε+D)²+D. -/
theorem bound_monotone (ε W_nl D : ℝ)
    (_hε : 0 ≤ ε) (_hW : 0 ≤ W_nl) (_hD : 0 ≤ D)
    (hle : W_nl ≤ D) :
    (ε + W_nl) ^ 2 + W_nl ≤ (ε + D) ^ 2 + D := by
  have h1 : ε + W_nl ≤ ε + D := by linarith
  nlinarith [sq_nonneg (ε + D - ε - W_nl)]


-- ═══════════════════════════════════════════════════════════════
-- MAIN BOUND ASSEMBLY
-- ═══════════════════════════════════════════════════════════════

/-- **Approximate identifiability** (Proposition 4.3, VERIFIED assembly):

    𝔼[‖h(z) − Qz‖²] ≤ D + (ε + D)²

    where D = δ/(2ρ(1−ρ)). -/
theorem approximate_identifiability
    (ρ δ ε W_nl M_Q_norm total_error : ℝ)
    (hρ0 : 0 < ρ) (hρ1 : ρ < 1)
    (hδ : 0 ≤ δ) (hε : 0 ≤ ε)
    (hW : 0 ≤ W_nl) (hMQ : 0 ≤ M_Q_norm)
    (hgap : δ ≥ 2 * ρ * (1 - ρ) * W_nl)
    (hpolar : M_Q_norm ≤ ε + W_nl)
    (hpythag : total_error = M_Q_norm ^ 2 + W_nl) :
    total_error ≤ δ / (2 * ρ * (1 - ρ))
              + (ε + δ / (2 * ρ * (1 - ρ))) ^ 2 := by
  set D := δ / (2 * ρ * (1 - ρ)) with hD_def
  have hsgap := two_spectral_gap_pos ρ hρ0 hρ1
  have hD_nonneg : 0 ≤ D := div_nonneg hδ (le_of_lt hsgap)
  -- Step 1: W_nl ≤ D
  have hW_le_D : W_nl ≤ D := nonlinear_energy_le_D ρ δ W_nl hρ0 hρ1 hδ hW hgap
  -- Step 4: ‖M−Q‖² ≤ (ε + W_nl)²
  have hMQ_sq : M_Q_norm ^ 2 ≤ (ε + W_nl) ^ 2 :=
    linear_deviation_sq_bound M_Q_norm ε W_nl hMQ hε hW hpolar
  -- Step 3 + 4: total_error ≤ (ε + W_nl)² + W_nl
  have h_inter : total_error ≤ (ε + W_nl) ^ 2 + W_nl := by
    rw [hpythag]; linarith
  -- Step 5: monotonicity
  have h_mono := bound_monotone ε W_nl D hε hW hD_nonneg hW_le_D
  -- Combine
  linarith


-- ═══════════════════════════════════════════════════════════════
-- EXACT RECOVERY AS SPECIAL CASE
-- ═══════════════════════════════════════════════════════════════

/-- **Exact recovery** (VERIFIED): setting δ = ε = 0 gives error = 0,
    recovering Theorem 4.1: h(z) = Qz almost everywhere. -/
theorem exact_recovery_special_case
    (ρ W_nl M_Q_norm total_error : ℝ)
    (hρ0 : 0 < ρ) (hρ1 : ρ < 1)
    (hW : 0 ≤ W_nl) (hMQ : 0 ≤ M_Q_norm)
    (hgap : (0 : ℝ) ≥ 2 * ρ * (1 - ρ) * W_nl)
    (hpolar : M_Q_norm ≤ 0 + W_nl)
    (hpythag : total_error = M_Q_norm ^ 2 + W_nl)
    (_htotal_nonneg : 0 ≤ total_error) :
    total_error = 0 := by
  -- δ = 0 forces W_nl = 0
  have hsgap := two_spectral_gap_pos ρ hρ0 hρ1
  have hW_zero : W_nl = 0 := by nlinarith
  -- W_nl = 0 and ε = 0 force ‖M − Q‖ = 0
  have hMQ_zero : M_Q_norm = 0 := by
    have : M_Q_norm ≤ 0 := by linarith [hpolar, hW_zero]
    linarith
  -- Total error = 0² + 0 = 0
  rw [hpythag, hMQ_zero, hW_zero]; ring


-- ═══════════════════════════════════════════════════════════════
-- BOUND STRUCTURE ANALYSIS
-- ═══════════════════════════════════════════════════════════════

/-- **First-order approximation** (VERIFIED): when ε + D ≤ 1,
    the quadratic term (ε+D)² ≤ ε+D, so the bound ≤ 2D + ε. -/
theorem bound_small_perturbation (ε D : ℝ)
    (hε : 0 ≤ ε) (hD : 0 ≤ D) (hsmall : ε + D ≤ 1) :
    D + (ε + D) ^ 2 ≤ D + ε + D := by
  have h1 : 0 ≤ ε + D := by linarith
  nlinarith [sq_nonneg (1 - (ε + D))]

end
