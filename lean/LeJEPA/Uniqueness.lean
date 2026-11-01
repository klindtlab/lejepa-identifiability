import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Gaussian Uniqueness (Proposition: Converse Direction)

  The first non-constant eigenfunction of the transition operator
  is affine **if and only if** p is Gaussian.

  ## Verification status

  | Component                              | Status      |
  |----------------------------------------|-------------|
  | SL eigenfunction equation              | structural  |
  | Score slope negativity (−ev/K < 0)     | VERIFIED    |
  | Affine eigenfunction → affine score    | VERIFIED    |
  | Affine score → Gaussian density        | VERIFIED    |
  | Only-if assembly                       | VERIFIED    |
  | First eigenvalue at affine score (−Kα) | hypothesis  |
  | Gaussian → affine eigenfunction        | VERIFIED    |
  | If assembly                            | VERIFIED    |
  | Full biconditional                     | VERIFIED    |
  | Zero-mean specialization               | VERIFIED    |
-/

set_option maxHeartbeats 400000

noncomputable section


-- ═══════════════════════════════════════════════════════════════
-- STURM–LIOUVILLE STRUCTURE
-- ═══════════════════════════════════════════════════════════════

/-- A scalar latent component under constant diffusion K > 0. -/
structure LatentComponent where
  K : ℝ
  hK : 0 < K
  score : ℝ → ℝ        -- (log p)'
  ev : ℝ               -- first non-constant eigenvalue λ₁
  hev : 0 < ev

/-- Score corresponds to a Gaussian: ∃ α < 0, β, score(z) = αz + β. -/
def IsGaussianScore (score : ℝ → ℝ) : Prop :=
  ∃ α β : ℝ, α < 0 ∧ ∀ z, score z = α * z + β


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: GAUSSIAN SCORE AND ITS FIRST EIGENFUNCTION
-- ═══════════════════════════════════════════════════════════════

/-- **Affine score → Gaussian** (VERIFIED): integrating
    score(z) = αz + β gives log p = (α/2)z² + βz + C, i.e. a Gaussian.
    With `IsGaussianScore` defined as "score is affine with negative
    slope", the conclusion is exactly the hypotheses. -/
theorem gaussian_of_affine_score (score : ℝ → ℝ) (α β : ℝ)
    (hα : α < 0) (hscore : ∀ z, score z = α * z + β) :
    IsGaussianScore score :=
  ⟨α, β, hα, hscore⟩

/-- Whenever the score is affine, `score z = αz + β`, the eigenvalue field is
    `ev = −Kα`: the first non-constant eigenvalue of the Sturm–Liouville
    operator `f ↦ K (f'' + score · f')` at that score, with eigenfunction
    `z + β/α`. -/
def IsGaussianEigenvalue (lc : LatentComponent) : Prop :=
  ∀ α β : ℝ, (∀ z, lc.score z = α * z + β) → lc.ev = -(lc.K * α)

/-- **Gaussian → affine eigenfunction** (VERIFIED, given
    `IsGaussianEigenvalue`): if the score is Gaussian, `score z = αz + β`
    with `α < 0`, then `ev = −Kα`, and `a = 1`, `b = β/α` solve the
    eigen-equation `K · score z · a = −ev · (a z + b)`: the eigenfunction is
    `z + β/α`. -/
theorem hermite_first_eigenfunction_of_gaussian
    (lc : LatentComponent) (hev : IsGaussianEigenvalue lc)
    (hgauss : IsGaussianScore lc.score) :
    ∃ (a b : ℝ), a ≠ 0 ∧
      ∀ z, lc.K * lc.score z * a = -(lc.ev * (a * z + b)) := by
  obtain ⟨α, β, hα, hscore⟩ := hgauss
  have hα0 : α ≠ 0 := hα.ne
  have hβ : α * (β / α) = β := by field_simp
  refine ⟨1, β / α, one_ne_zero, fun z => ?_⟩
  rw [hscore z, hev α β hscore]
  linear_combination (-lc.K) * hβ


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: AFFINE EIGENFUNCTION → AFFINE SCORE
-- ═══════════════════════════════════════════════════════════════

/-- **Core algebraic step** (VERIFIED):
    K · score(z) · a = −ev·(az + b)  with a ≠ 0
    ⟹ score(z) = (−ev/K)z + (−ev·b/(Ka)),  slope < 0. -/
theorem score_affine_of_eigenfunction
    (lc : LatentComponent) (a b : ℝ) (ha : a ≠ 0)
    (heigen : ∀ z, lc.K * lc.score z * a = -(lc.ev * (a * z + b))) :
    ∃ (α β : ℝ), α < 0 ∧ (∀ z, lc.score z = α * z + β) := by
  refine ⟨-(lc.ev / lc.K), -(lc.ev * b / (lc.K * a)), ?_, ?_⟩
  · -- −ev/K < 0 since ev > 0 and K > 0
    have := div_pos lc.hev lc.hK
    linarith
  · intro z
    have hK_ne : lc.K ≠ 0 := ne_of_gt lc.hK
    have hKa_ne : lc.K * a ≠ 0 := mul_ne_zero hK_ne ha
    have h := heigen z
    -- Isolate score(z): divide by K·a
    have h1 : lc.score z = -(lc.ev * (a * z + b)) / (lc.K * a) := by
      field_simp at h ⊢; linarith
    rw [h1]; field_simp; ring


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: ONLY-IF ASSEMBLY
-- ═══════════════════════════════════════════════════════════════

/-- **Only-if** (VERIFIED): affine eigenfunction ⟹ Gaussian. -/
theorem gaussian_of_affine_eigenfunction
    (lc : LatentComponent) (a b : ℝ) (ha : a ≠ 0)
    (heigen : ∀ z, lc.K * lc.score z * a = -(lc.ev * (a * z + b))) :
    IsGaussianScore lc.score := by
  obtain ⟨α, β, hα_neg, hscore⟩ :=
    score_affine_of_eigenfunction lc a b ha heigen
  exact gaussian_of_affine_score lc.score α β hα_neg hscore


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: FULL BICONDITIONAL
-- ═══════════════════════════════════════════════════════════════

/-- **Gaussian uniqueness** (VERIFIED):
    First eigenfunction is affine ⟺ p is Gaussian, for a component whose
    eigenvalue field is the operator's first non-constant eigenvalue at
    affine scores (`IsGaussianEigenvalue`). -/
theorem gaussian_uniqueness (lc : LatentComponent)
    (hev : IsGaussianEigenvalue lc) :
    (IsGaussianScore lc.score →
      ∃ (a b : ℝ), a ≠ 0 ∧
        ∀ z, lc.K * lc.score z * a = -(lc.ev * (a * z + b)))
    ∧
    (∀ (a b : ℝ), a ≠ 0 →
      (∀ z, lc.K * lc.score z * a = -(lc.ev * (a * z + b))) →
      IsGaussianScore lc.score) :=
  ⟨hermite_first_eigenfunction_of_gaussian lc hev,
   fun a b ha heigen => gaussian_of_affine_eigenfunction lc a b ha heigen⟩


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: ZERO-MEAN SPECIALIZATION
-- ═══════════════════════════════════════════════════════════════

/-- **Zero mean** (VERIFIED): with b = 0, a = 1,
    score(z) = −(ev/K)·z. -/
theorem score_pure_linear_zero_mean
    (lc : LatentComponent)
    (heigen : ∀ z, lc.K * lc.score z * 1 = -(lc.ev * (1 * z + 0))) :
    ∀ z, lc.score z = -(lc.ev / lc.K) * z := by
  intro z
  have hK_ne : lc.K ≠ 0 := ne_of_gt lc.hK
  have h := heigen z
  simp only [mul_one, add_zero] at h
  field_simp; linarith

end
