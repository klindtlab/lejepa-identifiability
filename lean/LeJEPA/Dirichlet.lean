import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Topology.MetricSpace.Isometry
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Analysis.Normed.Affine.MazurUlam

/-!
# Part B — Alternative Proof via Dirichlet Energy (Appendix C)

  Any C¹ diffeomorphism h : ℝⁿ → ℝⁿ that preserves the standard
  Gaussian measure and minimizes the Dirichlet energy 𝔼[‖Jₕ‖²_F]
  must be a linear orthogonal map h(z) = Uz.

  ## Proof sketch

  The formalized theorem takes the orthogonal-Jacobian conclusion as a
  hypothesis (`horth`) — the energy-minimization argument that would
  derive it (Steps 1–4: reduction to Dirichlet energy, the
  log-determinant lemma, AM-GM, and Jensen) is the measure-theoretic
  part and is not formalized here.

  From the orthogonal Jacobian onward everything is verified:
    Step 5: Orthogonal Jacobian → global isometry →
            Mazur–Ulam → linear                        (VERIFIED)

  ## Verification status

  | Component                        | Status      |
  |----------------------------------|-------------|
  | Mazur–Ulam theorem               | VERIFIED    |
  | Norm-preserving CLM → isometry   | VERIFIED    |
  | Orthogonal Jacobian → Lipschitz  | VERIFIED    |
  | Bilipschitz → global isometry    | VERIFIED    |
  | h(0)=0 → b=0 → linear isometry  | VERIFIED    |
  | Full theorem assembly            | VERIFIED    |
-/

open scoped Matrix BigOperators
open Matrix

noncomputable section

variable {n : ℕ}

/-- The type we work with: ℝⁿ as a Euclidean space. -/
private abbrev E (n : ℕ) := EuclideanSpace ℝ (Fin n)


-- ═══════════════════════════════════════════════════════════════
-- MAZUR–ULAM (proved from Mathlib)
-- ═══════════════════════════════════════════════════════════════

/-- **Mazur–Ulam theorem** (VERIFIED): every surjective isometry of a
    real normed space is affine. An isometry is injective, so together
    with surjectivity `f` is bijective and lifts to an `IsometryEquiv`;
    `IsometryEquiv.toRealLinearIsometryEquiv` (Mathlib's Mazur–Ulam)
    then gives the linear part, with `f x = (f·−f 0) x + f 0`. -/
theorem mazur_ulam
    {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    {f : V → V} (hiso : Isometry f) (hsurj : Function.Surjective f) :
    ∃ (A : V →ₗ[ℝ] V) (b : V), ∀ x, f x = A x + b := by
  let e : V ≃ᵢ V :=
    { toEquiv := Equiv.ofBijective f ⟨hiso.injective, hsurj⟩
      isometry_toFun := hiso }
  refine ⟨e.toRealLinearIsometryEquiv.toLinearEquiv.toLinearMap, f 0, fun x => ?_⟩
  have happ : (e.toRealLinearIsometryEquiv.toLinearEquiv.toLinearMap) x
      = f x - f 0 := e.toRealLinearIsometryEquiv_apply x
  rw [happ]; abel


-- ═══════════════════════════════════════════════════════════════
-- DIFFEOMORPHISM STRUCTURE
-- ═══════════════════════════════════════════════════════════════

/-- A smooth map h : ℝⁿ → ℝⁿ with its Jacobian, modeling a C¹
    diffeomorphism that preserves the standard Gaussian. -/
structure GaussianDiffeo (n : ℕ) where
  /-- The map itself -/
  toFun : E n → E n
  /-- The Jacobian at each point, as a continuous linear map -/
  jacobian : E n → (E n →L[ℝ] E n)
  /-- h is differentiable with the given Jacobian -/
  hasFDeriv : ∀ z, HasFDerivAt toFun (jacobian z) z
  /-- h is a homeomorphism (hence bijective) -/
  isHomeo : (E n) ≃ₜ (E n)
  /-- The homeomorphism agrees with toFun -/
  homeo_eq : ∀ z, isHomeo z = toFun z
  /-- Inverse differentiability from the **inverse function theorem**
      (`HasStrictFDerivAt.toOpenPartialHomeomorph` in
      `Mathlib.Analysis.Calculus.InverseFunctionTheorem.FDeriv`). -/
  hasFDeriv_inv : ∀ y, HasFDerivAt isHomeo.symm
    (ContinuousLinearMap.inverse (jacobian (isHomeo.symm y))) y


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: ORTHOGONAL JACOBIAN → GLOBAL ISOMETRY → LINEAR
-- ═══════════════════════════════════════════════════════════════

/-- A norm-preserving continuous linear map is an isometry. -/
theorem clm_isometry_of_norm_preserving
    (L : E n →L[ℝ] E n)
    (hL : ∀ v, ‖L v‖ = ‖v‖) :
    Isometry L := by
  rw [isometry_iff_dist_eq]
  intro x y
  simp only [dist_eq_norm, ← map_sub L x y]
  exact hL (x - y)

/-- **Mean value theorem** (VERIFIED): orthogonal Jacobian everywhere
    ⟹ h is 1-Lipschitz. By the MVT, ‖h(x)-h(y)‖ ≤ sup ‖Jₕ‖_op · ‖x-y‖,
    and the operator norm of a norm-preserving map is 1. -/
theorem lipschitz_of_orthogonal_jacobian
    (h : GaussianDiffeo n)
    (horth : ∀ z v, ‖h.jacobian z v‖ = ‖v‖) :
    LipschitzWith 1 h.toFun := by
  apply lipschitzWith_of_nnnorm_fderiv_le (𝕜 := ℝ)
  · intro x; exact (h.hasFDeriv x).differentiableAt
  · intro x
    have hfderiv : fderiv ℝ h.toFun x = h.jacobian x :=
      (h.hasFDeriv x).fderiv
    rw [hfderiv, ContinuousLinearMap.opNNNorm_le_iff]
    intro y; simp only [one_mul]
    exact_mod_cast le_of_eq (horth x y)

/-- **Bilipschitz → isometry** (VERIFIED): if both h and h⁻¹ are
    1-Lipschitz, h is a global isometry. Forward Lipschitz gives
    dist(hx,hy) ≤ dist(x,y); applying to h⁻¹ gives ≥. -/
theorem isometry_of_bilipschitz
    (h : GaussianDiffeo n)
    (hlip : LipschitzWith 1 h.toFun)
    (hinvlip : LipschitzWith 1 h.isHomeo.symm) :
    Isometry h.toFun := by
  rw [isometry_iff_dist_eq]
  intro x y
  apply le_antisymm
  · -- Forward: dist(hx, hy) ≤ dist(x, y)
    have hfwd := hlip.dist_le_mul x y
    simp only [NNReal.coe_one, one_mul] at hfwd; exact hfwd
  · -- Backward: apply Lipschitz to h⁻¹
    have hbwd := hinvlip.dist_le_mul (h.toFun x) (h.toFun y)
    simp only [NNReal.coe_one, one_mul] at hbwd
    have hx : h.isHomeo.symm (h.toFun x) = x := by
      rw [← h.homeo_eq]; exact h.isHomeo.symm_apply_apply x
    have hy : h.isHomeo.symm (h.toFun y) = y := by
      rw [← h.homeo_eq]; exact h.isHomeo.symm_apply_apply y
    rw [hx, hy] at hbwd; exact hbwd


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: MAIN THEOREM (APPENDIX C)
-- ═══════════════════════════════════════════════════════════════

/-- **LeJEPA identifiability via Dirichlet energy** (VERIFIED):

    C¹ diffeomorphism + Gaussian-preserving + orthogonal Jacobian
    ⟹ h(z) = Uz for a linear isometry U ∈ O(n).

    Verified chain:
    1. Orth. Jacobian → h is 1-Lipschitz    (MVT)
    2. Orth. inverse  → h⁻¹ is 1-Lipschitz  (IFT + MVT)
    3. Bilipschitz → global isometry
    4. Mazur–Ulam → h is affine: h(z) = Az + b
    5. h(0) = 0 → b = 0
    6. A preserves norms → A is a LinearIsometry -/
theorem dirichlet_identifiability
    (h : GaussianDiffeo n)
    (horth : ∀ z v, ‖h.jacobian z v‖ = ‖v‖)
    (horth_inv : ∀ z v,
      ‖(ContinuousLinearMap.inverse (h.jacobian z)) v‖ = ‖v‖)
    (hmean : h.toFun 0 = 0) :
    ∃ (U : E n →ₗᵢ[ℝ] E n), ∀ z, h.toFun z = U z := by
  -- Step 1: h is 1-Lipschitz
  have hlip := lipschitz_of_orthogonal_jacobian h horth
  -- Step 2: h⁻¹ is 1-Lipschitz (IFT gives derivative = J⁻¹, also orth.)
  have hinvlip : LipschitzWith 1 h.isHomeo.symm := by
    apply lipschitzWith_of_nnnorm_fderiv_le (𝕜 := ℝ)
    · intro x; exact (h.hasFDeriv_inv x).differentiableAt
    · intro x
      have hfderiv : fderiv ℝ h.isHomeo.symm x =
        (h.jacobian (h.isHomeo.symm x)).inverse :=
        (h.hasFDeriv_inv x).fderiv
      rw [hfderiv, ContinuousLinearMap.opNNNorm_le_iff]
      intro y; simp only [one_mul]
      exact_mod_cast le_of_eq (horth_inv (h.isHomeo.symm x) y)
  -- Step 3: h is a global isometry
  have hiso := isometry_of_bilipschitz h hlip hinvlip
  -- Step 4: Mazur–Ulam → h(z) = Az + b
  have hsurj : Function.Surjective h.toFun := by
    intro y
    exact ⟨h.isHomeo.symm y,
      by rw [← h.homeo_eq]; exact h.isHomeo.apply_symm_apply y⟩
  obtain ⟨A, b, hab⟩ := mazur_ulam hiso hsurj
  -- Step 5: b = 0 from h(0) = 0
  have hb : b = 0 := by
    have h0 := hab 0; simp [map_zero] at h0
    rw [hmean] at h0; exact h0.symm
  -- h(z) = Az for all z
  have hab' : ∀ z, h.toFun z = A z := by
    intro z; have := hab z; rw [hb, add_zero] at this; exact this
  -- Step 6: A preserves norms → LinearIsometry
  have hA_norm : ∀ v, ‖A v‖ = ‖v‖ := by
    intro v
    have hv := hiso.dist_eq v 0
    simp [dist_eq_norm] at hv
    rw [hab' v, hab' 0, map_zero] at hv
    simpa using hv
  exact ⟨⟨A, hA_norm⟩, hab'⟩

end
