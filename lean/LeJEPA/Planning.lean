import Mathlib

/-!
# Part D — Planning Equivalence (Corollary)

  Let h(z) = Qz with Q ∈ O(n) be the encoder at the optimum of Theorem 4.1.
  For any finite-horizon optimal control problem whose stage and terminal
  costs are O(n)-invariant in the state argument, the optimal value function
  and the set of optimal action sequences agree between the learned latent
  and the true latent.

  The proof reduces — via the rotation-invariance hypothesis and the
  pushforward property of expected costs — to the trivial fact that pointwise
  equal real-valued functions share minimizers.

  The covariant form transports the cost along Q instead of asking a fixed
  cost to be invariant. It covers goal-reaching, terminal cost ‖z − z*‖²,
  with state and goal rotating together.

  ## Verification status

  | Component                             | Status      |
  |---------------------------------------|-------------|
  | ControlProblem structure              | structural  |
  | Orthogonal invariance definition      | structural  |
  | ExpectedCosts abstraction             | structural  |
  | Total-cost definition                 | structural  |
  | Trajectory pushforward (stage)        | hypothesis  |
  | Trajectory pushforward (terminal)     | hypothesis  |
  | Per-step stage cost equivalence       | VERIFIED    |
  | Terminal cost equivalence             | VERIFIED    |
  | Total cost equivalence (main step)    | VERIFIED    |
  | Minimizer equivalence (plan agreement)| VERIFIED    |
  | Value equivalence                     | VERIFIED    |
  | Covariant form (transported cost)     | VERIFIED    |
  | Goal-reaching                         | VERIFIED    |
-/

set_option maxHeartbeats 400000

open scoped BigOperators

noncomputable section

abbrev Latent (n : ℕ) := Fin n → ℝ
abbrev Plan (Action : Type*) (T : ℕ) := Fin T → Action


-- ═══════════════════════════════════════════════════════════════
-- STRUCTURE: CONTROL PROBLEM AND ROTATION INVARIANCE
-- ═══════════════════════════════════════════════════════════════

/-- A finite-horizon optimal control problem with stage cost ℓ(z,a) and
    terminal cost ℓ_T(z). -/
structure ControlProblem (n : ℕ) (Action : Type*) where
  stage_cost : Latent n → Action → ℝ
  terminal_cost : Latent n → ℝ

/-- The costs of the control problem are O(n)-invariant in the state argument
    under a map Q:  ℓ(Q z, a) = ℓ(z, a) for all z, a, and ℓ_T(Q z) = ℓ_T(z)
    for all z. In the corollary, Q is the orthogonal recovery matrix from
    Theorem 4.1; the definition does not itself require Q to be linear or
    orthogonal — only the invariance property is used. -/
def IsOrthogonalInvariant {n : ℕ} {Action : Type*}
    (cp : ControlProblem n Action) (Q : Latent n → Latent n) : Prop :=
  (∀ z a, cp.stage_cost (Q z) a = cp.stage_cost z a) ∧
  (∀ z, cp.terminal_cost (Q z) = cp.terminal_cost z)


-- ═══════════════════════════════════════════════════════════════
-- STRUCTURE: EXPECTED COSTS UNDER SOME DYNAMICS
-- ═══════════════════════════════════════════════════════════════

/-- Expected costs along a trajectory under a specific (stochastic) dynamics.

    `stage_exp a z₀ t c` is the expected value of `c(z_t, a_t)` at time `t`
    along the trajectory starting from `z₀` and following the action sequence
    `a`. `term_exp a z₀ c` is the expected value of `c(z_T)` at the final
    time. Parameterizing over the cost function `c` lets the same dynamics
    object be reused for different costs, and makes the pushforward relation
    (below) statable without explicit measure theory. -/
structure ExpectedCosts (n : ℕ) (Action : Type*) (T : ℕ) where
  stage_exp :
    Plan Action T → Latent n → Fin T → (Latent n → Action → ℝ) → ℝ
  term_exp :
    Plan Action T → Latent n → (Latent n → ℝ) → ℝ


-- ═══════════════════════════════════════════════════════════════
-- TOTAL EXPECTED COST
-- ═══════════════════════════════════════════════════════════════

/-- Total expected cost for a plan `a` from initial state `z₀`: the sum of
    per-step stage costs plus the terminal cost. -/
def totalCost {n : ℕ} {Action : Type*} {T : ℕ}
    (cp : ControlProblem n Action) (E : ExpectedCosts n Action T)
    (a : Plan Action T) (z₀ : Latent n) : ℝ :=
  (∑ t : Fin T, E.stage_exp a z₀ t cp.stage_cost)
    + E.term_exp a z₀ cp.terminal_cost


-- ═══════════════════════════════════════════════════════════════
-- HYPOTHESIS: TRAJECTORY PUSHFORWARD
-- ═══════════════════════════════════════════════════════════════

/-- **Trajectory pushforward**: `E_hat` is the pushforward of `E` under `Q`.
    At each time `t`, the expected value of any cost `c` under `E_hat`
    starting from `Q z` equals the expected value under `E` starting from `z`
    of the pre-composed cost `c ∘ (Q × id)`; the same holds at the terminal
    time.

    Mathematically this is the content of "the joint law of (ẑ_0, …, ẑ_T)
    under the pushforward dynamics starting from ẑ_0 = Q z equals the joint
    law of (Q z_0, …, Q z_T) under the original dynamics starting from
    z_0 = z", restricted to per-time-step marginals and evaluated against
    arbitrary test functions. -/
def IsPushforwardOf {n : ℕ} {Action : Type*} {T : ℕ}
    (E_hat E : ExpectedCosts n Action T) (Q : Latent n → Latent n) : Prop :=
  (∀ (a : Plan Action T) (z : Latent n) (t : Fin T) (c : Latent n → Action → ℝ),
      E_hat.stage_exp a (Q z) t c = E.stage_exp a z t (fun z' act => c (Q z') act)) ∧
  (∀ (a : Plan Action T) (z : Latent n) (c : Latent n → ℝ),
      E_hat.term_exp a (Q z) c = E.term_exp a z (fun z' => c (Q z')))


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: PER-STEP COST EQUIVALENCE
-- ═══════════════════════════════════════════════════════════════

/-- **Stage-cost equivalence** (VERIFIED): the per-step expected stage cost
    at `Q z` under the pushforward dynamics equals the per-step expected
    stage cost at `z` under the original dynamics, when the stage cost is
    O(n)-invariant. This is the point where orthogonal invariance of the
    cost meets trajectory pushforward (both hypotheses). -/
theorem stage_cost_equiv
    {n : ℕ} {Action : Type*} {T : ℕ}
    (cp : ControlProblem n Action) (Q : Latent n → Latent n)
    (E_hat E : ExpectedCosts n Action T)
    (hpush : IsPushforwardOf E_hat E Q)
    (hinv : IsOrthogonalInvariant cp Q)
    (a : Plan Action T) (z : Latent n) (t : Fin T) :
    E_hat.stage_exp a (Q z) t cp.stage_cost
      = E.stage_exp a z t cp.stage_cost := by
  rw [hpush.1 a z t cp.stage_cost]
  have hfun : (fun z' act => cp.stage_cost (Q z') act) = cp.stage_cost := by
    funext z'
    funext act
    exact hinv.1 z' act
  rw [hfun]

/-- **Terminal-cost equivalence** (VERIFIED). -/
theorem terminal_cost_equiv
    {n : ℕ} {Action : Type*} {T : ℕ}
    (cp : ControlProblem n Action) (Q : Latent n → Latent n)
    (E_hat E : ExpectedCosts n Action T)
    (hpush : IsPushforwardOf E_hat E Q)
    (hinv : IsOrthogonalInvariant cp Q)
    (a : Plan Action T) (z : Latent n) :
    E_hat.term_exp a (Q z) cp.terminal_cost
      = E.term_exp a z cp.terminal_cost := by
  rw [hpush.2 a z cp.terminal_cost]
  have hfun : (fun z' => cp.terminal_cost (Q z')) = cp.terminal_cost := by
    funext z'
    exact hinv.2 z'
  rw [hfun]


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: TOTAL COST EQUIVALENCE (PLANNING EQUIVALENCE)
-- ═══════════════════════════════════════════════════════════════

/-- **Planning equivalence** (VERIFIED, main step): for any action sequence,
    the total expected cost under the pushforward dynamics at `Q z₀` equals
    the total expected cost under the original dynamics at `z₀`.

    This is the central computational content of the corollary; everything
    that follows (value and minimizer equivalence) is a consequence. -/
theorem planning_equivalence
    {n : ℕ} {Action : Type*} {T : ℕ}
    (cp : ControlProblem n Action) (Q : Latent n → Latent n)
    (E_hat E : ExpectedCosts n Action T)
    (hpush : IsPushforwardOf E_hat E Q)
    (hinv : IsOrthogonalInvariant cp Q)
    (a : Plan Action T) (z : Latent n) :
    totalCost cp E_hat a (Q z) = totalCost cp E a z := by
  unfold totalCost
  have hstage :
      (∑ t : Fin T, E_hat.stage_exp a (Q z) t cp.stage_cost)
        = ∑ t : Fin T, E.stage_exp a z t cp.stage_cost := by
    apply Finset.sum_congr rfl
    intro t _
    exact stage_cost_equiv cp Q E_hat E hpush hinv a z t
  have hterm :
      E_hat.term_exp a (Q z) cp.terminal_cost
        = E.term_exp a z cp.terminal_cost :=
    terminal_cost_equiv cp Q E_hat E hpush hinv a z
  rw [hstage, hterm]


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: MINIMIZER AND VALUE EQUIVALENCE
-- ═══════════════════════════════════════════════════════════════

/-- **Minimizer equivalence** (VERIFIED): an action sequence minimizes the
    expected cost under the pushforward dynamics at `Q z` iff it minimizes
    the expected cost under the original dynamics at `z`.

    Consequence: the optimal plan is the same whether it is computed in the
    learned latent or the true latent. -/
theorem minimizer_equivalence
    {n : ℕ} {Action : Type*} {T : ℕ}
    (cp : ControlProblem n Action) (Q : Latent n → Latent n)
    (E_hat E : ExpectedCosts n Action T)
    (hpush : IsPushforwardOf E_hat E Q)
    (hinv : IsOrthogonalInvariant cp Q)
    (a : Plan Action T) (z : Latent n) :
    (∀ a', totalCost cp E_hat a (Q z) ≤ totalCost cp E_hat a' (Q z)) ↔
    (∀ a', totalCost cp E a z ≤ totalCost cp E a' z) := by
  have h : ∀ a', totalCost cp E_hat a' (Q z) = totalCost cp E a' z :=
    fun a' => planning_equivalence cp Q E_hat E hpush hinv a' z
  constructor
  · intro hmin a'
    have ha := h a
    have ha' := h a'
    have := hmin a'
    linarith
  · intro hmin a'
    have ha := h a
    have ha' := h a'
    have := hmin a'
    linarith

/-- **Value equivalence** (VERIFIED): if `a` achieves total cost `V` under
    the original dynamics at `z`, it achieves the same `V` under the
    pushforward dynamics at `Q z`. Combined with `minimizer_equivalence`,
    this gives the corollary's `V̂*(Q z) = V*(z)` statement. -/
theorem value_equivalence
    {n : ℕ} {Action : Type*} {T : ℕ}
    (cp : ControlProblem n Action) (Q : Latent n → Latent n)
    (E_hat E : ExpectedCosts n Action T)
    (hpush : IsPushforwardOf E_hat E Q)
    (hinv : IsOrthogonalInvariant cp Q)
    (a : Plan Action T) (z : Latent n) (V : ℝ)
    (hV : totalCost cp E a z = V) :
    totalCost cp E_hat a (Q z) = V := by
  rw [planning_equivalence cp Q E_hat E hpush hinv a z, hV]


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: COVARIANT FORM (TRANSPORTED COST)
-- ═══════════════════════════════════════════════════════════════

/-- `cpHat` is the control problem `cp` transported along `Q`: the cost of
    `Q z` under `cpHat` is the cost of `z` under `cp`. The invariance
    hypothesis `IsOrthogonalInvariant cp Q` is the case `cpHat = cp`. -/
def IsTransportOf {n : ℕ} {Action : Type*}
    (cpHat cp : ControlProblem n Action) (Q : Latent n → Latent n) : Prop :=
  (∀ z a, cpHat.stage_cost (Q z) a = cp.stage_cost z a) ∧
  (∀ z, cpHat.terminal_cost (Q z) = cp.terminal_cost z)

theorem isTransportOf_self_iff {n : ℕ} {Action : Type*}
    (cp : ControlProblem n Action) (Q : Latent n → Latent n) :
    IsTransportOf cp cp Q ↔ IsOrthogonalInvariant cp Q := Iff.rfl

/-- **Planning equivalence, covariant form** (VERIFIED): with the cost
    transported along `Q`, the total expected cost under the pushforward
    dynamics at `Q z` equals the total expected cost under the original
    dynamics at `z`. No fixed cost is required to be invariant;
    `planning_equivalence` is the case `cpHat = cp`. -/
theorem planning_equivalence_transport
    {n : ℕ} {Action : Type*} {T : ℕ}
    (cpHat cp : ControlProblem n Action) (Q : Latent n → Latent n)
    (E_hat E : ExpectedCosts n Action T)
    (hpush : IsPushforwardOf E_hat E Q)
    (htr : IsTransportOf cpHat cp Q)
    (a : Plan Action T) (z : Latent n) :
    totalCost cpHat E_hat a (Q z) = totalCost cp E a z := by
  simp only [totalCost, hpush.1, hpush.2, htr.1, htr.2]

/-- **Minimizer equivalence, covariant form** (VERIFIED): an action sequence
    minimizes the transported problem under the pushforward dynamics at `Q z`
    iff it minimizes the original problem under the original dynamics at `z`. -/
theorem minimizer_equivalence_transport
    {n : ℕ} {Action : Type*} {T : ℕ}
    (cpHat cp : ControlProblem n Action) (Q : Latent n → Latent n)
    (E_hat E : ExpectedCosts n Action T)
    (hpush : IsPushforwardOf E_hat E Q)
    (htr : IsTransportOf cpHat cp Q)
    (a : Plan Action T) (z : Latent n) :
    (∀ a', totalCost cpHat E_hat a (Q z) ≤ totalCost cpHat E_hat a' (Q z)) ↔
    (∀ a', totalCost cp E a z ≤ totalCost cp E a' z) := by
  simp only [planning_equivalence_transport cpHat cp Q E_hat E hpush htr]


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: GOAL-REACHING
-- ═══════════════════════════════════════════════════════════════

/-- Euclidean distance on the latent. `Latent n` carries Mathlib's sup
    metric, so the Euclidean one is taken in `EuclideanSpace ℝ (Fin n)`. -/
def eucDist {n : ℕ} (x y : Latent n) : ℝ :=
  dist (WithLp.toLp 2 x : EuclideanSpace ℝ (Fin n)) (WithLp.toLp 2 y)

@[simp]
theorem eucDist_self {n : ℕ} (x : Latent n) : eucDist x x = 0 := by
  simp [eucDist]

@[simp]
theorem eucDist_eq_zero {n : ℕ} {x y : Latent n} : eucDist x y = 0 ↔ x = y := by
  simp [eucDist]

/-- An element of O(n), given as a linear isometry of
    `EuclideanSpace ℝ (Fin n)` (the type of the recovery map of
    Theorem 4.1), acting on `Latent n`. -/
def liftIso {n : ℕ}
    (U : EuclideanSpace ℝ (Fin n) →ₗᵢ[ℝ] EuclideanSpace ℝ (Fin n)) :
    Latent n → Latent n :=
  fun z => WithLp.ofLp (U (WithLp.toLp 2 z))

theorem eucDist_liftIso {n : ℕ}
    (U : EuclideanSpace ℝ (Fin n) →ₗᵢ[ℝ] EuclideanSpace ℝ (Fin n))
    (x y : Latent n) :
    eucDist (liftIso U x) (liftIso U y) = eucDist x y := by
  simp [eucDist, liftIso]

@[simp]
theorem liftIso_neg {n : ℕ} (z : Latent n) :
    liftIso (LinearIsometryEquiv.neg ℝ (E := EuclideanSpace ℝ (Fin n))).toLinearIsometry z
      = -z := by
  simp [liftIso]

/-- Goal-reaching to the goal `zg`: terminal cost
    ℓ_T(z) = ‖z − zg‖², and a stage cost `r` on actions alone. -/
def goalReaching {n : ℕ} {Action : Type*} (r : Action → ℝ) (zg : Latent n) :
    ControlProblem n Action where
  stage_cost := fun _ act => r act
  terminal_cost := fun z => eucDist z zg ^ 2

/-- Goal-reaching transported along `U ∈ O(n)` is goal-reaching to the
    encoded goal `U zg`. -/
theorem goalReaching_isTransportOf {n : ℕ} {Action : Type*}
    (U : EuclideanSpace ℝ (Fin n) →ₗᵢ[ℝ] EuclideanSpace ℝ (Fin n))
    (r : Action → ℝ) (zg : Latent n) :
    IsTransportOf (goalReaching r (liftIso U zg)) (goalReaching r zg) (liftIso U) :=
  ⟨fun _ _ => rfl, fun _ => by simp only [goalReaching, eucDist_liftIso]⟩

/-- **Goal-reaching** (VERIFIED): for a recovery map
    `U ∈ O(n)`, a plan minimizes goal-reaching in the learned latent, with
    the goal encoded by the same map, iff it minimizes goal-reaching in the
    true latent. -/
theorem goalReaching_minimizer_equivalence {n : ℕ} {Action : Type*} {T : ℕ}
    (U : EuclideanSpace ℝ (Fin n) →ₗᵢ[ℝ] EuclideanSpace ℝ (Fin n))
    (r : Action → ℝ) (zg : Latent n)
    (E_hat E : ExpectedCosts n Action T)
    (hpush : IsPushforwardOf E_hat E (liftIso U))
    (a : Plan Action T) (z : Latent n) :
    (∀ a', totalCost (goalReaching r (liftIso U zg)) E_hat a (liftIso U z)
        ≤ totalCost (goalReaching r (liftIso U zg)) E_hat a' (liftIso U z)) ↔
    (∀ a', totalCost (goalReaching r zg) E a z
        ≤ totalCost (goalReaching r zg) E a' z) :=
  minimizer_equivalence_transport _ _ (liftIso U) E_hat E hpush
    (goalReaching_isTransportOf U r zg) a z

/-- The invariance hypothesis of `planning_equivalence`, applied to
    goal-reaching with a fixed goal, forces `U` to fix the goal. -/
theorem liftIso_eq_self_of_isOrthogonalInvariant_goalReaching {n : ℕ} {Action : Type*}
    (U : EuclideanSpace ℝ (Fin n) →ₗᵢ[ℝ] EuclideanSpace ℝ (Fin n))
    (r : Action → ℝ) (zg : Latent n)
    (h : IsOrthogonalInvariant (goalReaching r zg) (liftIso U)) :
    liftIso U zg = zg := by
  simpa [goalReaching] using h.2 zg

/-- Hence `planning_equivalence` does not cover goal-reaching under
    `−I ∈ O(n)` with a nonzero goal, while the covariant form does. -/
theorem not_isOrthogonalInvariant_goalReaching_neg {n : ℕ} {Action : Type*}
    (r : Action → ℝ) (zg : Latent n) (hzg : zg ≠ 0) :
    ¬ IsOrthogonalInvariant (goalReaching r zg)
        (liftIso (LinearIsometryEquiv.neg ℝ
          (E := EuclideanSpace ℝ (Fin n))).toLinearIsometry) := fun h =>
  hzg (by simpa [self_eq_neg] using
    (liftIso_eq_self_of_isOrthogonalInvariant_goalReaching _ r zg h).symm)


end
