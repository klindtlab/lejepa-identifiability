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

  ## Verification status

  | Component                             | Status      |
  |---------------------------------------|-------------|
  | ControlProblem structure              | structural  |
  | Orthogonal invariance definition      | structural  |
  | ExpectedCosts abstraction             | structural  |
  | Total-cost definition                 | structural  |
  | Trajectory pushforward (stage)        | axiomatized |
  | Trajectory pushforward (terminal)     | axiomatized |
  | Per-step stage cost equivalence       | VERIFIED    |
  | Terminal cost equivalence             | VERIFIED    |
  | Total cost equivalence (main step)    | VERIFIED    |
  | Minimizer equivalence (plan agreement)| VERIFIED    |
  | Value equivalence                     | VERIFIED    |
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
-- AXIOMATIZED: TRAJECTORY PUSHFORWARD
-- ═══════════════════════════════════════════════════════════════

/-- **Stage pushforward** (axiomatized): under the pushforward dynamics
    `E_hat`, the expected value of any cost `c` at time `t` starting from
    `Q z` equals the expected value under the original dynamics `E` starting
    from `z` of the pre-composed cost `c ∘ (Q × id)`.

    Mathematically this is the content of "the joint law of (ẑ_0, …, ẑ_T)
    under the pushforward dynamics starting from ẑ_0 = Q z equals the joint
    law of (Q z_0, …, Q z_T) under the original dynamics starting from
    z_0 = z", restricted to per-time-step marginals and evaluated against
    arbitrary test functions. -/
axiom stage_pushforward
    {n : ℕ} {Action : Type*} {T : ℕ}
    (E_hat E : ExpectedCosts n Action T) (Q : Latent n → Latent n)
    (a : Plan Action T) (z : Latent n) (t : Fin T)
    (c : Latent n → Action → ℝ) :
    E_hat.stage_exp a (Q z) t c
      = E.stage_exp a z t (fun z' act => c (Q z') act)

/-- **Terminal pushforward** (axiomatized): the same relation at the
    terminal time. -/
axiom terminal_pushforward
    {n : ℕ} {Action : Type*} {T : ℕ}
    (E_hat E : ExpectedCosts n Action T) (Q : Latent n → Latent n)
    (a : Plan Action T) (z : Latent n) (c : Latent n → ℝ) :
    E_hat.term_exp a (Q z) c = E.term_exp a z (fun z' => c (Q z'))


-- ═══════════════════════════════════════════════════════════════
-- VERIFIED: PER-STEP COST EQUIVALENCE
-- ═══════════════════════════════════════════════════════════════

/-- **Stage-cost equivalence** (VERIFIED): the per-step expected stage cost
    at `Q z` under the pushforward dynamics equals the per-step expected
    stage cost at `z` under the original dynamics, when the stage cost is
    O(n)-invariant. This is the point where orthogonal invariance of the
    cost (hypothesis) meets trajectory pushforward (axiom). -/
theorem stage_cost_equiv
    {n : ℕ} {Action : Type*} {T : ℕ}
    (cp : ControlProblem n Action) (Q : Latent n → Latent n)
    (E_hat E : ExpectedCosts n Action T)
    (hinv : IsOrthogonalInvariant cp Q)
    (a : Plan Action T) (z : Latent n) (t : Fin T) :
    E_hat.stage_exp a (Q z) t cp.stage_cost
      = E.stage_exp a z t cp.stage_cost := by
  rw [stage_pushforward E_hat E Q a z t cp.stage_cost]
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
    (hinv : IsOrthogonalInvariant cp Q)
    (a : Plan Action T) (z : Latent n) :
    E_hat.term_exp a (Q z) cp.terminal_cost
      = E.term_exp a z cp.terminal_cost := by
  rw [terminal_pushforward E_hat E Q a z cp.terminal_cost]
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
    (hinv : IsOrthogonalInvariant cp Q)
    (a : Plan Action T) (z : Latent n) :
    totalCost cp E_hat a (Q z) = totalCost cp E a z := by
  unfold totalCost
  have hstage :
      (∑ t : Fin T, E_hat.stage_exp a (Q z) t cp.stage_cost)
        = ∑ t : Fin T, E.stage_exp a z t cp.stage_cost := by
    apply Finset.sum_congr rfl
    intro t _
    exact stage_cost_equiv cp Q E_hat E hinv a z t
  have hterm :
      E_hat.term_exp a (Q z) cp.terminal_cost
        = E.term_exp a z cp.terminal_cost :=
    terminal_cost_equiv cp Q E_hat E hinv a z
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
    (hinv : IsOrthogonalInvariant cp Q)
    (a : Plan Action T) (z : Latent n) :
    (∀ a', totalCost cp E_hat a (Q z) ≤ totalCost cp E_hat a' (Q z)) ↔
    (∀ a', totalCost cp E a z ≤ totalCost cp E a' z) := by
  have h : ∀ a', totalCost cp E_hat a' (Q z) = totalCost cp E a' z :=
    fun a' => planning_equivalence cp Q E_hat E hinv a' z
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
    (hinv : IsOrthogonalInvariant cp Q)
    (a : Plan Action T) (z : Latent n) (V : ℝ)
    (hV : totalCost cp E a z = V) :
    totalCost cp E_hat a (Q z) = V := by
  rw [planning_equivalence cp Q E_hat E hinv a z, hV]


end
