import Lake
open Lake DSL

package lejepa where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib LeJEPA where

require "leanprover-community" / "mathlib" @ git "v4.28.0"
