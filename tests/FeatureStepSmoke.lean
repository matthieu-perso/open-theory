import OpenTheory.Dynamics.FeatureStep

open OpenTheory

-- Moving the one-step identity or the three rate comparisons must break these checks.
#check FeatureStep.preactivation_step
#check FeatureStep.loss_hasDerivAt_entry
#check FeatureStep.ntk_preactivation_step_le
#check FeatureStep.mup_preactivation_step_le
#check FeatureStep.feature_learning_preactivation_step_le
#check FeatureStep.feature_learning_preactivation_step_ge
