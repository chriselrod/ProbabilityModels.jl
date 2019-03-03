module ProbabilityModels

using   MacroTools, DiffRules,
        VectorizationBase, SIMDPirates, LoopVectorization,
        PaddedMatrices, DistributionParameters, ProbabilityDistributions,
        DynamicHMC, LogDensityProblems,
        Random, VectorizedRNG, RandomNumbers

import MacroTools: postwalk, prewalk, @capture

export @model, LogDensityProblems.Value, LogDensityProblems.ValueGradient, LogDensityProblems.logdensity,
        NUTS_init_tune_mcmc_default

include("special_diff_rules.jl")
include("model_macro_passes.jl")
include("dynamic_hmc_interface.jl")

end # module
