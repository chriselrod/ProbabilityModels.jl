module ProbabilityModels

using   MacroTools, DiffRules,
        VectorizationBase, SIMDPirates, LoopVectorization, SLEEFPirates,
        PaddedMatrices, StructuredMatrices,
        DistributionParameters, ProbabilityDistributions,
        DynamicHMC, LogDensityProblems,
        Random, VectorizedRNG, RandomNumbers,
        LinearAlgebra, Statistics#, Distributed

import MacroTools: postwalk, prewalk, @capture, @q
import PaddedMatrices: RESERVED_INCREMENT_SEED_RESERVED, RESERVED_DECREMENT_SEED_RESERVED,
                RESERVED_MULTIPLY_SEED_RESERVED, RESERVED_NMULTIPLY_SEED_RESERVED,
                AbstractFixedSizePaddedVector

export @model, NUTS_init_tune_mcmc_default, sample_cov, sample_mean

include("adjoints.jl")
include("misc_functions.jl")
include("special_diff_rules.jl")
include("reverse_autodiff_passes.jl")
include("model_macro_passes.jl")
include("dynamic_hmc_interface.jl")

end # module
