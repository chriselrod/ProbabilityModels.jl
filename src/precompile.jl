
function precomp()
    Threads.@spawn precompile(
        LoopVectorization.vectorize_body, (Int, Float64, Int, Symbol, Vector{Any}, Dict{Symbol,Tuple{Symbol,Symbol}}, SIMDPirates.Vec, Bool, Symbol)
    )
    Threads.@spawn precompile(
        LoopVectorization.vectorize_body, (Expr, Float64, Int, Symbol, Vector{Any}, Dict{Symbol,Tuple{Symbol,Symbol}}, SIMDPirates.Vec, Bool, Symbol)
    )
    Threads.@spawn precompile(
        LoopVectorization.vectorize_body, (Int, Float64, Int, Symbol, Vector{Any}, Dict{Symbol,Tuple{Symbol,Symbol}}, SIMDPirates.Vec, Bool, Module)
    )
    Threads.@spawn precompile(
        LoopVectorization.vectorize_body, (Expr, Float64, Int, Symbol, Vector{Any}, Dict{Symbol,Tuple{Symbol,Symbol}}, SIMDPirates.Vec, Bool, Module)
    )
    Threads.@spawn precompile(StructuredMatrices.A_rdiv_U_kernel_quote, StructuredMatrices.A_rdiv_U_config{Float64})
    Threads.@spawn precompile(StructuredMatrices.A_rdiv_L_kernel_quote, StructuredMatrices.A_rdiv_L_config{Float64})
    Threads.@spawn precompile(PaddedMatrices.mulquote, (Int,Int,Int,Int,Int,Float64,Symbol,Nothing,Int))
    Threads.@spawn precompile(PaddedMatrices.kernel_quote, (PaddedMatrices.DynamicKernel,))
    
    Threads.@spawn precompile(generate_generated_funcs_expressions, (Symbol, Expr))
    Threads.@spawn precompile(load_and_constrain_quote, (Symbol, Symbol, Vector{Symbol}, Vector{Symbol}, Symbol, Symbol, Symbol))

    Threads.@spawn precompile(
        ProbabilityDistributions.distribution_diff_rule!,
        (ReverseDiffExpressionsBase.InitializedVarTracker, Vector{Any}, Set{Symbol}, Symbol, Symbol, Vector{Any}, Symbol, Bool)
    )
    Threads.@spawn precompile(
        ProbabilityDistributions.distribution_diff_rule!,
        (ReverseDiffExpressionsBase.InitializedVarTracker, Vector{Any}, Set{Symbol}, Symbol, Symbol, Vector{Symbol}, Symbol, Bool)
    )
    Threads.@spawn precompile(ProbabilityDistributions.Bernoulli_logit_quote, (Float64,))
    Threads.@spawn precompile(ProbabilityDistributions.∂Bernoulli_logit_quote, (Float64, Bool))
    Threads.@spawn precompile(ProbabilityDistributions.Binomial_logit_quote, (Float64, Bool))
    Threads.@spawn precompile(ProbabilityDistributions.∂Binomial_logit_quote, (Float64, Bool, Bool))
    Threads.@spawn precompile(gamma_quote, (Int, Float64, NTuple{3,Bool}, NTuple{3,Bool}, Bool, NTuple{3,Bool}))
    Threads.@spawn precompile(beta_quote, (Int, Float64, NTuple{3,Bool}, NTuple{3,Bool}, NTuple{3,Bool}, Bool))
    
    Threads.@spawn precompile(ProbabilityDistributions.multivariate_normal_SMLT_quote, (ProbabilityDistributions.NormalCholeskyConfiguration{Float64},))
    Threads.@spawn precompile(ProbabilityDistributions.∂multivariate_normal_SMLT_quote, (ProbabilityDistributions.NormalCholeskyConfiguration{Float64},))
    Threads.@spawn precompile(ProbabilityDistributions.univariate_normal_quote, (Int,Float64,Bool,Bool,Bool,NTuple{3,Bool},NTuple{3,Bool},Bool,Bool))

    Threads.@spawn precompile(
        ReverseDiffExpressions.differentiate!,
        (Vector{Any}, Vector{Any}, Set{Symbol}, ReverseDiffExpressionsBase.InitializedVarTracker,Symbol, Symbol, Vector{Any}, Symbol, Bool)
    )
    Threads.@spawn precompile(MCMCChainSummaries.MCMCChainSummary, (PaddedMatrices.DynamicPaddedArray{Float64,3},Vector{String},NTuple{5,Float64},Bool))

end

