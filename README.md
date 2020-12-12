# ProbabilityModels

[![CI](https://github.com/chriselrod/ProbabilityModels.jl/workflows/CI/badge.svg)](https://github.com/chriselrod/ProbabilityModels.jl/actions?query=workflow%3ACI)
[![CI (Julia nightly)](https://github.com/chriselrod/ProbabilityModels.jl/workflows/CI%20(Julia%20nightly)/badge.svg)](https://github.com/chriselrod/ProbabilityModels.jl/actions?query=workflow%3A%22CI+%28Julia+nightly%29%22)

This is alpha-quality software. It is under active development. Optimistically, I hope to have it and its dependencies reasonably well documented and tested, and all the libraries registered, by the end of the year. There is a roadmap issue [here](https://github.com/chriselrod/ProbabilityModels.jl/issues/5).

The primary goal of this library is to make it as easy as possible to specify models that run as quickly as possible $-$ providing both log densities and the associated gradients. This allows the library to be a front end to Hamiltonian Monte Carlo backends, such as [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl). [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)/[Turing's NUTS](https://github.com/TuringLang/Turing.jl) is also probably worth looking into, as it [reportedly converges the most reliably](https://discourse.julialang.org/t/mcmc-landscape/25654/11?u=elrod), at least within DiffEqBayes.


*A brief introduction.*

First, you specify a model using a DSL:
```julia
using PaddedMatrices, StructuredMatrices, DistributionParameters, LoopVectorization
using InplaceDHMC, VectorizedRNG
using Random, SpecialFunctions, MCMCChainSummaries, LinearAlgebra
using ProbabilityModels, LoopVectorization, SLEEFPirates, SIMDPirates, ProbabilityDistributions, PaddedMatrices
using ProbabilityModels: HierarchicalCentering, ∂HierarchicalCentering, ITPExpectedValue, ∂ITPExpectedValue
using DistributionParameters: CovarianceMatrix, MissingDataVector#, add
using PaddedMatrices: vexp
BLAS.set_num_threads(1)

@model ITPModel begin
    # Non-hierarchical Priors
    ρ ~ Beta(3, 1)
    lκ ~ lsgg(8.5, 1.5, 3.0, 1.5271796258079011) # μ = 1, σ² = 10
    σ ~ Gamma(1.5, 0.25) # μ = 6, σ² = 2.4
    θ ~ Normal(10)
    L ~ LKJ(2.0)
    # Hierarchical Priors.
    # h subscript, for highest in the hierarhcy.
    μₕ₁ ~ Normal(10) # μ = 0
    μₕ₂ ~ Normal(10) # μ = 0
    σₕ ~ Normal(10) # μ = 0
    # Raw μs; non-cenetered parameterization
    μᵣ₁ ~ Normal() # μ = 0, σ = 1
    μᵣ₂ ~ Normal() # μ = 0, σ = 1
    # Center the μs
    μᵦ₁ = HierarchicalCentering(μᵣ₁, μₕ₁, σₕ)
    μᵦ₂ = HierarchicalCentering(μᵣ₂, μₕ₂, σₕ)
    σᵦ ~ Normal(10) # μ = 0
    # Raw βs; non-cenetered parameterization
    βᵣ₁ ~ Normal()
    βᵣ₂ ~ Normal()
    # Center the βs.
    β₁ = HierarchicalCentering(βᵣ₁, μᵦ₁, σᵦ, domains)
    β₂ = HierarchicalCentering(βᵣ₂, μᵦ₂, σᵦ, domains)
    # Likelihood
    κ = vexp(lκ)
    μ₁ = vec(ITPExpectedValue(time, β₁, κ, θ))
    μ₂ = vec(ITPExpectedValue(time, β₂, κ, θ))
    Σ = CovarianceMatrix(ρ, Diagonal(σ) * L, time)
    # Tuple (Y₁, Y₂)
    (Y₁, Y₂) ~ Normal((μ₁, μ₂)[AvailableData], Σ[AvailableData])
end
# Defined model: ITPModel.
# Unknowns: Y₂, domains, μₕ₂, μᵣ₁, time, σ, AvailableData, σᵦ, θ, μᵣ₂, ρ, σₕ, lκ, μₕ₁, L, Y₁, βᵣ₂, βᵣ₁.
```

The `@model` macro uses the following expression to define a struct and several functions.
The struct has one field for each unknown.

We can then define an instance of the struct, where we specify each of these unknowns either with an instance (e.g. assign a piece of data), or with a type.
Those specified with a type are unknown parameters (of that type); those specified with an instance treat that instance as known priors or data. For example,
(if we had the appropriate functions defined in scope), we could create an instance with:

```julia
K = 7; D = 3
data = ITPModel(
    domains = domains,
    AvailableData = missingness,
    Y₁ = Y₁,
    Y₂ = Y₂,
    time = time_vector,
    L = LKJCorrCholesky{K},
    ρ = RealVector{K,Bounds(0,1)},
    lκ = RealVector{K},
    θ = RealVector{K},
    μₕ₁ = RealFloat,
    μₕ₂ = RealFloat,
    μᵣ₁ = RealVector{D},
    μᵣ₂ = RealVector{D},
    βᵣ₁ = RealVector{K},
    βᵣ₂ = RealVector{K},
    σᵦ = RealFloat{Bounds(0,Inf)},
    σₕ = RealFloat{Bounds(0,Inf)},
    σ = RealVector{K,Bounds(0,Inf)}
)
```
This would let `Y₁` equal the variable `Y₁`, while `ρ` is a vector (with length `K=7`) of unknown parameters bounded between 0 and 1. The default bounds are `(-Inf,Inf)`.

Before spewing boilerplate to generate random true values and fake data, a brief summary of the model:
We have longitudinal multivariate observations for some number of subjects. However, not all observations are measured at all times. That is, while subjects may be measured at multiple times (I use $T=36$ times below), only some measurements are taken at any given time, yielding missing data.
Therefore, we subset the full covariance matrix (produced from a vector of autocorrelations, and the Cholesky factor of a covariance matrix across measurements) to find the marginal.

The expected value is function of time (`ITPExpectedValue`). This function returns a matrix  (`time x measurement`), so we `vec` it and subset it.

We also expect some measurements to bare more in common than others, so we group them into "domains", and provide hierarchical priors. We use a non-cenetered parameterization, and the function `HierarchicalCentering` then centers our parameters for us. There are two methods we use above: one takes scalars, to transform a vector. The other accepts different domain means and standard deviations, and uses these to transform a vector, taking the indices from the `Domains` argument. That is, if the first element of `Domains` is 2, indicating that the first 2 measurements belong to the first domain, it will transform the first two elements of `βᵣ₁` with the first element of `μᵦ₁` and `σᵦ` (if either `μᵦ₁` or `σᵦ` are scalars, they will be broadcasted across each domain).


```julia
function rinvscaledgamma(::Val{N},a::T,b::T,c::T) where {N,T}
    rg = MutableFixedSizeVector{N,T}(undef)
    log100 = log(100)
    @inbounds for n ∈ 1:N
        rg[n] = log100 / exp(log(VectorizedRNG.randgamma(a/c)) / c + b)
    end
    rg
end

const domains = ProbabilityModels.Domains(2,2,3)

const n_endpoints = sum(domains)

const times = MutableFixedSizeVector{36,Float64,36}(undef); times .= 0:35;

structured_missing_pattern = push!(vcat(([1,0,0,0,0] for i ∈ 1:7)...), 1);
missing_pattern = vcat(
    structured_missing_pattern, fill(1, 4length(times)), structured_missing_pattern, structured_missing_pattern
);

const availabledata = MissingDataVector{Float64}(missing_pattern);

const κ₀ = (8.5, 1.5, 3.0)

AR1(ρ, t) = @. ρ ^ abs(t - t')


function generate_true_parameters(domains, times, κ₀)
    K = sum(domains)
    D = length(domains)
    T = length(times)
    μ = MutableFixedSizeVector{K,Float64}(undef)
    offset = 0
    for i ∈ domains
        domain_mean = 5randn()
        for j ∈ 1+offset:i+offset
            μ[j] = domain_mean + 5randn()
        end
        offset += i
    end
    σ = VectorizedRNG.randgamma( 6.0, 1/6.0)
    ρ = VectorizedRNG.randbeta(4.0,4.0)
    κ = rinvscaledgamma(Val(K), κ₀...)
    lt = last(times)
    β = MutableFixedSizeVector{7,Float64,7}(( 0.0625,  0.0575,  0.0525,  0.0475,  0.0425,  0.04,  0.0375))
    θ₁ = @. μ' - β' * ( 1.0 - exp( - κ' * times) ) / (1.0 - exp( - κ' * lt) ) 
    θ₂ = @. μ' + β' * ( 1.0 - exp( - κ' * times) ) / (1.0 - exp( - κ' * lt) ) 
    
    L_T, info = LAPACK.potrf!('L', AR1(ρ, times))
    @inbounds for tc ∈ 2:T, tr ∈ 1:tc-1
        L_T[tr,tc] = 0.0
    end
    X = PaddedMatrices.MutableFixedSizeMatrix{K,K+3,Float64,K}(undef); randn!(X)
    U_K, info = LAPACK.potrf!('U', BLAS.syrk!('U', 'N', σ, X, 0.0, zero(MutableFixedSizeMatrix{K,K,Float64,K})))
    (
        U_K = U_K, L_T = L_T, μ = μ, θ₁ = θ₁, θ₂ = θ₂, domains = domains, time = times
    )
end

@generated function randomize!(
    sp::PaddedMatrices.StackPointer,
    A::AbstractArray{T,P},
    B::PaddedMatrices.AbstractMutableFixedSizeMatrix{M,M,T},
    C::PaddedMatrices.AbstractMutableFixedSizeMatrix{N,N,T},
    D::PaddedMatrices.AbstractMutableFixedSizeMatrix{M,N,T}
) where {M,N,T,P}
    quote
        @boundscheck begin
            d = size(A,1)
            for p ∈ 2:$(P-1)
                d *= size(A,p)
            end
            d == M*N || PaddedMatrices.ThrowBoundsError("Earlier dims size(A) == $(size(A)) does not match size(D) == ($M,$N)")
        end
        ptr = pointer(sp, $T)
        E = PtrMatrix{$M,$N,$T,$M}( ptr )
        F = PtrMatrix{$M,$N,$T,$M}( ptr + $(sizeof(T) * M * N) )
        ptr_A = pointer(A)
        GC.@preserve A begin
            for n ∈ 0:size(A,$P)-1
                Aₙ = PtrMatrix{$M,$N,$T,$M}( ptr_A + n*$(sizeof(T)*M*N) )
                randn!(ProbabilityModels.GLOBAL_PCGs[1], E)
                mul!(F, B, E)
                Aₙ .= D
                PaddedMatrices.gemm!(Aₙ, F, C)
            end
        end
        sp
    end
end

sample_data( N, truth, missingness, missingvals = (Val{0}(),Val{0}()) ) = sample_data( N, truth, missingness, missingvals, truth.domains )
@generated function sample_data(
    N::Tuple{Int,Int},
	truth, missingness,
	::Tuple{Val{M1},Val{M2}},
	::ProbabilityModels.Domains{S}
) where {S,M1,M2}
    K = sum(S)
    D = length(S)
    quote
        N₁, N₂ = N
        L_T = truth.L_T
        U_K = truth.U_K
        T = size(L_T,1)

        sp = ProbabilityModels.STACK_POINTER_REF[]
        (sp,Y₁) = PaddedMatrices.DynamicPtrArray{Float64,3}(sp, (T, $K, N₁), T)
        (sp,Y₂) = PaddedMatrices.DynamicPtrArray{Float64,3}(sp, (T, $K, N₂), T)
        randomize!(sp, Y₁, L_T, U_K, truth.θ₁)
        randomize!(sp, Y₂, L_T, U_K, truth.θ₂)
        
        c = length(missingness.indices)
        inds = missingness.indices
        
        Y₁sub = reshape(Y₁, (T * $K, N₁))[inds, :]
        $(M1 > 0 ? quote
            Y₁union = Array{Union{Missing,Float64}}(Y₁sub)
            perm = randperm(length(Y₁sub))
            @inbounds for m in 1:$M1
                Y₁union[perm[m]] = Base.missing
	    	end
    		Y₁ = convert(MissingDataArray{$M1,Bounds(-Inf,Inf)}, Y₁union)
		end : quote
			Y₁ = Y₁sub
		end)

        Y₂sub = reshape(Y₂, (T * $K, N₂))[inds, :]
		$(M2 > 0 ? quote
            Y₂union = Array{Union{Missing,Float64}}(Y₂sub)
            perm = randperm(length(Y₂sub))
			@inbounds for m in 1:$M2
                Y₂union[perm[m]] = Base.missing
            end
            Y₂ = convert(MissingDataArray{$M2,Bounds(-Inf,Inf)},Y₂union)
		end : quote
		    Y₂ = Y₂sub
		end)
		
        ITPModel(
            domains = truth.domains,
	        AvailableData = missingness,
            Y₁ = Y₁,
            Y₂ = Y₂,
            time = truth.time,
            L = LKJCorrCholesky{$K},
            ρ = RealVector{$K,Bounds(0,1)},
            lκ = RealVector{$K},
            θ = RealVector{$K},
            μₕ₁ = RealFloat,
            μₕ₂ = RealFloat,
            μᵣ₁ = RealVector{$D},
            μᵣ₂ = RealVector{$D},
            βᵣ₁ = RealVector{$K},
            βᵣ₂ = RealVector{$K},
            σᵦ = RealFloat{Bounds(0,Inf)},
            σₕ = RealFloat{Bounds(0,Inf)},
            σ = RealVector{$K,Bounds(0,Inf)}
        )
    end
end

```
The library [DistributionParameters.jl](https://github.com/chriselrod/DistributionParameters.jl) provides a variety of parameter types.
These types define constrianing transformations, to transform an unconstrained parameter vector and add the appropriate jacobians.

All parameters are typed by size. The library currently provides a DynamicHMC interface, defining `logdensity(::Value,::ITPModel)` and `logdensity(::ValueGradient,::ITPModel)` methods.

```julia


```
For comparison, a Stan implementation of this model takes about to 13ms to evaluate the gradient.

The `@model` macro also defines a helper function for [constraining](https://mc-stan.org/docs/2_19/reference-manual/variable-transforms-chapter.html) unconstrained parameter vectors:

```julia


```
So you can use this to constrain the unconstrained parameter vectors `DynamicHMC` sampled and proceed with your convergence assessments and posterior analysis as normal.

Alternatively, it also supports [MCMCChainSummaries.jl](https://github.com/chriselrod/MCMCChainSummaries.jl), constraining the parameters for you and providing posterior summaries as well as plotting methods:
```julia

```
The difference in time between the slowest chain (748 s) and the overall time (775 s) roughly yields the compilation time. If we refit the model, the total time would roughly equal the length of time to sample from the slowest chain.

If you don't pass the sampler a tuner object, it'll create one by default with `M=5` and `term=50`. The section on [Automatic Parameter Tuning](https://mc-stan.org/docs/2_19/reference-manual/hmc-algorithm-parameters.html) in the Stan reference manual explains what these mean. I added an extra slow adaptation step (with twice the length of the previous step) and doubled the length of the terminal adaptation window (a final window calculating step size), as this makes adaptation more consistent.
The idea is to try and decrease the probability of one or two chains being much slower than the others. We can look at NUTS statistics of the chains:
```julia
julia> NUTS_statistics.(chains)
18-element Array{DynamicHMC.NUTS_Statistics{Float64,DataStructures.Accumulator{DynamicHMC.Termination,Int64},DataStructures.Accumulator{Int64,Int64}},1}:
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.94, min/25%/median/75%/max: 0.0 0.94 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 6% DoubledTurn => 93%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 13% 6 => 87%
                         
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.96 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 0% DoubledTurn => 100%
  depth: 4 => 0% 5 => 1% 6 => 99% 7 => 0%
                                 
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.96 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 0% DoubledTurn => 99%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 1% 6 => 99% 7 => 0%
                  
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.97 0.99 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 0% DoubledTurn => 99%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 0% 6 => 100% 7 => 0%
                 
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.98, min/25%/median/75%/max: 0.0 0.98 0.99 1.0 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 17% DoubledTurn => 83%
  depth: 4 => 0% 5 => 0% 6 => 93% 7 => 7% 8 => 0%
                          
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.97 0.99 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 0% DoubledTurn => 100%
  depth: 3 => 0% 4 => 0% 5 => 0% 6 => 100% 7 => 0%
                        
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.94, min/25%/median/75%/max: 0.0 0.95 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 4% DoubledTurn => 96%
  depth: 3 => 0% 4 => 0% 5 => 8% 6 => 92% 7 => 0%
                          
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.95, min/25%/median/75%/max: 0.0 0.96 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 1% DoubledTurn => 99%
  depth: 3 => 0% 4 => 0% 5 => 1% 6 => 99% 7 => 0%
                          
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.94, min/25%/median/75%/max: 0.0 0.95 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 3% DoubledTurn => 97%
  depth: 1 => 0% 2 => 0% 3 => 0% 4 => 0% 5 => 6% 6 => 94% 7 => 0%
          
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.94, min/25%/median/75%/max: 0.0 0.94 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 2% DoubledTurn => 98%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 4% 6 => 96% 8 => 0%
                  
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.91, min/25%/median/75%/max: 0.0 0.92 0.97 0.99 1.0
  termination: AdjacentDivergent => 1% AdjacentTurn => 12% DoubledTurn => 87%
  depth: 1 => 0% 2 => 0% 3 => 0% 4 => 0% 5 => 28% 6 => 71% 7 => 0%
        
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.96 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 0% DoubledTurn => 100%
  depth: 1 => 0% 2 => 0% 3 => 0% 4 => 0% 5 => 1% 6 => 99% 7 => 0% 8 => 0%
 
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.96 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 0% DoubledTurn => 100%
  depth: 3 => 0% 4 => 0% 5 => 1% 6 => 99% 7 => 0% 8 => 0%
                 
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.94, min/25%/median/75%/max: 0.0 0.94 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 4% DoubledTurn => 96%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 8% 6 => 92% 7 => 0%
                  
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.95, min/25%/median/75%/max: 0.0 0.95 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 1% DoubledTurn => 99%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 3% 6 => 97% 7 => 0%
                  
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.96 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 1% DoubledTurn => 99%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 1% 6 => 99% 7 => 0% 8 => 0%
          
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.91, min/25%/median/75%/max: 0.0 0.92 0.97 0.99 1.0
  termination: AdjacentDivergent => 1% AdjacentTurn => 16% DoubledTurn => 83%
  depth: 1 => 0% 2 => 1% 3 => 0% 4 => 1% 5 => 39% 6 => 60% 7 => 0% 8 => 0%

 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.91, min/25%/median/75%/max: 0.0 0.92 0.97 0.99 1.0
  termination: AdjacentDivergent => 2% AdjacentTurn => 9% DoubledTurn => 89%
  depth: 1 => 0% 2 => 1% 3 => 0% 4 => 0% 5 => 20% 6 => 78% 7 => 0%
```
and see that the fifth chain had an acceptance rate of 0.98, vs the target acceptance rate of `δ = 0.95`. 7% of samples hit a treedepth of 7, meaning this was probably the slowest chain (worker 15). Maybe increasing term further would help.


*Overview of how the library works.*

The `ITPModel` struct (or whatever you've named your model, with the first argument to the macro) is defined as a struct with a field for each unknown. Each field is parametrically typed.

`logdensity` and `constrain` are defined as generated functions, so that they can compile appropriate code given these parameteric types. The `@model` macro does a little preprocessing of the expression; most of the code generation occurs within these generated functions.

The macro's preprocessing consists of simply translating the sampling statements into log probability increments (`target`, terminology taken from the [Stan](https://mc-stan.org/users/documentation/) language), and lowering the expression.

We can manually perform these passes on the expression (note that variables with `#` in their names are NOT comments -- they're hygienic names, guaranteeing all the variables I add don't clash with any of the model's variables):
```julia
itp_q = quote
    # Non-hierarchical Priors
    ρ ~ Beta(3, 1)
    κ ~ Gamma(0.1, 0.1) # μ = 1, σ² = 10
    σ ~ Gamma(1.5, 0.25) # μ = 6, σ² = 2.4
    θ ~ Normal(10)
    L ~ LKJ(2.0)

    # Hierarchical Priors.
    # h subscript, for highest in the hierarhcy.
    μₕ₁ ~ Normal(10) # μ = 0
    μₕ₂ ~ Normal(10) # μ = 0
    σₕ ~ Normal(10) # μ = 0
    # Raw μs; non-cenetered parameterization
    μᵣ₁ ~ Normal() # μ = 0, σ = 1
    μᵣ₂ ~ Normal() # μ = 0, σ = 1
    # Center the μs
    μᵦ₁ = HierarchicalCentering(μᵣ₁, μₕ₁, σₕ)
    μᵦ₂ = HierarchicalCentering(μᵣ₂, μₕ₂, σₕ)
    σᵦ ~ Normal(10) # μ = 0
    # Raw βs; non-cenetered parameterization
    βᵣ₁ ~ Normal()
    βᵣ₂ ~ Normal()
    # Center the βs.
    β₁ = HierarchicalCentering(βᵣ₁, μᵦ₁, σᵦ, domains)
    β₂ = HierarchicalCentering(βᵣ₂, μᵦ₂, σᵦ, domains)

    # Likelihood
    μ₁ = vec(ITPExpectedValue(time, β₁, κ, θ))
    μ₂ = vec(ITPExpectedValue(time, β₂, κ, θ))
    Σ = CovarianceMatrix(ρ, Diagonal(σ) * L, time)

    (Y₁, Y₂) ~ Normal((μ₁, μ₂)[AvailableData], Σ[AvailableData])

end

itp_preprocessed = itp_q |> ProbabilityModels.translate_sampling_statements |> ProbabilityModels.flatten_expression;

using MacroTools: striplines

striplines(itp_preprocessed)
```
This yields the following expression:

```julia
quote
    ##SSAValue##1## = Beta(ρ, 3, 1)
    ##SSAValue##2## = vadd(target, ##SSAValue##1##)
    target = ##SSAValue##2##
    ##SSAValue##4## = Gamma(κ, 0.1, 0.1)
    ##SSAValue##5## = vadd(target, ##SSAValue##4##)
    target = ##SSAValue##5##
    ##SSAValue##7## = Gamma(σ, 1.5, 0.25)
    ##SSAValue##8## = vadd(target, ##SSAValue##7##)
    target = ##SSAValue##8##
    ##SSAValue##10## = Normal(θ, 10)
    ##SSAValue##11## = vadd(target, ##SSAValue##10##)
    target = ##SSAValue##11##
    ##SSAValue##13## = LKJ(L, 2.0)
    ##SSAValue##14## = vadd(target, ##SSAValue##13##)
    target = ##SSAValue##14##
    ##SSAValue##16## = Normal(μₕ₁, 10)
    ##SSAValue##17## = vadd(target, ##SSAValue##16##)
    target = ##SSAValue##17##
    ##SSAValue##19## = Normal(μₕ₂, 10)
    ##SSAValue##20## = vadd(target, ##SSAValue##19##)
    target = ##SSAValue##20##
    ##SSAValue##22## = Normal(σₕ, 10)
    ##SSAValue##23## = vadd(target, ##SSAValue##22##)
    target = ##SSAValue##23##
    ##SSAValue##25## = Normal(μᵣ₁)
    ##SSAValue##26## = vadd(target, ##SSAValue##25##)
    target = ##SSAValue##26##
    ##SSAValue##28## = Normal(μᵣ₂)
    ##SSAValue##29## = vadd(target, ##SSAValue##28##)
    target = ##SSAValue##29##
    ##SSAValue##31## = HierarchicalCentering(μᵣ₁, μₕ₁, σₕ)
    μᵦ₁ = ##SSAValue##31##
    ##SSAValue##33## = HierarchicalCentering(μᵣ₂, μₕ₂, σₕ)
    μᵦ₂ = ##SSAValue##33##
    ##SSAValue##35## = Normal(σᵦ, 10)
    ##SSAValue##36## = vadd(target, ##SSAValue##35##)
    target = ##SSAValue##36##
    ##SSAValue##38## = Normal(βᵣ₁)
    ##SSAValue##39## = vadd(target, ##SSAValue##38##)
    target = ##SSAValue##39##
    ##SSAValue##41## = Normal(βᵣ₂)
    ##SSAValue##42## = vadd(target, ##SSAValue##41##)
    target = ##SSAValue##42##
    ##SSAValue##44## = HierarchicalCentering(βᵣ₁, μᵦ₁, σᵦ, domains)
    β₁ = ##SSAValue##44##
    ##SSAValue##46## = HierarchicalCentering(βᵣ₂, μᵦ₂, σᵦ, domains)
    β₂ = ##SSAValue##46##
    ##SSAValue##48## = ITPExpectedValue(time, β₁, κ, θ)
    ##SSAValue##49## = vec(##SSAValue##48##)
    μ₁ = ##SSAValue##49##
    ##SSAValue##51## = ITPExpectedValue(time, β₂, κ, θ)
    ##SSAValue##52## = vec(##SSAValue##51##)
    μ₂ = ##SSAValue##52##
    ##SSAValue##54## = Diagonal(σ)
    ##SSAValue##55## = ##SSAValue##54## * L
    ##SSAValue##56## = CovarianceMatrix(ρ, ##SSAValue##55##, time)
    Σ = ##SSAValue##56##
    ##SSAValue##58## = Core.tuple(Y₁, Y₂)
    ##SSAValue##59## = Core.tuple(μ₁, μ₂)
    ##SSAValue##60## = Base.getindex(##SSAValue##59##, AvailableData)
    ##SSAValue##61## = Base.getindex(Σ, AvailableData)
    ##SSAValue##62## = Normal(##SSAValue##58##, ##SSAValue##60##, ##SSAValue##61##)
    ##SSAValue##63## = vadd(target, ##SSAValue##62##)
    target = ##SSAValue##63##
end
```
After translating the sampling statements into function calls, the code is transformed with `Meta.lower`. The resulting `Code.Info` is then transformed back into a Julia expression, as you see above. This has the advantage of flattening the expression, so that there is at most only a single function call per line.

The `logdensity` generated functions are defined with the above expressions assigned to a variable, so that they can apply additional transformations.


I'll focus on the `ValueGradient` method, as this one performs the more interesting transformations. Aside from loading and constraining all the parameters from an input vector, it performs a `reverse_diff!` pass on the expression.
```julia
forward_pass = quote end
reverse_pass = quote end
tracked_vars = Set([:ρ, :κ, :σ, :θ, :L, :μₕ₁, :μₕ₂, :σₕ, :μᵣ₁, :μᵣ₂, :σᵦ, :βᵣ₁, :βᵣ₂, ]);
ProbabilityModels.reverse_diff_pass!(forward_pass, reverse_pass, itp_preprocessed, tracked_vars);
striplines(forward_pass)
```
It walks the expression, replacing each function with a function that also returns the adjoint; the forward pass becomes:
```julia
quote
    (##SSAValue##1##, ###adjoint###_##∂##SSAValue##1####∂ρ##) = ProbabilityDistributions.∂Beta(ρ, 3, 1, Val{(true, false, false)}())
    ##SSAValue##2## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##1##)
    target = ##SSAValue##2##
    (##SSAValue##4##, ###adjoint###_##∂##SSAValue##4####∂κ##) = ProbabilityDistributions.∂Gamma(κ, 0.1, 0.1, Val{(true, false, false)}())
    ##SSAValue##5## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##4##)
    target = ##SSAValue##5##
    (##SSAValue##7##, ###adjoint###_##∂##SSAValue##7####∂σ##) = ProbabilityDistributions.∂Gamma(σ, 1.5, 0.25, Val{(true, false, false)}())
    ##SSAValue##8## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##7##)
    target = ##SSAValue##8##
    (##SSAValue##10##, ###adjoint###_##∂##SSAValue##10####∂θ##) = ProbabilityDistributions.∂Normal(θ, 10, Val{(true, false)}())
    ##SSAValue##11## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##10##)
    target = ##SSAValue##11##
    (##SSAValue##13##, ###adjoint###_##∂##SSAValue##13####∂L##) = ProbabilityDistributions.∂LKJ(L, 2.0, Val{(true, false)}())
    ##SSAValue##14## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##13##)
    target = ##SSAValue##14##
    (##SSAValue##16##, ###adjoint###_##∂##SSAValue##16####∂μₕ₁##) = ProbabilityDistributions.∂Normal(μₕ₁, 10, Val{(true, false)}())
    ##SSAValue##17## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##16##)
    target = ##SSAValue##17##
    (##SSAValue##19##, ###adjoint###_##∂##SSAValue##19####∂μₕ₂##) = ProbabilityDistributions.∂Normal(μₕ₂, 10, Val{(true, false)}())
    ##SSAValue##20## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##19##)
    target = ##SSAValue##20##
    (##SSAValue##22##, ###adjoint###_##∂##SSAValue##22####∂σₕ##) = ProbabilityDistributions.∂Normal(σₕ, 10, Val{(true, false)}())
    ##SSAValue##23## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##22##)
    target = ##SSAValue##23##
    (##SSAValue##25##, ###adjoint###_##∂##SSAValue##25####∂μᵣ₁##) = ProbabilityDistributions.∂Normal(μᵣ₁, Val{(true,)}())
    ##SSAValue##26## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##25##)
    target = ##SSAValue##26##
    (##SSAValue##28##, ###adjoint###_##∂##SSAValue##28####∂μᵣ₂##) = ProbabilityDistributions.∂Normal(μᵣ₂, Val{(true,)}())
    ##SSAValue##29## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##28##)
    target = ##SSAValue##29##
    (##SSAValue##31##, ###adjoint###_##∂##SSAValue##31####∂μᵣ₁##, ###adjoint###_##∂##SSAValue##31####∂μₕ₁##, ###adjoint###_##∂##SSAValue##31####∂σₕ##) = ∂HierarchicalCentering(μᵣ₁, μₕ₁, σₕ, Val{(true, true, true)}())
    μᵦ₁ = ##SSAValue##31##
    (##SSAValue##33##, ###adjoint###_##∂##SSAValue##33####∂μᵣ₂##, ###adjoint###_##∂##SSAValue##33####∂μₕ₂##, ###adjoint###_##∂##SSAValue##33####∂σₕ##) = ∂HierarchicalCentering(μᵣ₂, μₕ₂, σₕ, Val{(true, true, true)}())
    μᵦ₂ = ##SSAValue##33##
    (##SSAValue##35##, ###adjoint###_##∂##SSAValue##35####∂σᵦ##) = ProbabilityDistributions.∂Normal(σᵦ, 10, Val{(true, false)}())
    ##SSAValue##36## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##35##)
    target = ##SSAValue##36##
    (##SSAValue##38##, ###adjoint###_##∂##SSAValue##38####∂βᵣ₁##) = ProbabilityDistributions.∂Normal(βᵣ₁, Val{(true,)}())
    ##SSAValue##39## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##38##)
    target = ##SSAValue##39##
    (##SSAValue##41##, ###adjoint###_##∂##SSAValue##41####∂βᵣ₂##) = ProbabilityDistributions.∂Normal(βᵣ₂, Val{(true,)}())
    ##SSAValue##42## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##41##)
    target = ##SSAValue##42##
    (##SSAValue##44##, ###adjoint###_##∂##SSAValue##44####βᵣ₁##, ###adjoint###_##∂##SSAValue##44####∂μᵦ₁##, ###adjoint###_##∂##SSAValue##44####∂σᵦ##) = ∂HierarchicalCentering(βᵣ₁, μᵦ₁, σᵦ, domains, Val{(true, true, true)}())
    β₁ = ##SSAValue##44##
    (##SSAValue##46##, ###adjoint###_##∂##SSAValue##46####∂βᵣ₂##, ###adjoint###_##∂##SSAValue##46####∂μᵦ₂##, ###adjoint###_##∂##SSAValue##46####∂σᵦ##) = ∂HierarchicalCentering(βᵣ₂, μᵦ₂, σᵦ, domains, Val{(true, true, true)}())
    β₂ = ##SSAValue##46##
    (##SSAValue##48##, ###adjoint###_##∂##SSAValue##48####∂β₁##, ###adjoint###_##∂##SSAValue##48####∂κ##, ###adjoint###_##∂##SSAValue##48####∂θ##) = ProbabilityModels.∂ITPExpectedValue(time, β₁, κ, θ, Val{(true, true, true)}())
    (##SSAValue##49##, ###adjoint###_##∂##SSAValue##49####∂##SSAValue##48####) = ProbabilityModels.∂vec(##SSAValue##48##)
    μ₁ = ##SSAValue##49##
    (##SSAValue##51##, ###adjoint###_##∂##SSAValue##51####∂β₂##, ###adjoint###_##∂##SSAValue##51####∂κ##, ###adjoint###_##∂##SSAValue##51####∂θ##) = ProbabilityModels.∂ITPExpectedValue(time, β₂, κ, θ, Val{(true, true, true)}())
    (##SSAValue##52##, ###adjoint###_##∂##SSAValue##52####∂##SSAValue##51####) = ProbabilityModels.∂vec(##SSAValue##51##)
    μ₂ = ##SSAValue##52##
    ##SSAValue##54## = LinearAlgebra.Diagonal(σ)
    ##SSAValue##55## = ##SSAValue##54## * L
    (##SSAValue##56##, ###adjoint###_##∂##SSAValue##56####∂ρ##, ###adjoint###_##∂##SSAValue##56####∂##SSAValue##55####) = ProbabilityModels.DistributionParameters.∂CovarianceMatrix(ρ, ##SSAValue##55##, time, Val{(true, true)}())
    Σ = ##SSAValue##56##
    ##SSAValue##58## = Core.tuple(Y₁, Y₂)
    ##SSAValue##59## = Core.tuple(μ₁, μ₂)
    (##SSAValue##60##, ###adjoint###_##∂##SSAValue##60####∂##SSAValue##59####) = PaddedMatrices.∂getindex(##SSAValue##59##, AvailableData)
    (##SSAValue##61##, ###adjoint###_##∂##SSAValue##61####∂Σ##) = PaddedMatrices.∂getindex(Σ, AvailableData)
    (##SSAValue##62##, ###adjoint###_##∂##SSAValue##62####∂##SSAValue##60####, ###adjoint###_##∂##SSAValue##62####∂##SSAValue##61####) = ProbabilityDistributions.∂Normal(##SSAValue##58##, ##SSAValue##60##, ##SSAValue##61##, Val{(false, true, true)}())
    ##SSAValue##63## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##62##)
    target = ##SSAValue##63##
end
```
It also prepends the corresponding operations to a second expression for the reverse diff pass:
```julia
striplines(reverse_pass)
```
yielding:
```julia
quote
    ###seed#####SSAValue##63## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##63##)
    ###seed#####SSAValue##62## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##63##, ###seed#####SSAValue##62##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##63##, ###seed###target)
    ###seed#####SSAValue##61## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##62##, ###adjoint###_##∂##SSAValue##62####∂##SSAValue##61####, ###seed#####SSAValue##61##)
    ###seed#####SSAValue##60## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##62##, ###adjoint###_##∂##SSAValue##62####∂##SSAValue##60####, ###seed#####SSAValue##60##)
    ###seed###Σ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##61##, ###adjoint###_##∂##SSAValue##61####∂Σ##, ###seed###Σ)
    ###seed#####SSAValue##59## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##60##, ###adjoint###_##∂##SSAValue##60####∂##SSAValue##59####, ###seed#####SSAValue##59##)
    ###seed###μ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##59##[2], ###seed###μ₂)
    ###seed###μ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##59##[1], ###seed###μ₁)
    ###seed#####SSAValue##56## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###Σ, ###seed#####SSAValue##56##)
    ###seed#####SSAValue##55## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##56##, ###adjoint###_##∂##SSAValue##56####∂##SSAValue##55####, ###seed#####SSAValue##55##)
    ###seed###ρ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##56##, ###adjoint###_##∂##SSAValue##56####∂ρ##, ###seed###ρ)
    (###adjoint###_##∂##SSAValue##55####∂##SSAValue##54####, ###adjoint###_##∂##SSAValue##55####∂L##) = ProbabilityModels.∂mul(##SSAValue##54##, L, Val{(true, true)}())
    ###seed###L = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##55##, ###adjoint###_##∂##SSAValue##55####∂L##, ###seed###L)
    ###seed#####SSAValue##54## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##55##, ###adjoint###_##∂##SSAValue##55####∂##SSAValue##54####, ###seed#####SSAValue##54##)
    ###seed###σ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##54##, ###seed###σ)
    ###seed#####SSAValue##52## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###μ₂, ###seed#####SSAValue##52##)
    ###seed#####SSAValue##51## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##52##, ###adjoint###_##∂##SSAValue##52####∂##SSAValue##51####, ###seed#####SSAValue##51##)
    ###seed###θ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##51##, ###adjoint###_##∂##SSAValue##51####∂θ##, ###seed###θ)
    ###seed###κ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##51##, ###adjoint###_##∂##SSAValue##51####∂κ##, ###seed###κ)
    ###seed###β₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##51##, ###adjoint###_##∂##SSAValue##51####∂β₂##, ###seed###β₂)
    ###seed#####SSAValue##49## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###μ₁, ###seed#####SSAValue##49##)
    ###seed#####SSAValue##48## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##49##, ###adjoint###_##∂##SSAValue##49####∂##SSAValue##48####, ###seed#####SSAValue##48##)
    ###seed###θ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##48##, ###adjoint###_##∂##SSAValue##48####∂θ##, ###seed###θ)
    ###seed###κ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##48##, ###adjoint###_##∂##SSAValue##48####∂κ##, ###seed###κ)
    ###seed###β₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##48##, ###adjoint###_##∂##SSAValue##48####∂β₁##, ###seed###β₁)
    ###seed#####SSAValue##46## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###β₂, ###seed#####SSAValue##46##)
    ###seed###σᵦ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##46##, ###adjoint###_##∂##SSAValue##46####∂σᵦ##, ###seed###σᵦ)
    ###seed###μᵦ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##46##, ###adjoint###_##∂##SSAValue##46####∂μᵦ₂##, ###seed###μᵦ₂)
    ###seed###βᵣ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##46##, ###adjoint###_##∂##SSAValue##46####∂βᵣ₂##, ###seed###βᵣ₂)
    ###seed#####SSAValue##44## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###β₁, ###seed#####SSAValue##44##)
    ###seed###σᵦ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##44##, ###adjoint###_##∂##SSAValue##44####∂σᵦ##, ###seed###σᵦ)
    ###seed###μᵦ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##44##, ###adjoint###_##∂##SSAValue##44####∂μᵦ₁##, ###seed###μᵦ₁)
    ###seed###βᵣ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##44##, ###adjoint###_##∂##SSAValue##44####∂βᵣ₁##, ###seed###βᵣ₁)
    ###seed#####SSAValue##42## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##42##)
    ###seed#####SSAValue##41## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##42##, ###seed#####SSAValue##41##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##42##, ###seed###target)
    ###seed###βᵣ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##41##, ###adjoint###_##∂##SSAValue##41####∂βᵣ₂##, ###seed###βᵣ₂)
    ###seed#####SSAValue##39## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##39##)
    ###seed#####SSAValue##38## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##39##, ###seed#####SSAValue##38##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##39##, ###seed###target)
    ###seed###βᵣ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##38##, ###adjoint###_##∂##SSAValue##38####∂βᵣ₁##, ###seed###βᵣ₁)
    ###seed#####SSAValue##36## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##36##)
    ###seed#####SSAValue##35## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##36##, ###seed#####SSAValue##35##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##36##, ###seed###target)
    ###seed###σᵦ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##35##, ###adjoint###_##∂##SSAValue##35####∂σᵦ##, ###seed###σᵦ)
    ###seed#####SSAValue##33## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###μᵦ₂, ###seed#####SSAValue##33##)
    ###seed###σₕ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##33##, ###adjoint###_##∂##SSAValue##33####∂σₕ##, ###seed###σₕ)
    ###seed###μₕ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##33##, ###adjoint###_##∂##SSAValue##33####∂μₕ₂##, ###seed###μₕ)
    ###seed###μᵣ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##33##, ###adjoint###_##∂##SSAValue##33####∂μᵣ₂##, ###seed###μᵣ₂)
    ###seed#####SSAValue##31## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###μᵦ₁, ###seed#####SSAValue##31##)
    ###seed###σₕ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##31##, ###adjoint###_##∂##SSAValue##31####∂σₕ##, ###seed###σₕ)
    ###seed###μₕ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##31##, ###adjoint###_##∂##SSAValue##31####∂μₕ₁##, ###seed###μₕ₁)
    ###seed###μᵣ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##31##, ###adjoint###_##∂##SSAValue##31####∂μᵣ₁##, ###seed###μᵣ₁)
    ###seed#####SSAValue##29## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##29##)
    ###seed#####SSAValue##28## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##29##, ###seed#####SSAValue##28##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##29##, ###seed###target)
    ###seed###μᵣ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##28##, ###adjoint###_##∂##SSAValue##28####∂μᵣ₂##, ###seed###μᵣ₂)
    ###seed#####SSAValue##26## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##26##)
    ###seed#####SSAValue##25## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##26##, ###seed#####SSAValue##25##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##26##, ###seed###target)
    ###seed###μᵣ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##25##, ###adjoint###_##∂##SSAValue##25####∂μᵣ₁##, ###seed###μᵣ₁)
    ###seed#####SSAValue##23## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##23##)
    ###seed#####SSAValue##22## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##23##, ###seed#####SSAValue##22##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##23##, ###seed###target)
    ###seed###σₕ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##22##, ###adjoint###_##∂##SSAValue##22####∂σₕ##, ###seed###σₕ)
    ###seed#####SSAValue##20## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##20##)
    ###seed#####SSAValue##19## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##20##, ###seed#####SSAValue##19##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##20##, ###seed###target)
    ###seed###μₕ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##19##, ###adjoint###_##∂##SSAValue##19####∂μₕ₂##, ###seed###μₕ₂)
    ###seed#####SSAValue##17## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##17##)
    ###seed#####SSAValue##16## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##17##, ###seed#####SSAValue##16##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##17##, ###seed###target)
    ###seed###μₕ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##16##, ###adjoint###_##∂##SSAValue##16####∂μₕ₁##, ###seed###μₕ₁)
    ###seed#####SSAValue##14## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##14##)
    ###seed#####SSAValue##13## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##14##, ###seed#####SSAValue##13##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##14##, ###seed###target)
    ###seed###L = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##13##, ###adjoint###_##∂##SSAValue##13####∂L##, ###seed###L)
    ###seed#####SSAValue##11## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##11##)
    ###seed#####SSAValue##10## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##11##, ###seed#####SSAValue##10##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##11##, ###seed###target)
    ###seed###θ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##10##, ###adjoint###_##∂##SSAValue##10####∂θ##, ###seed###θ)
    ###seed#####SSAValue##8## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##8##)
    ###seed#####SSAValue##7## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##8##, ###seed#####SSAValue##7##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##8##, ###seed###target)
    ###seed###σ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##7##, ###adjoint###_##∂##SSAValue##7####∂σ##, ###seed###σ)
    ###seed#####SSAValue##5## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##5##)
    ###seed#####SSAValue##4## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##5##, ###seed#####SSAValue##4##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##5##, ###seed###target)
    ###seed###κ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##4##, ###adjoint###_##∂##SSAValue##4####∂κ##, ###seed###κ)
    ###seed#####SSAValue##2## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##2##)
    ###seed#####SSAValue##1## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##2##, ###seed#####SSAValue##1##)
    ###seed###ρ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##1##, ###adjoint###_##∂##SSAValue##1####∂ρ##, ###seed###ρ)
end
```

The `RESERVED_INCREMENT_SEED_RESERVED` and `RESERVED_MULTIPLY_SEED_RESERVED` fall back to `mulladd` and `*` methods, respectively. They however allow for different overloads.

The keys to performance are that this source-to-source reverse mode differentation avoids unnecessary overhead, and does not interfere with vectorization, which typical dual-number based approaches, like Stan's var type or ForwardDiff's duals, do.
The reverse diff pass uses [DiffRules.jl](https://github.com/JuliaDiff/DiffRules.jl) as well as custom derivatives defined for probability distributions and a few other special functions.

By defining high level derivatives for commonly used functions we can optimize them much further than if we relied on autodiff for them. Additionally, we abuse multiple dispatch here: whenevever there is some structure we can exploit in the adjoint, we return a type with `INCREMENT/MULTIPLY` methods defined to exploit it. Examples include [BlockDiagonalColumnView](https://github.com/chriselrod/StructuredMatrices.jl/blob/master/src/block_diagonal.jl#L4)s or various [Reducer](https://github.com/chriselrod/ProbabilityModels.jl/blob/master/src/adjoints.jl#L55) types.

Another optimization is that it performs a [stack pointer pass](https://github.com/chriselrod/PaddedMatrices.jl/blob/master/src/stack_pointer.jl#L55). Supported functions are passed a pointer to a preallocated block of memory. This memory reuse can both give us considerable savings, and allow the memory to stay hot in the cache. [PaddedMatrices.jl](https://github.com/chriselrod/PaddedMatrices.jl), aside from providing [optimized matrix operations](https://bayeswatch.org/2019/06/06/small-matrix-multiplication-performance-shootout/) (contrary to the name, it's performance advantage over other libraries is actually greatest when none of them use padding), it also provides `PtrArray` (parameterized by size) and `DynamicPtrArray` (dynamically sized) types for taking advantage of this preallocated memory.





