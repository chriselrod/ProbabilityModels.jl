
using ProbabilityModels
sg = quote
    σ ~ Gamma(1.0, 0.05)
    L ~ LKJ(2.0)
    β ~ Normal(10.0) # μ = 0
    μ ~ Normal(100.0)

    σL = Diagonal(σ) * L
    Y₁ ~ Normal( X₁, β,       μ',  σL )
    Y₂ ~ Normal( X₂, β[1:7,:], μ', σL )
    # Y₁ ~ Normal( μ' .+ X₁ .*ˡ β,        σL )
    # Y₂ ~ Normal( μ' .+ X₂ .*ˡ β[1:7,:], σL )
end;
m = ProbabilityModels.read_model(sg, Main);
ProbabilityModels.ReverseDiffExpressions.lower(m)
dm = ProbabilityModels.ReverseDiffExpressions.differentiate(m)



@model SubGroup begin
    σ ~ Gamma(1.0, 0.05)
    L ~ LKJ(2.0)
    β ~ Normal(10.0) # μ = 0
    μ ~ Normal(100.0)

    σL = Diagonal(σ) * L
    Y₁ ~ Normal( μ' .+ X₁ .*ˡ β,        σL )
    Y₂ ~ Normal( μ' .+ X₂ .*ˡ β[1:7,:], σL )
end


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


truth = generate_true_parameters(domains, times, κ₀);

data = sample_data((100,100), truth, availabledata);
# mdata = sample_data((100,100), truth, availabledata, ((Val(10),Val(9))));


a = randn(length(data)); g = similar(a);

@time logdensity(data, a)
@time logdensity_and_gradient(data, a)


@time ProbabilityModels.check_gradient(data, a)


using BenchmarkTools

@benchmark logdensity($data, $a)
@benchmark logdensity_and_gradient!($g, $data, $a)



pg = PtrVector{73,Float64,73}(pointer(g));
pa = PtrVector{73,Float64,73}(pointer(a));
@code_warntype logdensity_and_gradient!(pg, data, pa, ProbabilityModels.STACK_POINTER_REF[])



using MacroTools, PaddedMatrices, DiffRules, VectorizationBase, SLEEFPirates
using MacroTools: striplines, @capture, prewalk, postwalk, @q

using LogDensityProblems, DynamicHMC, VectorizedRNG


includet("/home/chriselrod/.julia/dev/DistributionParameters/src/uniform_mapped_parameters.jl")
includet("/home/chriselrod/Documents/progwork/julia/autodiff_work/distribution_functions.jl")
includet("/home/chriselrod/Documents/progwork/julia/autodiff_work/special_diff_rules.jl")
includet("/home/chriselrod/Documents/progwork/julia/autodiff_work/model_macro_passes.jl")

# q = @q begin
#     β₀ ~ Normal(μ₀, σ₀)
#     β₁ ~ Normal(μ₁, σ₁)
#     y ~ Bernoulli_logit(β₀ + x * β₁)
# end
#
# q2 = translate_sampling_statements(q)
# q3 = flatten_expression(q2)

model_parameters = Set([:β₀,  :β₁])
ProbabilityModels.first_updates_to_assignemnts(q3, varnames)

q = @q begin
    β₀ ~ Normal(μ₀, σ₀)
    β₁ ~ Normal(μ₁, σ₁)
    y ~ Bernoulli_logit(β₀ + x * β₁)
end
model_name = :BernoulliLogitModel
variable_set = ProbabilityModels.determine_variables(q)
variables = [v for v ∈ variable_set] # ensure order is constant
variable_type_names = [Symbol("##Type##", v) for v ∈ variables]
struct_quote = quote
    struct $model_name{$(variable_type_names...)} <: LogDensityProblems.AbstractLogDensityProblem
        $([:( $(variables[i])::$(variable_type_names[i]) ) for i ∈ eachindex(variables)]...)
    end
end
struct_kwarg_quote = quote
    function $model_name(; $(variables...))
        $model_name($([:(ProbabilityModels.types_to_vals($v)) for v ∈ variables]...))
    end
end

# Translate the sampling statements, and then flatten the expression to remove nesting.
expr = ProbabilityModels.translate_sampling_statements(q) |>
            ProbabilityModels.flatten_expression
first_pass = quote end
second_pass = quote end
tracked_vars = Set(model_parameters)
first_pass, name_dict = ProbabilityModels.rename_assignments(first_pass)
expr, name_dict = ProbabilityModels.rename_assignments(expr, name_dict)


θ_sym = $(QuoteNode(θ)) # This creates our symbol θ

second_pass, name_dict = ProbabilityModels.rename_assignments(second_pass, name_dict)
ProbabilityModels.reverse_diff_pass!(first_pass, second_pass, expr, tracked_vars)
# variable renaming rather than incrementing makes initiazing
# target to an integer okay.
expr_out = quote
    # target = 0
    $first_pass
    $(Symbol("###seed###", name_dict[:target])) = ProbabilityModels.One()
    $second_pass
    (
        $(name_dict[:target]),
        $(Expr(:tuple, [Symbol("###seed###", mp) for mp ∈ model_parameters]...))
    )
end




using ProbabilityModels, MacroTools, PaddedMatrices, DiffRules, VectorizationBase, SLEEFPirates
using DistributionParameters
using MacroTools: striplines, @capture, prewalk, postwalk, @q

using LogDensityProblems, DynamicHMC, VectorizedRNG


# model_parameters = Set([:β₀,  :β₁])
# ProbabilityModels.first_updates_to_assignemnts(q3, varnames)

q = @q begin
    β₀ ~ Normal(μ₀, σ₀)
    β₁ ~ Normal(μ₁, σ₁)
    y ~ Bernoulli_logit(β₀ + x * β₁)
end


# variable_set = ProbabilityModels.determine_variables(expr)
# variables = [v for v ∈ variable_set] # ensure order is constant
# variable_type_names = [Symbol("##Type##", v) for v ∈ variables]


q2 = ProbabilityModels.translate_sampling_statements(q)
expr = ProbabilityModels.flatten_expression(q2)
return_partials = true
model_parameters = Symbol[]
first_pass = quote end
second_pass = quote end

push!(model_parameters, :β₀)
DistributionParameters.load_parameter(first_pass.args, second_pass.args, :β₀, RealFloat, return_partials)
push!(model_parameters, :β₁)
DistributionParameters.load_parameter(first_pass.args, second_pass.args, :β₁, RealVector{4,Float64}, return_partials)

tracked_vars = Set(model_parameters)
first_pass, name_dict = ProbabilityModels.rename_assignments(first_pass)
expr, name_dict = ProbabilityModels.rename_assignments(expr, name_dict)

second_pass, name_dict = ProbabilityModels.rename_assignments(second_pass, name_dict)
TLθ = 5 #type_length($θ) # This refers to the type of the input
ProbabilityModels.reverse_diff_pass!(first_pass, second_pass, expr, tracked_vars)
first_pass
second_pass

combined_q = quote
    $first_pass
    $(Symbol("###seed###", name_dict[:target])) = ProbabilityModels.One()
    $second_pass
end;

ProbabilityModels.first_updates_to_assignemnts(combined_q, model_parameters)


T_sym = gensym(:T); θ_sym = gensym(:θ);
expr_out = quote
    # target = zero($T_sym)
    $(Symbol("##θparameter##")) = VectorizationBase.vectorizable($θ_sym)
    $first_pass
    $(Symbol("##∂θparameter##m")) = PaddedMatrices.MutableFixedSizeVector{$TLθ,$T_sym}(undef)
    $(Symbol("##∂θparameter##")) = VectorizationBase.vectorizable($(Symbol("##∂θparameter##m")))
    $(Symbol("###adjoint###", name_dict[:target])) = ProbabilityModels.One()
    $second_pass
    LogDensityProblems.ValueGradient(
        $(name_dict[:target]),
        PaddedMatrices.ConstantFixedSizeVector($(Symbol("##∂θparameter##m")))
    )
end;
final_expr = first_updates_to_assignemnts(expr_out, model_parameters)



using MacroTools, PaddedMatrices, DiffRules, VectorizationBase, SLEEF, Random, SIMDPirates
using MacroTools: striplines, @capture, prewalk, postwalk, @q
using LogDensityProblems, DynamicHMC, VectorizedRNG, RandomNumbers

includet("/home/chriselrod/.julia/dev/DistributionParameters/src/uniform_mapped_parameters.jl")
includet("/home/chriselrod/Documents/progwork/julia/autodiff_work/distribution_functions.jl")
includet("/home/chriselrod/Documents/progwork/julia/autodiff_work/special_diff_rules.jl")
includet("/home/chriselrod/Documents/progwork/julia/autodiff_work/model_macro_passes.jl")
includet("/home/chriselrod/.julia/dev/ProbabilityModels/src/dynamic_hmc_interface.jl")





using ProbabilityModels, DistributionParameters, LogDensityProblems, DynamicHMC
using VectorizationBase, LoopVectorization
@model BernoulliLogitModel begin
    β₀ ~ Normal(μ₀, σ₀)
    β₁ ~ Normal(μ₁, σ₁)
    y ~ Bernoulli_logit(β₀ + x * β₁)
end




# using StaticArrays
N = 800; N_β = 4;
X = randn(N, N_β);
β₁ = [-1.595526740808615, -1.737875659746032, -0.26107993378119343, 0.6500851571519769];
β₀ = -0.05
Xβ₁ = X * β₁;
p = rand(N);
y = @. p < 1 / (1 + exp( - Xβ₁ - β₀));
sum(y)

ℓ = BernoulliLogitModel(
    σ₀ = 10.0, σ₁ = 5.0, μ₀ = 0.0, μ₁ = 0.0,
    β₀ = RealFloat, β₁ = RealVector{4},
    y = y, x = X
);


# eval(θquote)
dimension(ℓ)
a = fill(1.0, dimension(ℓ));
logdensity(LogDensityProblems.ValueGradient, ℓ, a)

using LogDensityProblems, DynamicHMC
@time mcmc_chain, tuned_sampler = NUTS_init_tune_mcmc_default(ℓ, 4000);
sample_mean(mcmc_chain)
sample_cov(mcmc_chain)



using BenchmarkTools
@benchmark logdensity(LogDensityProblems.ValueGradient, $ℓ, $a)
@benchmark NUTS_init_tune_mcmc_default($ℓ, 4000, report = DynamicHMC.ReportSilent())


struct_quote, struct_kwarg_quote, directquote, thetaquote, dimquote = generate_generated_funcs_expressions(:BernoulliLogitModel, q);
using LogDensityProblems, DynamicHMC
eval(struct_quote)
eval(struct_kwarg_quote)
eval(thetaquote)
eval(dimquote)
@time mcmc_chain, tuned_sampler = NUTS_init_tune_mcmc_default(rng, ℓ1, 4000);
sample_mean(mcmc_chain)
sample_cov(mcmc_chain)

@benchmark NUTS_init_tune_mcmc($rng, $ℓ1, 4000, report = DynamicHMC.ReportSilent())


mcmc_chainv = reshape(reinterpret(Float64, get_position.(mcmc_chain)), (8,length(mcmc_chain)))[1:5,:];


@code_warntype NUTS_init_tune_mcmc(rng, ℓ1, 4000, report = DynamicHMC.ReportSilent())
# Not type stable?

@code_warntype DynamicHMC.NUTS_init(rng, ℓ1, report = DynamicHMC.ReportSilent())
sampler_init = DynamicHMC.NUTS_init(rng, ℓ1, report = DynamicHMC.ReportSilent())
tuners = DynamicHMC.bracketed_doubling_tuner()
@code_warntype DynamicHMC.tune(sampler_init, tuners)
sampler_tuned = DynamicHMC.tune(sampler_init, DynamicHMC.bracketed_doubling_tuner());
@code_warntype DynamicHMC.mcmc(sampler_tuned, 4000)
mcmcres = DynamicHMC.mcmc(sampler_tuned, 4000)

function iterator_test(N)
    out = 0.0
    for i ∈ 1:4:N
        out += 2.3i
    end
    out
end

GK1 = GaussianKE(dimension(ℓ1));
H = DynamicHMC.Hamiltonian(ℓ1, GK1);
q = randn(rng, dimension(ℓ1));
p = rand(rng, GK1);
z = DynamicHMC.phasepoint_in(H, q, p)
DynamicHMC.find_initial_stepsize(DynamicHMC.InitialStepsizeSearch(), H, z)
@code_warntype DynamicHMC.NUTS_init(rng, ℓ1, report = false)

sampler_init = DynamicHMC.NUTS_init(rng, ℓ1);
tuners = DynamicHMC.bracketed_doubling_tuner()
@which DynamicHMC.tune(sampler_init,tuners.tuners[1])

sample, A = DynamicHMC.mcmc_adapting_ϵ(sampler_init, tuners.tuners[1].N)


function sample_sum(sample)
    N = length(sample)
    x̄ = DynamicHMC.get_position(sample[1])
    @inbounds for n ∈ 2:N
        x̄ += DynamicHMC.get_position(sample[n])
    end
    x̄
end
sample_v = get_position.(sample);
sample_mat = reshape(reinterpret(Float64, sample_v), (8,75))[1:5,:];



using ForwardDiff, Optim
inv_logit(x) = 1 / (1 + exp(-x))
function Bernoulli_logit_fmadd_simple(y, X, β, α)
    p = inv_logit.( X * β .+ α )
    T = eltype(p)
    target = zero(T)
    @inbounds @simd ivdep for i ∈ eachindex(p, y)
        OmP = one(T) - p[i]
        target += y[i] ? log(p[i]) : log(OmP)
    end
    target
end
opt = optimize(b -> -Bernoulli_logit_fmadd_simple(y, X, b, 1.0), ones(4), BFGS())


βones =  fill(1.0, PaddedMatrices.Static{4}())
∂Bernoulli_logit_fmadd_logeval_dropconst(y, X, βones, 1.0, Val{(false,false,true,true)}())

Bernoulli_logit_fmadd_simple(y, X, ones(4), 1.0)
ForwardDiff.derivative(a -> Bernoulli_logit_fmadd_simple(y, X, ones(4), a), 1.0)
ForwardDiff.gradient(b -> Bernoulli_logit_fmadd_simple(y, X, b, 1.0), ones(4))
Normal_logeval_dropconst(βones, 0.0, 5.0, Val{(true,false,false)}())



using ProbabilityModels, DistributionParameters, ProbabilityDistributions,
    LoopVectorization, LinearAlgebra,
    LogDensityProblems, SLEEFPirates, SIMDPirates, StructuredMatrices, ScatteredArrays, PaddedMatrices
using DistributionParameters: LKJ_Correlation_Cholesky, RealFloat, PositiveFloat, UnitFloat, RealVector, PositiveVector
using ProbabilityModels: HierarchicalCentering, ∂HierarchicalCentering, ITPExpectedValue, ∂ITPExpectedValue

@model ITPModel begin

    # Non-hierarchical Priors
    (0.5ρ + 0.5) ~ Beta(2, 2)
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

    U = inv′(Diagonal(σ) * L)

    # Likelihood
    μ₁ = ITPExpectedValue(t, β₁, κ, θ)
    μ₂ = ITPExpectedValue(t, β₂, κ, θ)
    #AR = AutoregressiveMatrix(2ρ-1, δₜ)
    AR = AutoregressiveMatrix(ρ, δₜ)
    Y₁ ~ Normal(μ₁, AR, U)
    Y₂ ~ Normal(μ₂, AR, U)

end

# mub1 = HierarchicalCentering(xc.μᵣ₁, xc.μₕ₁, xc.σₕ);
# mub2 = HierarchicalCentering(xc.μᵣ₂, xc.μₕ₂, xc.σₕ);
# b1 = HierarchicalCentering(xc.βᵣ₁, mub1, xc.σᵦ, domains);
# b2 = HierarchicalCentering(xc.βᵣ₂, mub2, xc.σᵦ, domains);
# mu1 = ITPExpectedValue(times, b1, xc.κ, xc.θ);
# mu2 = ITPExpectedValue(times, b2, xc.κ, xc.θ);
#
# ProbabilityDistributions.Normal(Y₁, mu1, AutoregressiveMatrix(xc.ρ, δₜ), sLinv, Val((false,true,true,true)))
# ProbabilityDistributions.Normal(Y₂, mu2, AutoregressiveMatrix(xc.ρ, δₜ), sLinv, Val((false,true,true,true)))


#     Defined model: ITPModel.
#     Unknowns: Y₂, domains, μₕ₂, δt, μᵣ₁, βₖ, η₀, αₖ, σᵦ, θ, μᵣ₂, ρ, σₕ, μₕ₁, κ, L, Y₁, βᵣ₂, t, βᵣ₁.

using ProbabilityModels, DistributionParameters, ProbabilityDistributions,
    LoopVectorization, LinearAlgebra, Random, LogDensityProblems, SLEEFPirates, SIMDPirates, StructuredMatrices, ScatteredArrays, PaddedMatrices
using DistributionParameters: LKJ_Correlation_Cholesky, RealFloat, PositiveFloat, UnitFloat, RealVector, PositiveVector
using ProbabilityModels: HierarchicalCentering, ∂HierarchicalCentering, ITPExpectedValue, ∂ITPExpectedValue
BLAS.set_num_threads(1)

domains = ProbabilityModels.Domains(2,2,2,3);
# domains = ProbabilityModels.Domains(2,2,3);
T = 24; K = sum(domains); D = length(domains);


κ = (1/32) * reduce(+, (@Constant randexp(K)) for i ∈ 1:8) # κ ~ Gamma(8, 32)
σd = sum(@Constant randexp(4)) / 16 # σd ~ Gamma(4,16)
θ = 2.0 * (@Constant randn(K)) # θ ~ Normal(0,2)
S = (@Constant randn(K,4K)) |> x -> x * x'
S *= (1/16)
pS = StructuredMatrices.SymmetricMatrixL(S)
L = PaddedMatrices.chol(S); U = PaddedMatrices.invchol(pS)
muh1, muh2 = -3.0, 9.0
m01 = muh1 + (@Constant randn(D)); # placebo
m02 = muh2 + (@Constant randn(D)); #treatment
b1 = HierarchicalCentering((@Constant randn(K)), m01, σd, domains); # placebo
b2 = HierarchicalCentering((@Constant randn(K)), m02, σd, domains); # treatment

δₜ = 0.06125 *  reduce(+, (@Constant randexp(T-1)) for i ∈ 1:8);
t = vcat(zero(ConstantFixedSizeVector{1,Float64}), cumsum(δₜ));
mu1 = ProbabilityModels.ITPExpectedValue(t, b1, κ, θ);
mu2 = ProbabilityModels.ITPExpectedValue(t, b2, κ, θ);

ρ = 0.7; ARmat = StructuredMatrices.AutoregressiveMatrix(ρ, δₜ);
# ARcholinv = ConstantFixedSizeMatrix(ARmat);
ARchol = PaddedMatrices.chol(ConstantFixedSizeMatrix(ARmat));


pσU = MutableFixedSizeMatrix{K,K,Float64}(undef); pσU .= L'; cσU = ConstantFixedSizeMatrix(pσU);
Y1 = [ARchol * (@Constant randn(T, K)) * cσU + mu1 for n in 1:56];
Y2 = [ARchol * (@Constant randn(T, K)) * cσU + mu2 for n in 1:56];
Y1c = ChunkedArray(Y1);
Y2c = ChunkedArray(Y2);

mu1a = Array(mu1);
ARcholinva = inv(Array(ARchol)); Ua = Array(U);
-0.5*sum(A -> sum(abs2,ARcholinva * (Array(A) .- mu1a) * Ua), Y1)
ProbabilityDistributions.Normal(Y1c, mu1, ARmat, U, Val((false,true,false,false)))
ProbabilityDistributions.∂Normal(Y1c, mu1, ARmat, U, Val((false,true,false,false)))
ProbabilityDistributions.Normal(Y1c, mu1, ARmat, U, Val((false,true,true,true)))
ProbabilityDistributions.∂Normal(Y1c, mu1, ARmat, U, Val((false,true,true,true)))

using BenchmarkTools
@benchmark ProbabilityDistributions.Normal($Y1c, $mu1, $ARmat, $U, Val((false,true,true,true)))
@benchmark ProbabilityDistributions.∂Normal($Y1c, $mu1, $ARmat, $U, Val((false,true,true,true)))


Y1r = reshape(reinterpret(Float64, Y1), (T, K, length(Y1)));
Y1p = permutedims(Y1r, (2,3,1));
Y1p2 = Array{Float64,3}(undef, K,length(Y1),T);
Y1p3 = Array{Float64,3}(undef, K,length(Y1),T);
mu1a2 = reshape(Array(mu1)', (K,1,T));
ARmata = Bidiagonal(vcat(1.0,Array(ARmat.spacing.ρᵗ)), -1.0 .* ARmat.spacing.rinvOmρ²ᵗ .* ARmat.spacing.ρᵗ, :U)
function lpdfn(Y1p3::AbstractArray{Tf,3}, Y1p2::AbstractArray{Tf,2}, Y1p, mu1, ARmat, U) where {Tf}
    @inbounds Y1p3 .= Y1p .- mu1
    K,N,T = size(Y1p3)
    ρᵗ = ARmat.spacing.ρᵗ
    rinvOmρ²ᵗ = ARmat.spacing.rinvOmρ²ᵗ
    Y1p3_2 = reshape(Y1p3, (K,N*T))
    Y1p2_3 = reshape(Y1p2, (K,N,T))
    @inbounds Y1p2_3[:,:,1] .= Y1p3[:,:,1]
    Y1p3_t2 = reshape(Y1p3, (K*N,T))
    Y1p2_t2 = reshape(Y1p2, (K*N,T))
    @inbounds for t ∈ 1:T-1
        rt = ρᵗ[t]
        rit = rinvOmρ²ᵗ[t]
        @fastmath @simd ivdep for i ∈ 1:K*N
            Y1p2_t2[i,t+1] =  rit * (Y1p3_t2[i,t+1] - rt* Y1p3_t2[i,t])
        end
    end
    mul!(Y1p3_2, U', Y1p2)
    -0.5sum(abs2, Y1p3_2)
end
lpdfn(Y1p3, Y1p2, Y1p, mu1a2, ARmat, Ua)


Y1r = reshape(reinterpret(Float64, Y1), (T, K, length(Y1)));
Y1p = permutedims(Y1r, (1,3,2));
Y1p2 = Array{Float64,3}(undef, T,length(Y1),K);
Y1p3 = Array{Float64,3}(undef, T,length(Y1),K);
mu1a2 = reshape(Array(mu1), (T,1,K));
Ua = Array(U);
ARmata = Bidiagonal(vcat(1.0,Array(ARmat.spacing.rinvOmρ²ᵗ)), -1.0 .* ARmat.spacing.rinvOmρ²ᵗ .* ARmat.spacing.ρᵗ, :L)
function lpdfn2(Y1p3::AbstractArray{Tf,3}, Y1p2::AbstractArray{Tf,3}, Y1p, mu1, ARmat, U) where {Tf}
    T,N,K = size(Y1p3)
    @inbounds Y1p3 .= Y1p .- mu1
    mul!(reshape(Y1p2,(T,N*K)), ARmat, reshape(Y1p3,(T,N*K)))
    mul!(reshape(Y1p3,(T*N,K)), reshape(Y1p2,(T*N,K)), U)
    N*(T*logdet(UpperTriangular(U)) + K*sum(log, ARmat.dv)) - 0.5dot(Y1p3, Y1p3)
end
lpdfn2(Y1p3, Y1p2, Y1p, mu1a2, ARmata, Ua)

Y1c2 = ChunkedArray(Y1);
using VectorizationBase
function mul2!(Y1c2, Y1c, U)
    vY1c = VectorizationBase.vectorizable(Y1c)
    vY1c2 = VectorizationBase.vectorizable(Y1c2)
    V = Vec{8,Float64}
    for i ∈ 0:(length(Y1c)>>3)-1
        mul!(vload(V, vY1c2+8i), vload(V,vY1c + 8i), U)
    end
end
Y1pm = reshape(Y1p, (size(Y1p,1)*size(Y1p,2),size(Y1p,3)));
Y1pmUa = similar(Y1pm);
@benchmark mul!($Y1pmUa, $Y1pm, $Ua)
@benchmark mul2!($Y1c2, $Y1c, $U)

ℓ = ITPModel(
    domains = domains, Y₁ = Y1c, Y₂ = Y2c, t = t, δₜ = δₜ,
    L = LKJ_Correlation_Cholesky{K}, ρ = UnitFloat,
    κ = PositiveVector{K}, θ = RealVector{K},
    μₕ₁ = RealFloat, μₕ₂ = RealFloat,
    μᵣ₁ = RealVector{D}, μᵣ₂ = RealVector{D},
    βᵣ₁ = RealVector{K}, βᵣ₂ = RealVector{K},
    σᵦ = PositiveFloat, σₕ = PositiveFloat,
    σ = PositiveVector{K}
    #η₀ = 2.0, αₖ = 0.1, βₖ = 0.1
);
## domains = (2,2,3)
# Unknowns: Y₂, domains, μₕ₂, μᵣ₁, σ, σᵦ, θ, μᵣ₂, ρ, σₕ, μₕ₁, κ, L, δₜ, Y₁, βᵣ₂, t, βᵣ₁.
# D = 3, K = 7
# μₕ₂ = 1, μᵣ = 2:4, σ = 5:11, σᵦ = 12, θ = 13:19, μᵣ₂ = 20:22, ρ = 23, σₕ = 24, μₕ₁ = 25, κ = 26:32
# L = 33:53,  βᵣ₂ = 54:60, βᵣ₁ = 61:67

## domains = (2,2,2,3)
# Unknowns: Y₂, domains, μₕ₂, μᵣ₁, σ, σᵦ, θ, μᵣ₂, ρ, σₕ, μₕ₁, κ, L, δₜ, Y₁, βᵣ₂, t, βᵣ₁.
# D = 4, K = 9
# μₕ₂ = 1, μᵣ = 2:5, σ = 6:14, σᵦ = 15, θ = 16:24, μᵣ₂ = 25:28, ρ = 29, σₕ = 30, μₕ₁ = 31, κ = 32:40
# L = 41:76,  βᵣ₂ = 77:85, βᵣ₁ = 86:94
## 33 through 39: wrong

##

using BenchmarkTools
dimension(ℓ)
# a = fill(1.0, dimension(ℓ));
a = randn(dimension(ℓ));
logdensity(LogDensityProblems.Value, ℓ, a)
vg = logdensity(LogDensityProblems.ValueGradient, ℓ, a)
vgg = vg.gradient;

@benchmark logdensity(LogDensityProblems.ValueGradient, $ℓ, $a)

function gradi(ℓ, a, a2, i)
    step = sqrt(eps(a[i]))
    # step = cbrt(eps(a[i]))
    v1 = logdensity(LogDensityProblems.Value, ℓ, a)
    a2[i] += step
    v2 = logdensity(LogDensityProblems.Value, ℓ, a2)
    a2[i] = a[i]
    (v2.value - v1.value) / step
end
function gradicompare(ℓ, vgg, a, a2, i)
    d1, d2 = (gradi(ℓ, a, a2, i), vgg[i])
    d1, d2, round((d1-d2)/d1, digits = 4)
end
function compare_grads(ℓ, a)
    a2 = MutableFixedSizeVector(a);
    vgg = logdensity(LogDensityProblems.ValueGradient, ℓ, a).gradient
    for i ∈ 1:dimension(ℓ)
       # ℓ, vgg, a, a2
       @show i, gradicompare(ℓ, vgg, a, a2, i)
    end
end
compare_grads(ℓ, a)
compare_grads(ℓ, randn(dimension(ℓ)))


using LogDensityProblems, DynamicHMC, MCMCDiagnostics
# @time mcmc_chain, tuned_sampler = NUTS_init_tune_mcmc_default(ℓ, 500);#, max_depth = 15);#, ϵ = 0.007);#, δ = 0.99);
@time mcmc_chain, tuned_sampler = NUTS_init_tune_mcmc_default(ℓ, 2000, δ = 0.75);#, max_depth = 15);#, ϵ = 0.007);#, δ = 0.99);
NUTS_statistics(mcmc_chain)
tuned_sampler
chain_matrix = get_position_matrix(mcmc_chain);
[effective_sample_size(chain_matrix[:,i]) for i in 1:20]'

@time mcmc_chain2, tuned_sampler2 = NUTS_init_tune_mcmc_default(ℓ, 2000, δ = 0.75);#, max_depth = 15);#, ϵ = 0.007);#, δ = 0.99);
NUTS_statistics(mcmc_chain2)
tuned_sampler2
chain_matrix2 = get_position_matrix(mcmc_chain2);
[effective_sample_size(chain_matrix2[:,i]) for i in 1:20]'

@time mcmc_chain3, tuned_sampler3 = NUTS_init_tune_mcmc_default(ℓ, 2000, δ = 0.75);#, max_depth = 15);#, ϵ = 0.007);#, δ = 0.99);
NUTS_statistics(mcmc_chain3)
tuned_sampler3
chain_matrix3 = get_position_matrix(mcmc_chain3);
[effective_sample_size(chain_matrix3[:,i]) for i in 1:20]'


using VectorizationBase, PaddedMatrices, StructuredMatrices, BenchmarkTools
δ = MutableFixedSizeMatrix{T,K,NTuple{8,Core.VecElement{Float64}},T,(T*K)}(undef);
δU = MutableFixedSizeMatrix{T,K,NTuple{8,Core.VecElement{Float64}},T,(T*K)}(undef);
Yᵥ = VectorizationBase.vectorizable(Y1c);
remmask = Y1c.mask;
@benchmark PaddedMatrices.vload!($δ, $Yᵥ + 0)
@benchmark PaddedMatrices.diff!($δ, $mu1, $δ)
@benchmark PaddedMatrices.mask!($δ, $remmask)
@benchmark mul!($δU, $δ, $U)


function chain_to_array(chain::Array{NUTS_Transition{ConstantFixedSizeArray{Tuple{N},Float64,1,Tuple{1},L},Float64},1}) where {N,L}
    pos = get_position.(chain)
    s = sizeof(eltype(pos)) >> 3
    reshape(reinterpret(Float64, pos), (s,length(pos)))[1:N,:]
end
function process_chain(chain)
    copy(chain_to_array(chain)')
end
c = process_chain(mcmc_chain);
ntuple(i -> effective_sample_size(c[:,i]), Val(16))
NUTS_statistics(mcmc_chain)
tuned_sampler


@time mcmc_chains = ProbabilityModels.NUTS_init_tune_threaded(ℓ, 1000, max_depth = 12);

@time mcmc_chains = ProbabilityModels.NUTS_init_tune_threaded(ℓ, 2000, max_depth = 12);

@time mcmc_chain, tuned_sampler = NUTS_init_tune_mcmc_default(ℓ, 1000, max_depth = 12);


report = ReportIO(countΔ = -1, time_nsΔ = -1);
@time mcmc_chain, tuned_sampler = NUTS_init_tune_mcmc_default(ℓ, 4000, report = report);



c1 = process_chain(mcmc_chains[2]);
# for i ∈ eachindex(mcmc_chains)
effective_sample_size(c1[:,1])
effective_sample_size(c1[:,2])
effective_sample_size(c1[:,3])
effective_sample_size(c1[:,4])


μₕ₁ = @Constant randn(4); # placebo
μₕ₂ = 2.0 + @Constant randn(4); #treatment
β₁ = vcat(ntuple(i -> (@Constant randn(domains[i])) + μₕ₁[i], Val(4))...); # placebo
β₂ = vcat(ntuple(i -> (@Constant randn(domains[i])) + μₕ₂[i], Val(4))...); # treatment

δt = @Constant rand(23); lastt = Ref(0.0);
t = ConstantFixedSizeVector{24,Float64}(ntuple(i -> i == 1 ? 0.0 : lastt[] += δt[i-1], Val(24)));

μ₁ = ITPExpectedValue(t, β₁, κ)
μ₂ = ITPExpectedValue(t, β₂, κ)

β = @CFixedSize randn(16);
κ = @CFixedSize randexp(16);

μ = ProbabilityModels.ITPExpectedValue(t, β, κ)

ℓitp = BernoulliLogitModel(
    α_κ = 1.0, δt = δtv, μ_β, η₀, σ_β, Y, β, ρ, α_ρ, κ, L, β_κ, t, β_ρ
);

# y =
