
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
    $(Symbol("##∂θparameter##m")) = PaddedMatrices.MutableFixedSizePaddedVector{$TLθ,$T_sym}(undef)
    $(Symbol("##∂θparameter##")) = VectorizationBase.vectorizable($(Symbol("##∂θparameter##m")))
    $(Symbol("###adjoint###", name_dict[:target])) = ProbabilityModels.One()
    $second_pass
    LogDensityProblems.ValueGradient(
        $(name_dict[:target]),
        PaddedMatrices.ConstantFixedSizePaddedVector($(Symbol("##∂θparameter##m")))
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
    ρ ~ Beta(2, 2)
    κ ~ Gamma(0.1, 0.1) # μ = 1, σ² = 10
    σ ~ Gamma(0.5, 0.1) # μ = 5, σ² = 50
    θ ~ Normal(10)
    L ~ LKJ(2.0)

    # Hierarchical Priors.
    # h subscript, for highest in the hierarhcy.
    μₕ₁ ~ Normal(5) # μ = 0
    μₕ₂ ~ Normal(5) # μ = 0
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
    AR = AutoregressiveMatrix(2ρ-1, δt)
    Y₁ ~ Normal(μ₁, AR, U)
    Y₂ ~ Normal(μ₂, AR, U)

end

#     Defined model: ITPModel.
#     Unknowns: Y₂, domains, μₕ₂, δt, μᵣ₁, βₖ, η₀, αₖ, σᵦ, θ, μᵣ₂, ρ, σₕ, μₕ₁, κ, L, Y₁, βᵣ₂, t, βᵣ₁.

using ProbabilityModels, DistributionParameters, ProbabilityDistributions,
    LoopVectorization,
    LogDensityProblems, SLEEFPirates, SIMDPirates, StructuredMatrices, ScatteredArrays, PaddedMatrices

domains = ProbabilityModels.Domains(3,4,4,5);
M = 24; N = sum(domains); Num_domains = length(domains);

ρ = 0.7;
κ = 0.25 * (@Constant randexp(N));
θ = 10.0 * (@Constant randn(N));
S = (@Constant randn(N,2N)) |> x -> x * x';
pS = StructuredMatrices.SymmetricMatrixL(S);
L = PaddedMatrices.chol(S); U = PaddedMatrices.invchol(pS);
m01 = @Constant randn(Num_domains); # placebo
m02 = 2.0 + @Constant randn(Num_domains); #treatment
b1 = vcat(ntuple(i -> (@Constant randn(domains[i])) + m01[i], Val(Num_domains))...); # placebo
b2 = vcat(ntuple(i -> (@Constant randn(domains[i])) + m02[i], Val(Num_domains))...); # treatment

δt = @Constant rand(M-1); lastt = Ref(0.0);
t = ConstantFixedSizePaddedVector{24,Float64}(ntuple(i -> i == 1 ? 0.0 : lastt[] += δt[i-1], Val(M)));
mu1 = ProbabilityModels.ITPExpectedValue(t, b1, κ, θ);
mu2 = ProbabilityModels.ITPExpectedValue(t, b2, κ, θ);

ARmat = StructuredMatrices.AutoregressiveMatrix(ρ, δt);
# ARcholinv = ConstantFixedSizePaddedMatrix(ARmat);
ARchol = PaddedMatrices.chol(ConstantFixedSizePaddedMatrix(ARmat));

Y1 = [ARchol * (@Constant randn(M, N)) * L' + mu1 for n in 1:120];
Y2 = [ARchol * (@Constant randn(M, N)) * L' + mu2 for n in 1:120];
Y1c = ChunkedArray(Y1);
Y2c = ChunkedArray(Y1);

mu1a = Array(mu1);
ARcholinva = inv(Array(ARchol)); Ua = Array(U);
-0.5*sum(A -> sum(abs2,ARcholinva * (Array(A) .- mu1a) * Ua), Y1)
ProbabilityDistributions.Normal(Y1c, mu1, ARmat, U, Val((false,true,false,false)))
ProbabilityDistributions.Normal(Y1c, mu1, ARmat, U, Val((false,true,true,true)))
# using BenchmarkTools
# @benchmark ProbabilityDistributions.Normal($Y1c, $mu1, $ARmat, $U, Val((false,true,true,true)))
# @benchmark ProbabilityDistributions.∂Normal($Y1c, $mu1, $ARmat, $U, Val((false,true,true,true)))

ℓ = ITPModel(
    domains = domains, Y₁ = Y1c, Y₂ = Y2c, t = t, δt = δt,
    L = LKJ_Correlation_Cholesky{N}, ρ = UnitFloat,
    κ = PositiveVector{N}, θ = RealVector{N},
    μₕ₁ = RealFloat, μₕ₂ = RealFloat,
    μᵣ₁ = RealVector{Num_domains}, μᵣ₂ = RealVector{Num_domains},
    βᵣ₁ = RealVector{N}, βᵣ₂ = RealVector{N},
    σᵦ = PositiveFloat, σₕ = PositiveFloat,
    σ = PositiveVector{N}
    #η₀ = 2.0, αₖ = 0.1, βₖ = 0.1
);
dimension(ℓ)
a = fill(1.0, dimension(ℓ));
logdensity(LogDensityProblems.Value, ℓ, a)
vg = logdensity(LogDensityProblems.ValueGradient, ℓ, a)
vgg = vg.gradient;

a2 = MutableFixedSizePaddedVector(a);
function gradi(ℓ, a, a2, i)
    v1 = logdensity(LogDensityProblems.Value, ℓ, a)
    a2[i] += sqrt(eps())
    v2 = logdensity(LogDensityProblems.Value, ℓ, a2)
    a2[i] = a[i]
    (v2.value - v1.value) / sqrt(eps())
end
gradi(ℓ, a, a2, 1)
gradicompare(ℓ, vgg, a, a2, i) = (gradi(ℓ, a, a2, i), vgg[i])
for i ∈ 1:213
   global ℓ, vgg, a, a2
   @show i, gradicompare(ℓ, vgg, a, a2, i)
end

using LogDensityProblems, DynamicHMC
@time mcmc_chain, tuned_sampler = ProbabilityModels.NUTS_init_tune_threaded(ℓ, 2000, max_depth = 12);

@time mcmc_chain, tuned_sampler = NUTS_init_tune_mcmc_default(ℓ, 4000);


report = ReportIO(countΔ = -1, time_nsΔ = -1);
@time mcmc_chain, tuned_sampler = NUTS_init_tune_mcmc_default(ℓ, 4000, report = report);


μₕ₁ = @Constant randn(4); # placebo
μₕ₂ = 2.0 + @Constant randn(4); #treatment
β₁ = vcat(ntuple(i -> (@Constant randn(domains[i])) + μₕ₁[i], Val(4))...); # placebo
β₂ = vcat(ntuple(i -> (@Constant randn(domains[i])) + μₕ₂[i], Val(4))...); # treatment

δt = @Constant rand(23); lastt = Ref(0.0);
t = ConstantFixedSizePaddedVector{24,Float64}(ntuple(i -> i == 1 ? 0.0 : lastt[] += δt[i-1], Val(24)));

μ₁ = ITPExpectedValue(t, β₁, κ)
μ₂ = ITPExpectedValue(t, β₂, κ)

β = @CFixedSize randn(16);
κ = @CFixedSize randexp(16);

μ = ProbabilityModels.ITPExpectedValue(t, β, κ)

ℓitp = BernoulliLogitModel(
    α_κ = 1.0, δt = δtv, μ_β, η₀, σ_β, Y, β, ρ, α_ρ, κ, L, β_κ, t, β_ρ
);

# y =
