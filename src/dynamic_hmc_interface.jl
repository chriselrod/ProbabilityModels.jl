struct ScalarVectorPCG{N} <: Random.AbstractRNG
    scalar::RandomNumbers.PCG.PCGStateUnique{UInt64,Val{:RXS_M_XS},UInt64}
    vector::VectorizedRNG.PCG{N}
end
@inline Random.rand(pcg::ScalarVectorPCG) = rand(pcg.scalar)
@inline Random.randn(pcg::ScalarVectorPCG) = randn(pcg.scalar)
@inline Random.randexp(pcg::ScalarVectorPCG) = randexp(pcg.scalar)
@inline Random.rand(pcg::ScalarVectorPCG, ::Type{Float32}) = rand(pcg.scalar, Float32)
@inline Random.randn(pcg::ScalarVectorPCG, ::Type{Float32}) = randn(pcg.scalar, Float32)
@inline Random.randexp(pcg::ScalarVectorPCG, ::Type{Float32}) = randexp(pcg.scalar, Float32)
@inline Random.rand(pcg::ScalarVectorPCG, ::Type{Float64}) = rand(pcg.scalar, Float64)
@inline Random.randn(pcg::ScalarVectorPCG, ::Type{Float64}) = randn(pcg.scalar, Float64)
@inline Random.randexp(pcg::ScalarVectorPCG, ::Type{Float64}) = randexp(pcg.scalar, Float64)
@inline Random.rand(pcg::ScalarVectorPCG, ::Type{T}) where {T <: SIMDPirates.IntegerTypes} = rand(pcg.scalar, T)
@inline Random.randn(pcg::ScalarVectorPCG, ::Type{T}) where {T <: SIMDPirates.IntegerTypes} = randn(pcg.scalar, T)
@inline Random.randexp(pcg::ScalarVectorPCG, ::Type{T}) where {T <: SIMDPirates.IntegerTypes} = randexp(pcg.scalar, T)
@inline Random.rand(pcg::ScalarVectorPCG, ::Type{T}) where {T <: VectorizationBase.AbstractSIMDVector} = rand(pcg.vector, T)
@inline Random.randn(pcg::ScalarVectorPCG, ::Type{T}) where {T <: VectorizationBase.AbstractSIMDVector} = randn(pcg.vector, T)
@inline Random.randexp(pcg::ScalarVectorPCG, ::Type{T}) where {T <: VectorizationBase.AbstractSIMDVector} = randexp(pcg.vector, T)
@inline Random.rand(pcg::ScalarVectorPCG, ::PaddedMatrices.Static{N}) where {N} = rand(pcg.vector, PaddedMatrices.Static{N}())
@inline Random.randn(pcg::ScalarVectorPCG, ::PaddedMatrices.Static{N}) where {N} = randn(pcg.vector, PaddedMatrices.Static{N}())
@inline Random.randexp(pcg::ScalarVectorPCG, ::PaddedMatrices.Static{N}) where {N} = randexp(pcg.vector, PaddedMatrices.Static{N}())
const GLOBAL_ScalarVectorPCG = ScalarVectorPCG(
    RandomNumbers.PCG.PCGStateUnique(PCG.PCG_RXS_M_XS),
    VectorizedRNG.GLOBAL_vPCG
);


import DynamicHMC
using LinearAlgebra, Random

function DynamicHMC.GaussianKE(::PaddedMatrices.Static{S}, m⁻¹ = 1.0) where {S}
    GaussianKE(Diagonal(fill(m⁻¹, PaddedMatrices.Static{S}())))
end
function DynamicHMC.GaussianKE(M⁻¹::Diagonal{T,<:ConstantFixedSizePaddedVector{N,T}}) where {N,T}
    GaussianKE(
        M⁻¹,
        # For a diagonal matrix, invchol is equivalent to cholinv
        # both are simply W_{i,i} = 1/sqrt(M⁻¹_{i,i})
        # should be lower triangular
        # equals cholesky(inv(M⁻¹)).L
        PaddedMatrices.invchol(M⁻¹)
    )
end
function DynamicHMC.GaussianKE(M⁻¹::PaddedMatrices.AbstractConstantFixedSizePaddedMatrix)
    GaussianKE(M⁻¹, PaddedMatrices.invchol(M⁻¹))
end

@generated function calc_q′(q, ϵ,
            κ::DynamicHMC.GaussianKE{Diagonal{T,ConstantFixedSizePaddedVector{N,T,L,L}},Diagonal{T,ConstantFixedSizePaddedVector{N,T,L,L}}},
            pₘ) where {N,T,L}
    quote
        $(Expr(:meta,:inline))
        mv = MutableFixedSizePaddedVector{$N,$T,$L,$L}(undef)
        M⁻¹diag = κ.Minv.diag
        @vectorize for i ∈ 1:$L
            mv[i] = vmuladd(ϵ * M⁻¹diag[i], pₘ[i], q[i])
        end
        ConstantFixedSizePaddedArray(mv)
    end
end
@generated function calc_q′(q, ϵ,
            κ::DynamicHMC.GaussianKE{<:PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{N,N,T},<:PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{N,N,T}},
            pₘ) where {N,T,L}
    quote
        $(Expr(:meta,:inline))
        loggrad = κ.Minv * pₘ
        SIMDPirates.vmuladd(ϵ, loggrad, q)
    end
end

function DynamicHMC.leapfrog(H::DynamicHMC.Hamiltonian{Tℓ,Tκ}, z::DynamicHMC.PhasePoint, ϵ) where {Tℓ,N,T,A<:Union{<:PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{N,N,T},<:Diagonal{T,<:ConstantFixedSizePaddedVector{N,T}}},Tκ <: DynamicHMC.GaussianKE{A}}
    DynamicHMC.@unpack ℓ, κ = H
    DynamicHMC.@unpack p, q, ℓq = z
    ϵₕ = ϵ/2
    pₘ = SIMDPirates.vmuladd(ϵₕ, ℓq.gradient, p)
    q′ = calc_q′(q, ϵ, κ, pₘ) #SIMDPirates.vfnmadd(ϵ, loggradient
    ℓq′ = LogDensityProblems.logdensity(LogDensityProblems.ValueGradient, ℓ, q′)
    p′ = SIMDPirates.vmuladd(ϵₕ, ℓq′.gradient, pₘ)
    DynamicHMC.PhasePoint(q′, p′, ℓq′)
end

@inline function Random.rand(rng::VectorizedRNG.PCG,
            κ::DynamicHMC.GaussianKE{Diagonal{T,ConstantFixedSizePaddedVector{N,T,L,L}},Diagonal{T,ConstantFixedSizePaddedVector{N,T,L,L}}},
            q = nothing) where {T,N,L}

    κ.W * randn(rng, ConstantFixedSizePaddedVector{N,T})
end
@inline function Random.rand(rng::VectorizedRNG.PCG,
            κ::DynamicHMC.GaussianKE{<:PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{N,N,T},<:PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{N,N,T}},
            q = nothing) where {T,N}
    # @show N
    # @show κ.W
    r = randn(rng, ConstantFixedSizePaddedVector{N,T})
    # @show r
    # @show typeof(κ.W), typeof(r)
    κ.W * r
end
@inline Random.rand(rng::ScalarVectorPCG, κ::DynamicHMC.GaussianKE{Diagonal{T,ConstantFixedSizePaddedVector{N,T,L,L}},
    Diagonal{T,ConstantFixedSizePaddedVector{N,T,L,L}}}, q = nothing) where {T,N,L} = rand(rng.vector, κ, q)
@inline Random.rand(rng::ScalarVectorPCG, κ::DynamicHMC.GaussianKE{<:PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{N,N,T},
                    <:PaddedMatrices.AbstractConstantFixedSizePaddedMatrix{N,N,T}}, q = nothing) where {T,N} = rand(rng.vector, κ, q)








function sample_crossprod_quote(N,T,Ptrunc,Pfull,stride)

    L3 = Pfull * Ptrunc
    W = VectorizationBase.pick_vector_width(Pfull, T)
    m_rep = Pfull ÷ W
    V = Vec{W,T}
    num_reps = cld(L3 ÷ W + 3, VectorizationBase.REGISTER_COUNT)

    piter = cld(Ptrunc, num_reps)
    q = quote
        $(Expr(:meta, :inline))
        plow = 0
        vA = VectorizationBase.vectorizable(Base.unsafe_convert(Ptr{T}, pointer(sample)))
        vout = VectorizationBase.vectorizable(out)
    end
    if num_reps > 1
        push!(q.args, quote
            for pmax ∈ 1:$(num_reps-1)
                $(mul_block_nt(V, W, stride, stride, m_rep, N, piter, :plow, :vA, :vA, Pfull))
                $(store_block(W, Pfull, m_rep, piter, :plow))
                plow += $piter
            end
        end)
    end
    plow = piter * (num_reps-1)
    prem = Ptrunc - plow
    prem > 0 && push!(q.args, PaddedMatrices.mul_block_nt(V, W, stride, stride, m_rep, N, prem, plow, :vA, :vA, Pfull))
    prem > 0 && push!(q.args, PaddedMatrices.store_block(W, Pfull, m_rep, prem, plow))
    # push!(q.args,  :(ConstantFixedSizePaddedMatrix( out )) )
    q
end

function stable_tune(sampler_0, seq::DynamicHMC.TunerSequence)
    tuners = seq.tuners
    Base.Cartesian.@nexprs 7 i -> begin
        tuner_i = tuners[i]
        sampler_i = DynamicHMC.tune(sampler_{i-1}, tuner_i)
    end
    sampler_7
end

@inline function sample_sum(sample)
    x̄ = DynamicHMC.get_position(sample[1])
    @inbounds for n ∈ 2:length(sample)
        x̄ += DynamicHMC.get_position(sample[n])
    end
    x̄
end
@inline function sample_mean(sample)
    sample_sum(sample) * (1/length(sample))
end

@generated function DynamicHMC.sample_cov(sample::Vector{DynamicHMC.NUTS_Transition{Tv,Tf}}, reg) where {T,P,Tv <: PaddedMatrices.AbstractFixedSizePaddedVector{P,T},Tf,L}
    sample_mat_stride, leftover_stride = divrem(sizeof(DynamicHMC.NUTS_Transition{Tv,Tf}), sizeof(T))
    @assert leftover_stride == 0
    W = VectorizationBase.pick_vector_width(P, T)
    rem = P & (W - 1)
    L = rem == 0 ? P : P - rem + W
    quote
        # $(Expr(:meta,:inline))
        N = length(sample)
        x̄ = sample_sum(sample)
        x̄ *= sqrt(1/N)
        out = MutableFixedSizePaddedMatrix{$P,$P,$T,$L,$(L*P)}(undef)
        @inbounds for p ∈ 1:$P
            x̄ₚ = - x̄[p]
            @vectorize $T for l ∈ 1:$L
                out[l, p] = x̄[l] * x̄ₚ
            end
        end
        # @inbounds for i ∈ 1:$(L*P)
        #     out[i] = zero(T)
        # end
        # ConstantFixedSizePaddedMatrix(out)
        $(sample_crossprod_quote(:N,T,P,L,sample_mat_stride))
        m = MutableFixedSizePaddedVector{$P,$T,$L,$L}(undef)
        @inbounds for p ∈ 1:$P
            m[p] = out[p,p]
        end
        off_diag_reg = (1 - reg) / (N - 1)
        diag_reg = quickmedian(m) * reg / (N - 1)
        # off_diag_reg = (1 - reg) / (N - 1)
        @vectorize $T for i ∈ 1:$(L*P)
            out[i] = out[i] * off_diag_reg
        end
        @inbounds for p ∈ 1:$P
            out[p,p] += diag_reg
        end
        # @inbounds for i ∈ 1:$(L*P)
        #     out[i] *= (1/(N-1))
        # end
        # out
        ConstantFixedSizePaddedMatrix(out)
    end
end
@generated function DynamicHMC.sample_cov(sample::Vector{DynamicHMC.NUTS_Transition{Tv,Tf}}) where {T,P,Tv <: PaddedMatrices.AbstractFixedSizePaddedVector{P,T},Tf,L}
    sample_mat_stride, leftover_stride = divrem(sizeof(DynamicHMC.NUTS_Transition{Tv,Tf}), sizeof(T))
    @assert leftover_stride == 0
    W = VectorizationBase.pick_vector_width(P, T)
    rem = P & (W - 1)
    L = rem == 0 ? P : P - rem + W
    quote
        # $(Expr(:meta,:inline))
        N = length(sample)
        x̄ = sample_sum(sample)
        x̄ *= sqrt(1/N)
        out = MutableFixedSizePaddedMatrix{$P,$P,$T,$L,$(L*P)}(undef)
        @inbounds for p ∈ 1:$P
            x̄ₚ = - x̄[p]
            @vectorize $T for l ∈ 1:$L
                out[l, p] = x̄[l] * x̄ₚ
            end
        end
        $(sample_crossprod_quote(:N,T,P,L,sample_mat_stride))
        denom = 1/(N-1)
        @inbounds for i ∈ 1:$(L*P)
            out[i] *= denom
        end
        ConstantFixedSizePaddedMatrix(out)
    end
end

function DynamicHMC.tune(sampler::NUTS{Tv}, tuner::DynamicHMC.StepsizeCovTuner) where {P,T,Tv <: PaddedMatrices.AbstractFixedSizePaddedVector{P,T}}
    regularize = tuner.regularize
    N = tuner.N
    rng = sampler.rng
    H = sampler.H
    max_depth = sampler.max_depth
    report = sampler.report

    sample, A = DynamicHMC.mcmc_adapting_ϵ(sampler, N)
    Σ = sample_cov(sample, regularize/N)
    κ = DynamicHMC.GaussianKE(Σ)
    DynamicHMC.NUTS(rng, DynamicHMC.Hamiltonian(H.ℓ, κ), sample[end].q, DynamicHMC.get_final_ϵ(A), max_depth, report)
end

@inline function partition!(a::PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T}, x::T, i, j) where {N,T}
    @inbounds while i < j
        while a[i] < x
            i += 1
        end
        while a[j] > x
            j -= 1
        end
        a[i], a[j] = a[j], a[i]
    end
    i, j
end
@inline function quickmedian(a::PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T}) where {N,T}
    x = T(0.5) * (a[1] + a[N])
    L = 1
    R = N
    K = N >> 1
    # a = MVector(as)
    @inbounds while L < R
        x = a[K]
        i, j = partition!(a, x, L, R)
        j <= K && (L = i)
        i >= K && (R = j)
    end
    if iseven(N)
        xlow = x
        K += 1
        R = N
        @inbounds while L < R
            x = a[K]
            i, j = partition!(a, x, L, R)
            j <= K && (L = i)
            i >= K && (R = j)
        end
        x = T(0.5)*(xlow + x)
    end
    x
end

@generated default_tuners() = DynamicHMC.bracketed_doubling_tuner()
function NUTS_init_tune_mcmc_default(rng, ℓ, N; args...)
    sampler_init = DynamicHMC.NUTS_init(rng, ℓ; args...)
    sampler_tuned = stable_tune(sampler_init, default_tuners())
    DynamicHMC.mcmc(sampler_tuned, N), sampler_tuned
end
NUTS_init_tune_mcmc_default(ℓ, N; args...) = NUTS_init_tune_mcmc_default(GLOBAL_ScalarVectorPCG, ℓ, N; args...)
