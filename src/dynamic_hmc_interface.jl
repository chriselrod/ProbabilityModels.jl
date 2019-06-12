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
issmall(::PaddedMatrices.Static{S}) where {S} = 2S <= VectorizationBase.REGISTER_SIZE
issmall(n::Number) = 2n <= VectorizationBase.REGISTER_SIZE
@generated function Random.rand(pcg::ScalarVectorPCG, ::PaddedMatrices.Static{N}) where {N}
    if 2N > VectorizationBase.REGISTER_SIZE
        q = quote
            x = Vector{Float64}(undef, $N)
            rand!(pcg.vector, x)
            x
        end
    else
        q = quote
            $(Expr(:meta,:inline))
            rand(pcg.vector, PaddedMatrices.Static{$N}())
        end
    end
    q
end
@generated function Random.randn(pcg::ScalarVectorPCG, ::PaddedMatrices.Static{N}) where {N}
    if 2N > VectorizationBase.REGISTER_SIZE
        q = quote
            x = Vector{Float64}(undef, $N)
            randn!(pcg.vector, x)
            x
        end
    else
        q = quote
            $(Expr(:meta,:inline))
            randn(pcg.vector, PaddedMatrices.Static{$N}())
        end
    end
    q
end
@generated function Random.randexp(pcg::ScalarVectorPCG, ::PaddedMatrices.Static{N}) where {N}
    if 2N > VectorizationBase.REGISTER_SIZE
        q = quote
            x = Vector{Float64}(undef, $N)
            randexp!(pcg.vector, x)
            x
        end
    else
        q = quote
            $(Expr(:meta,:inline))
            randexp(pcg.vector, PaddedMatrices.Static{$N}())
        end
    end
    q
end
function threadrandinit()
    N = Base.Threads.nthreads()
    W = VectorizationBase.pick_vector_width(Float64)
    rngs = Vector{ScalarVectorPCG{4}}(undef, N)
    seeds = rand(UInt64, 4W*N)
    myprocid = myid()-1
    Base.Threads.@threads for n ∈ 1:N
        rngs[n] = ScalarVectorPCG(
            RandomNumbers.PCG.PCGStateUnique(PCG.PCG_RXS_M_XS),
            VectorizedRNG.PCG(ntuple(j -> seeds[j+4W*((n-1))], Val(4W)), (n-1) + N*myprocid )
        )
    end
    rngs
end


import DynamicHMC
using LinearAlgebra, Random

@generated function DynamicHMC.GaussianKE(::PaddedMatrices.Static{S}) where {S}
    if 2S > VectorizationBase.REGISTER_SIZE
        # return quote
        #     m⁻¹ = 1.0
        #     rm = 1.0
        #     out = MutableFixedSizePaddedVector{$S,Float64}(undef)
        #     @inbounds for s ∈ 1:$S
        #         out[s] = m⁻¹
        #     end
        #     GaussianKE(Diagonal(out),Diagonal(out)) # Why not let them alias each other?
        # end
        return :(DynamicHMC.GaussianKE($S))
    else
        return quote
            m⁻¹ = 1.0
            rm = 1.0
            M⁻¹ = fill(m⁻¹, PaddedMatrices.Static{S}())
            GaussianKE(Diagonal(M⁻¹), Diagonal(M⁻¹))
        end
    end
end
@generated function DynamicHMC.GaussianKE(::PaddedMatrices.Static{S}, m⁻¹) where {S}
    if 2S > VectorizationBase.REGISTER_SIZE
        # return quote
        #     M⁻¹ = MutableFixedSizePaddedVector{$S,Float64}(undef)
        #     W = MutableFixedSizePaddedVector{$S,Float64}(undef)
        #     @fastmath rm = 1 / sqrt(m⁻¹)
        #     @inbounds for s ∈ 1:$S
        #         M⁻¹[s] = m⁻¹
        #         W[s] = rm
        #     end
        #     GaussianKE(Diagonal(M⁻¹), Diagonal(W))
        # end
        return :(DynamicHMC.GaussianKE($S), m⁻¹)
    else
        return :(GaussianKE(Diagonal(fill(m⁻¹, PaddedMatrices.Static{S}())),Diagonal(fill(@fastmath(1/sqrt(m⁻¹)), PaddedMatrices.Static{S}()))))
    end
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
function DynamicHMC.GaussianKE(M⁻¹::Diagonal{T,<:PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T}}) where {N,T}
    W = MutableFixedSizePaddedVector{N,T}(undef)
    M⁻¹d = M⁻¹.diag
    @inbounds @fastmath @simd ivdep for n ∈ 1:N
        W[n] = 1 / sqrt(M⁻¹d[n])
    end
    GaussianKE(
        M⁻¹,
        # For a diagonal matrix, invchol is equivalent to cholinv
        # both are simply W_{i,i} = 1/sqrt(M⁻¹_{i,i})
        # should be lower triangular
        # equals cholesky(inv(M⁻¹)).L
        Diagonal(W)
    )
end
function DynamicHMC.GaussianKE(M⁻¹::PaddedMatrices.AbstractConstantFixedSizePaddedMatrix)
    GaussianKE(M⁻¹, PaddedMatrices.invchol(M⁻¹))
end
function DynamicHMC.GaussianKE(M⁻¹::PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{N,N,T,P,L}) where {N,T,P,L}
    W = MutableFixedSizePaddedMatrix{N,N,T,P,L}(undef)
    @inbounds for l ∈ 1:L
        W[l] = M⁻¹[l]
    end
    PaddedMatrices.LAPACK_chol!(W)
    PaddedMatrices.LAPACK_tri_inv!(W)
    GaussianKE(Symmetric(M⁻¹), UpperTriangular(W))
end
function DynamicHMC.GaussianKE(M⁻¹::Symmetric{T,M}) where {N,T,P,L,M<:PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{N,N,T,P,L}}
    W = MutableFixedSizePaddedMatrix{N,N,T,P,L}(undef)
    M⁻¹data = M⁻¹.data
    @inbounds for c ∈ 1:N
        for r ∈ 1:c
            W[r,c] = M⁻¹data[r,c]
        end
    end
    PaddedMatrices.LAPACK_chol!(W)
    PaddedMatrices.LAPACK_tri_inv!(W)
    GaussianKE(M⁻¹, UpperTriangular(W))
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
@generated function calc_q′(q, ϵ,
            κ::DynamicHMC.GaussianKE{Diagonal{T,PV},Diagonal{T,PV}}, pₘ) where {T,N,L, PV <: PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T,L,L}}
    quote
        $(Expr(:meta,:inline))
        mv = MutableFixedSizePaddedVector{$N,$T,$L,$L}(undef)
        M⁻¹diag = κ.Minv.diag
        @vectorize for i ∈ 1:$L
            mv[i] = vmuladd(ϵ * M⁻¹diag[i], pₘ[i], q[i])
        end
        mv
    end
end
@generated function calc_q′(q, ϵ,
            κ::DynamicHMC.GaussianKE{Symmetric{T,M},UpperTriangular{T,M}},
            pₘ) where {N,T,M<:PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{N,N,T}}
    quote
        # $(Expr(:meta,:inline))
        q′ = copy(q)
        PaddedMatrices.BLAS_dsymv!(κ.Minv.data, pₘ, q′, ϵ, 1.0)
        q′
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
function DynamicHMC.leapfrog(H::DynamicHMC.Hamiltonian{Tℓ,Tκ}, z::DynamicHMC.PhasePoint, ϵ) where {Tℓ,N,T,A<:Union{Symmetric{T,<:PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{N,N,T}},<:Diagonal{T,<:PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T}}},Tκ <: DynamicHMC.GaussianKE{A}}
    DynamicHMC.@unpack ℓ, κ = H
    DynamicHMC.@unpack p, q, ℓq = z
    ϵₕ = ϵ/2
    # println("Dispatch is working!")
    # println("typeof(A) = $(typeof(A))")
    p′ = MutableFixedSizePaddedVector{N,T}(undef)
    ℓqg = ℓq.gradient
    @inbounds @fastmath @simd ivdep for n ∈ 1:N
        p′[n] = ϵₕ * ℓqg[n] + p[n]
    end
    # pₘ = SIMDPirates.vmuladd(ϵₕ, ℓq.gradient, p)
    q′ = calc_q′(q, ϵ, κ, p′) #SIMDPirates.vfnmadd(ϵ, loggradient
    ℓq′ = LogDensityProblems.logdensity(LogDensityProblems.ValueGradient, ℓ, q′)
    ℓq′g = ℓq′.gradient
    @inbounds @fastmath @simd ivdep for n ∈ 1:N
        p′[n] = ϵₕ * ℓq′g[n] + p[n]
    end
    # p′ = SIMDPirates.vmuladd(ϵₕ, ℓq′.gradient, pₘ)
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

@inline function Random.rand(rng::VectorizedRNG.PCG,
            κ::DynamicHMC.GaussianKE{Diagonal{T,PV},Diagonal{T,PV}},
            q = nothing) where {T,N,L, PV <: PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T,L,L}}

    r = randn(rng, MutableFixedSizePaddedVector{N,T})
    W = κ.W.diag
    @inbounds @fastmath @simd ivdep for l ∈ 1:L
        r[l] *= W[l]
    end
    r
end

@inline function Random.rand(rng::VectorizedRNG.PCG,
            κ::DynamicHMC.GaussianKE{Symmetric{T,M},UpperTriangular{T,M}},
            q = nothing) where {T,N,M<: PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{N,N,T}}
    # @show N
    # @show κ.W
    r = randn(rng, MutableFixedSizePaddedVector{N,T})
    # @show r
    # @show typeof(κ.W), typeof(r)
    PaddedMatrices.BLAS_dtrmv!(κ.W.data, r)
    r
    # ConstantFixedSizePaddedVector(r)
end
@inline Random.rand(rng::ScalarVectorPCG, κ::DynamicHMC.GaussianKE, q = nothing) where {T,N,L} = rand(rng.vector, κ, q)


function DynamicHMC.get_p♯(
            κ::DynamicHMC.GaussianKE{Symmetric{T,M},UpperTriangular{T,M}},
            p::PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T}, q = nothing) where {N,T,M<:PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{N,N,T}}

    p♯ = MutableFixedSizePaddedVector{N,T}(undef)
    PaddedMatrices.BLAS_dsymv!(κ.Minv.data, p, p♯)
    p♯
end
function DynamicHMC.loggradient(
            κ::DynamicHMC.GaussianKE{Symmetric{T,M},UpperTriangular{T,M}},
            p::PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T}, q = nothing) where {N,T,M<:PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{N,N,T}}

    p♯ = MutableFixedSizePaddedVector{N,T}(undef)
    PaddedMatrices.BLAS_dsymv!(κ.Minv.data, p, p♯, -one(T))
    p♯
end




function sample_crossprod_quote(N,T,Ptrunc,Pfull,stride,out = :Σ)

    L3 = Pfull * Ptrunc
    W = VectorizationBase.pick_vector_width(Pfull, T)
    m_rep = Pfull ÷ W
    V = Vec{W,T}
    num_reps = cld(L3 ÷ W + 3, VectorizationBase.REGISTER_COUNT)

    piter = cld(Ptrunc, num_reps)
    q = quote
        $(Expr(:meta, :inline))
        plow = 0
        vA = VectorizationBase.vectorizable(Base.unsafe_convert(Ptr{$T}, pointer(sample)))
        vout = VectorizationBase.vectorizable($out)
    end
    if num_reps > 1
        push!(q.args, quote
            for pmax ∈ 1:$(num_reps-1)
                $(PaddedMatrices.mul_block_nt(V, W, stride, stride, m_rep, N, piter, :plow, :vA, :vA, Pfull))
                $(PaddedMatrices.store_block(W, Pfull, m_rep, piter, :plow))
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

@generated function stable_tune(sampler_0, seq::DynamicHMC.TunerSequence{T}, δ) where {T}
    ntuners = length(T.parameters)
    quote
        tuners = seq.tuners
        # @show typeof(sampler_0)
        Base.Cartesian.@nexprs $ntuners i -> begin
            tuner_i = tuners[i]
            sampler_i = DynamicHMC.tune(sampler_{i-1}, tuner_i, δ)
            # @show typeof(sampler_i)
        end
        $(Symbol(:sampler_, ntuners))
    end
end
function neg_energy(
        κ::DynamicHMC.GaussianKE{Symmetric{T,M},UpperTriangular{T,M}}, p::AbstractFixedSizePaddedVector{N,T}, q = nothing) where {N,T,M<:PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{N,N,T}}

    M⁻¹ = κ.Minv.data
    qfo = zero(T)
    qfd = zero(T)
    @fastmath @inbounds for nc ∈ 1:N
        pnc = p[nc]
        @simd ivdep for nr ∈ 1:nc-1
            qfo += pnc*p[nr]*M⁻¹[nr,nc]
        end
        qfd += pnc*pnc*M⁻¹[nc,nc]
    end
    @fastmath - qfo - 0.5qfd
end
function neg_energy(
        κ::DynamicHMC.GaussianKE{Diagonal{T,M},Diagonal{T,M}}, p::AbstractFixedSizePaddedVector{N,T}, q = nothing) where {N,T,M<:PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T}}

    M⁻¹ = κ.Minv.diag
    qf = zero(T)
    @fastmath @inbounds @simd ivdep for n ∈ 1:N
        qf += M⁻¹[n] * p[n] * [n]
    end
    qf
end
# @inline function sample_sum(sample)
#     x̄ = DynamicHMC.get_position(sample[1])
#     @inbounds for n ∈ 2:length(sample)
#         x̄ += DynamicHMC.get_position(sample[n])
#     end
#     x̄
# end
@generated function sample_sum(sample::Vector{DynamicHMC.NUTS_Transition{Tv,Tf}}) where {Tf,P,L,Tv <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{P,Tf,L,L}}
    stride_bytes = sizeof(DynamicHMC.NUTS_Transition{Tv,Tf})
    sample_mat_stride, leftover_stride = divrem(stride_bytes, sizeof(Tf))
    @assert leftover_stride == 0

    W, Wshift = VectorizationBase.pick_vector_width_shift(P, Tf)

    WT = W * sizeof(Tf)
    V = Vec{W,Tf}

    # +2, to divide by an additional 4
    iterations = L >> (Wshift + 2)
    r = L & ((W << 2) - 1)
    riter = r >> Wshift
    if P <= 4W
        return quote
            $(Expr(:meta,:inline))
            x̄ = DynamicHMC.get_position(sample[1])
            @inbounds for n ∈ 2:length(sample)
                x̄ += DynamicHMC.get_position(sample[n])
            end
            x̄
        end
    end

    remainder_quote = quote
        Base.Cartesian.@nexprs $riter j -> begin
            offset_j = $(4WT*iterations) + $WT*(j-1)
            # @show offset_j
            # x_j = SIMDPirates.vload($V, ptrx̄ + offset_j)
            x_j = SIMDPirates.vbroadcast($V, zero($Tf))
        end
        for n ∈ 0:N-1
            offset_n = n * $stride_bytes
            Base.Cartesian.@nexprs $riter j -> begin
                x_j = SIMDPirates.vadd(x_j, SIMDPirates.vload($V, ptrsample + offset_n + offset_j))
            end
        end
        Base.Cartesian.@nexprs $riter j -> SIMDPirates.vstore!(ptrx̄ + offset_j, x_j)
    end

    quote
        # x̄ = zero(PaddedMatrices.MutableFixedSizePaddedVector{P,T})
        x̄ = PaddedMatrices.MutableFixedSizePaddedVector{$P,$Tf}(undef)
        ptrx̄ = pointer(x̄)
        ptrsample = Base.unsafe_convert(Ptr{$Tf}, pointer(sample))
        N = length(sample)
        # x̄ = DynamicHMC.get_position(sample[1])
        GC.@preserve x̄ sample begin
            for i ∈ 0:$(iterations-1)
                Base.Cartesian.@nexprs 4 j -> begin
                    offset_j = $(4WT)*i + $WT*(j-1)
                    # @show offset_j
                    # x_j = SIMDPirates.vload($V, ptrx̄ + offset_j)
                    x_j = SIMDPirates.vbroadcast($V, zero($Tf))
                end
                for n ∈ 0:N-1
                    offset_n = n * $stride_bytes
                    Base.Cartesian.@nexprs 4 j -> begin
                        x_j = SIMDPirates.vadd(x_j, SIMDPirates.vload($V, ptrsample + offset_n + offset_j))
                    end
                end
                Base.Cartesian.@nexprs 4 j -> SIMDPirates.vstore!(ptrx̄ + offset_j, x_j)
            end
            $(riter == 0 ? nothing : remainder_quote)
        end
        x̄
    end
end

@generated function columnwise_subtract!(sample::Vector{DynamicHMC.NUTS_Transition{Tv,Tf}}, x̄, s ) where {Tf,P,L,Tv <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{P,Tf,L,L}}
    stride_bytes = sizeof(DynamicHMC.NUTS_Transition{Tv,Tf})
    sample_mat_stride, leftover_stride = divrem(stride_bytes, sizeof(Tf))
    @assert leftover_stride == 0

    W, Wshift = VectorizationBase.pick_vector_width_shift(P, Tf)

    WT = W * sizeof(Tf)
    V = Vec{W,Tf}

    # +2, to divide by an additional 4
    iterations = L >> (Wshift + 2)
    r = L & ((W << 2) - 1)
    riter = r >> Wshift

    remainder_quote = quote
        Base.Cartesian.@nexprs $riter j -> begin
            offset_j = $(4WT*iterations) + $WT*(j-1)
            x_j = SIMDPirates.vmul(SIMDPirates.vload($V, ptrx̄ + offset_j), vs)
        end
        for n ∈ 0:N-1
            offset_n = n * $stride_bytes
            Base.Cartesian.@nexprs $riter j -> begin
                SIMDPirates.vstore!(
                    ptrsample + offset_n + offset_j,
                    SIMDPirates.vsub(SIMDPirates.vload($V, ptrsample + offset_n + offset_j), x_j)
                )
            end
        end
    end

    quote
        vs = SIMDPirates.vbroadcast($V, s)
        ptrx̄ = pointer(x̄)
        ptrsample = Base.unsafe_convert(Ptr{$Tf}, pointer(sample))
        N = length(sample)
        # x̄ = DynamicHMC.get_position(sample[1])
        GC.@preserve x̄ sample begin
            for i ∈ 0:$(iterations-1)
                Base.Cartesian.@nexprs 4 j -> begin
                    offset_j = $(4WT)*i + $WT*(j-1)
                    x_j = SIMDPirates.vmul(SIMDPirates.vload($V, ptrx̄ + offset_j), vs)
                end
                for n ∈ 0:N-1
                    offset_n = n * $stride_bytes
                    Base.Cartesian.@nexprs 4 j -> begin
                        SIMDPirates.vstore!(
                            ptrsample + offset_n + offset_j,
                            SIMDPirates.vsub(SIMDPirates.vload($V, ptrsample + offset_n + offset_j), x_j)
                        )
                    end
                end
            end
            $(riter == 0 ? nothing : remainder_quote)
        end
    end
end
@generated function columnwise_variance!(
                σ²::PaddedMatrices.AbstractMutableFixedSizePaddedVector{P,Tf,L,L},
                sample::Vector{DynamicHMC.NUTS_Transition{Tv,Tf}}, x̄, s
            ) where {Tf,L,P,Tv <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{P,Tf,L,L}}

    stride_bytes = sizeof(DynamicHMC.NUTS_Transition{Tv,Tf})
    sample_mat_stride, leftover_stride = divrem(stride_bytes, sizeof(Tf))
    @assert leftover_stride == 0

    W, Wshift = VectorizationBase.pick_vector_width_shift(P, Tf)

    WT = W * sizeof(Tf)
    V = Vec{W,Tf}

    # +2, to divide by an additional 4
    iterations = L >> (Wshift + 2)
    r = L & ((W << 2) - 1)
    riter = r >> Wshift

    remainder_quote = quote
        Base.Cartesian.@nexprs $riter j -> begin
            offset_j = $(4WT*iterations) + $WT*(j-1)
            x_j = SIMDPirates.vmul(SIMDPirates.vload($V, ptrx̄ + offset_j), vs)
            x2_j = SIMDPirates.vbroadcast($V, zero($Tf))
        end
        for n ∈ 0:N-1
            offset_n = n * $stride_bytes
            Base.Cartesian.@nexprs $riter j -> begin
                xdiff_j = SIMDPirates.vsub(SIMDPirates.vload($V, ptrsample + offset_n + offset_j), x_j)
                x2_j = SIMDPirates.vmuladd(xdiff_j, xdiff_j, x2_j)
            end
        end
        Base.Cartesian.@nexprs $riter j -> begin
            SIMDPirates.vstore!(vσ² + offset_j, x2_j)
        end
    end

    quote
        vs = SIMDPirates.vbroadcast($V, s)
        ptrx̄ = pointer(x̄)
        vσ² = pointer(σ²)
        ptrsample = Base.unsafe_convert(Ptr{$Tf}, pointer(sample))
        N = length(sample)
        vNm1⁻¹ = SIMDPirates.vbroadcast($V, 1 / (N-1))
        # x̄ = DynamicHMC.get_position(sample[1])
        GC.@preserve x̄ sample begin
            for i ∈ 0:$(iterations-1)
                Base.Cartesian.@nexprs 4 j -> begin
                    offset_j = $(4WT)*i + $WT*(j-1)
                    x_j = SIMDPirates.vmul(SIMDPirates.vload($V, ptrx̄ + offset_j), vs)
                    x2_j = SIMDPirates.vbroadcast($V, zero($Tf))
                end
                for n ∈ 0:N-1
                    offset_n = n * $stride_bytes
                    Base.Cartesian.@nexprs 4 j -> begin
                        xdiff_j = SIMDPirates.vsub(SIMDPirates.vload($V, ptrsample + offset_n + offset_j), x_j)
                        x2_j = SIMDPirates.vmuladd(xdiff_j, xdiff_j, x2_j)
                    end
                end
                Base.Cartesian.@nexprs 4 j -> begin
                    SIMDPirates.vstore!(vσ² + offset_j, SIMDPirates.vmul(vNm1⁻¹, x2_j))
                end
            end
            $(riter == 0 ? nothing : remainder_quote)
        end
    end
end
@generated function sample_mean(sample::Vector{DynamicHMC.NUTS_Transition{Tv,Tf}}) where {T,P,Tv <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{P,T},Tf}
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, T)
    if P <= 4W
        return quote
            N⁻¹ = 1 / length(sample)
            sample_sum(sample) * N⁻¹
        end
    else
        L = (P + W - 1) & ~(W - 1)
        return quote
            N⁻¹ = 1 / length(sample)
            x = sample_sum(sample)
            @vectorize $T for l ∈ 1:$L
                x[l] = N⁻¹ * x[l]
            end
            x
        end
    end
end

function LAPACK_dsyrk!(
        C::PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{N,N,Float64,LDC},
        A::Vector{DynamicHMC.NUTS_Transition{Tv,Tf}}, α = 1.0, β = 0.0,
        UPLO = 'U', TRANS = 'N') where {N,LDC,Tv <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{N,Float64},Tf}

    K = length(A)
    LDA = sizeof(DynamicHMC.NUTS_Transition{Tv,Tf}) >> 3

    ccall((LinearAlgebra.BLAS.@blasfunc(dsyrk_), LinearAlgebra.BLAS.libblas), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{LinearAlgebra.BLAS.BlasInt}),
        UPLO, TRANS, N, K, α, A, LDA, β, C, LDC)

end

@generated function sample_cov!(sample::Vector{DynamicHMC.NUTS_Transition{Tv,Tf}}, reg) where {Tf,P,L,Tv <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{P,Tf,L,L}}
    sample_mat_stride, leftover_stride = divrem(sizeof(DynamicHMC.NUTS_Transition{Tv,Tf}), sizeof(Tf))
    @assert leftover_stride == 0
    W = VectorizationBase.pick_vector_width(P, Tf)
    Wm1 = W - 1
    # rem = P & Wm1
    # @show P, W
    # L = (P + Wm1) & ~Wm1
    if P > 4W
        return quote
            N = length(sample)
            x̄ = sample_sum(sample)
            # x̄ *= sqrt(1/N)
            Σ = MutableFixedSizePaddedMatrix{$P,$P,$Tf,$L,$(L*P)}(undef)
            off_diag_reg = (1 - reg / N) /  (N - 1)

            # This commented out code is faster
            # nN⁻¹ = - 1 / N
            # @inbounds for p ∈ 1:$P
            #     x̄ₚ = nN⁻¹ * x̄[p]
            #     s = $L * (p-1)
            #     @vectorize $Tf for l ∈ 1:$L
            #         Σ[l + s] = x̄[l] * x̄ₚ
            #     end
            # end
            # LAPACK_dsyrk!(Σ, sample, off_diag_reg, off_diag_reg)

            # this code is more numerically accurate
            columnwise_subtract!(sample, x̄, 1/N)
            LAPACK_dsyrk!(Σ, sample, off_diag_reg, 0.0)
            # While we do regularize afterwards, I still find it irksome
            # that the faster code can lead to zeros and negative diagonal values.
            # That is because, if x^2 is large relative to the variance,
            # we can get catastrophic cancellation.

            # @show Symmetric(Σ)
            # println("\n\n")
            @inbounds for p ∈ 1:$P
                x̄[p] = Σ[p,p]
            end
            # @show x̄
            # println("\n\n")
            # m̃ = quickmedian!(x̄)
            m̃ = median!(x̄)

            # off_diag_reg⁻¹ = 1 / (1 - reg)
            diag_reg = m̃ * reg / (N - reg)
            # off_diag_reg = (1 - reg) / (N - 1)
            # @vectorize $Tf for i ∈ 1:$(L*P)
            #     Σ[i] = Σ[i] * off_diag_reg
            # end
            @fastmath @inbounds for p ∈ 1:$P
                Σ[p,p] += diag_reg
            end
            Symmetric(Σ)
        end

    end
    quote
        # $(Expr(:meta,:inline))
        N = length(sample)
        x̄ = sample_sum(sample)
        nN⁻¹ = - 1 / N
        Σ = MutableFixedSizePaddedMatrix{$P,$P,$Tf,$L,$(L*P)}(undef)
        @inbounds for p ∈ 1:$P
            x̄ₚ = nN⁻¹ * x̄[p]
            s = $L * (p-1)
            @vectorize $Tf for l ∈ 1:$L
                Σ[l + s] = x̄[l] * x̄ₚ
            end
        end
        # @inbounds for i ∈ 1:$(L*P)
        #     out[i] = zero(T)
        # end
        # ConstantFixedSizePaddedMatrix(out)
        $(sample_crossprod_quote(:N,Tf,P,L,sample_mat_stride))
        m = MutableFixedSizePaddedVector{$P,$Tf,$L,$L}(undef)
        @inbounds for p ∈ 1:$P
            m[p] = Σ[p,p]
        end
        off_diag_reg = (1 - reg / N) / (N - 1)
        diag_reg = median!(m) * reg / (N - reg)
        # diag_reg = quickmedian!(m) * reg / (N - 1)
        # off_diag_reg = (1 - reg) / (N - 1)
        @vectorize $Tf for i ∈ 1:$(L*P)
            Σ[i] = Σ[i] * off_diag_reg
        end
        @inbounds for p ∈ 1:$P
            Σ[p,p] += diag_reg
        end
        # @inbounds for i ∈ 1:$(L*P)
        #     out[i] *= (1/(N-1))
        # end
        # out
        ConstantFixedSizePaddedMatrix(Σ)
    end
end
@inline function DynamicHMC.sample_cov(sample::Vector{DynamicHMC.NUTS_Transition{Tv,Tf}}, reg) where {Tf,P,L,Tv <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{P,Tf,L,L}}
    sample_cov!(copy(sample), reg)
end

@generated function DynamicHMC.sample_cov(sample::Vector{DynamicHMC.NUTS_Transition{Tv,Tf}}) where {P,Tf,L,Tv <: PaddedMatrices.AbstractFixedSizePaddedVector{P,Tf,L,L}}
    sample_mat_stride, leftover_stride = divrem(sizeof(DynamicHMC.NUTS_Transition{Tv,Tf}), sizeof(Tf))
    @assert leftover_stride == 0
    W = VectorizationBase.pick_vector_width(P, Tf)
    Wm1 = W - 1
    # rem = P & Wm1
    # R = (P + Wm1) & ~Wm1
    # @show P, sample_mat_stride, W
    if sample_mat_stride > 4W
        return quote
            # println("Calculating lenngth.")
            N = length(sample)
            # println("Calculating sample sum.")
            x̄ = sample_sum(sample)
            # @show x̄
            Σ = MutableFixedSizePaddedMatrix{$P,$P,$Tf,$L,$(L*P)}(undef)

            # nN⁻¹ = - 1 / N
            # @inbounds for p ∈ 1:$P
            #     x̄ₚ = nN⁻¹ * x̄[p]
            #     s = $L * (p-1)
            #     @vectorize $Tf for l ∈ 1:$L
            #         Σ[l + s] = x̄[l] * x̄ₚ
            #     end
            # end
            # Nm1⁻¹ = 1 / (N - 1)
            # LAPACK_dsyrk!(Σ, sample, Nm1⁻¹, Nm1⁻¹)

            # See comments in the other method.
            # Because this function is aimed at users, we copy the sample to preserve it.
            δsample = copy(sample)
            columnwise_subtract!(δsample, x̄, 1/N)
            LAPACK_dsyrk!(Σ, δsample, 1/(N-1), 0.0)

            Symmetric(Σ)
        end
    end
    quote
        # $(Expr(:meta,:inline))
        N = length(sample)
        x̄ = sample_sum(sample)
        nN⁻¹ = - 1 / N
        Σ = MutableFixedSizePaddedMatrix{$P,$P,$Tf,$L,$(L*P)}(undef)
        @inbounds for p ∈ 1:$P
            x̄ₚ = nN⁻¹ * x̄[p]
            s = $L * (p-1)
            @vectorize $Tf for l ∈ 1:$L
                Σ[l + s] = x̄[l] * x̄ₚ
            end
        end
        $(sample_crossprod_quote(:N,Tf,P,L,sample_mat_stride))
        denom = 1/(N-1)
        @inbounds for i ∈ 1:$(L*P)
            Σ[i] *= denom
        end
        ConstantFixedSizePaddedMatrix(Σ)
    end
end


@generated function sample_diagcov(sample::Vector{DynamicHMC.NUTS_Transition{Tv,Tf}}, reg) where {Tf,L,P,Tv <: PaddedMatrices.AbstractConstantFixedSizePaddedVector{P,Tf,L,L}}
    sample_mat_stride, leftover_stride = divrem(sizeof(DynamicHMC.NUTS_Transition{Tv,Tf}), sizeof(Tf))
    @assert leftover_stride == 0
    W = VectorizationBase.pick_vector_width(P, Tf)
    Wm1 = W - 1
    # rem = P & Wm1
    # @show P, W
    # L = (P + Wm1) & ~Wm1
    if P > 4W
        return quote
            N = length(sample)
            x̄ = sample_sum(sample)
            Σ = MutableFixedSizePaddedVector{$P,$Tf,$L,$L}(undef)
            columnwise_variance!(Σ, sample, x̄, 1/N)

            # @show Σ
            # Σ = Statistics.var(reshape(reinterpret(Float64, get_position.(sample)), ($L,N)), dims = 2)
            @inbounds for l ∈ 1:$L
                x̄[l] = Σ[l]
            end
            # return Diagonal(x̄)
            # # @show x̄
            # m̃ = quickmedian!(x̄)
            # m̃ = median!(x̄)
            # regmul = 1 - reg / N
            # regadd = m̃ * reg / N
            regmul = Tf(N / (N+reg))
            regadd = Tf(1e-3 * (reg / (N+reg)))

            # x̄ already escaped the function in median!, so we'll store there and return it
            @fastmath @inbounds @simd ivdep for l ∈ 1:$L
                x̄[l] = Σ[l] * regmul + regadd
            end
            # @show x̄
            # @show Diagonal(d).diag
            Diagonal(x̄)
        end

    end
    quote
        # $(Expr(:meta,:inline))
        N = length(sample)
        x̄ = sample_sum(sample)
        nN⁻¹ = - 1 / N
        Σ = MutableFixedSizePaddedMatrix{$P,$P,$Tf,$L,$(L*P)}(undef)
        @inbounds for p ∈ 1:$P
            x̄ₚ = nN⁻¹ * x̄[p]
            s = $L * (p-1)
            @vectorize $Tf for l ∈ 1:$L
                Σ[l + s] = x̄[l] * x̄ₚ
            end
        end
        # @inbounds for i ∈ 1:$(L*P)
        #     out[i] = zero(T)
        # end
        # ConstantFixedSizePaddedMatrix(out)
        $(sample_crossprod_quote(:N,Tf,P,L,sample_mat_stride))
        m = MutableFixedSizePaddedVector{$P,$Tf,$L,$L}(undef)
        @inbounds for p ∈ 1:$P
            m[p] = Σ[p,p]
        end
        off_diag_reg = (1 - reg / N) / (N - 1)
        diag_reg = median!(m) * reg / (N - reg)
        # diag_reg = quickmedian!(m) * reg / (N - 1)
        # off_diag_reg = (1 - reg) / (N - 1)
        @vectorize $Tf for i ∈ 1:$(L*P)
            Σ[i] = Σ[i] * off_diag_reg
        end
        @inbounds for p ∈ 1:$P
            Σ[p,p] += diag_reg
        end
        # @inbounds for i ∈ 1:$(L*P)
        #     out[i] *= (1/(N-1))
        # end
        # out
        ConstantFixedSizePaddedMatrix(Σ)
    end
end

@inline function last_position(sample::Vector{NUTS_Transition{PaddedMatrices.ConstantFixedSizePaddedVector{P,T,L,L},T}}) where {P, T, L}
    issmall(PaddedMatrices.Static(P)) && return sample[end].q
    N = length(sample)
    last_position = MutableFixedSizePaddedVector{P,T}(undef)
    stride = sizeof(eltype(sample))
    GC.@preserve sample begin
        ptr = Base.unsafe_convert(Ptr{T}, pointer(sample))
        copyto!(last_position, PaddedMatrices.PtrVector{P,T,L,L}(ptr + stride*(N-1)))
    end
    last_position
end
# @inline function last_position(sample::Vector{NUTS_Transition{Vector{T},T}}) where {T}
#     sample[end].q
# end
@inline last_position(sample) = sample[end].q

function DynamicHMC.tune(
                sampler::NUTS{Tv,Tf,TR,TH}, tuner::DynamicHMC.StepsizeCovTuner, δ::Tf
            ) where {P,Tf,Tv <: PaddedMatrices.AbstractFixedSizePaddedVector{P,Tf},TR,Tℓ<:AbstractProbabilityModel,TH<:DynamicHMC.Hamiltonian{Tℓ}}
    # @show typeof(sampler)
    regularize = tuner.regularize
    N = tuner.N
    rng = sampler.rng
    H = sampler.H
    max_depth = sampler.max_depth
    report = sampler.report

    sample, A = DynamicHMC.mcmc_adapting_ϵ(sampler, N, DynamicHMC.adapting_ϵ(sampler.ϵ, δ = δ)...)
    last_pos = last_position(sample)
    # Σ = sample_cov!(sample, regularize)
    Σ = sample_diagcov(sample, regularize)
    κ = DynamicHMC.GaussianKE(Σ)
    DynamicHMC.NUTS(rng, DynamicHMC.Hamiltonian(H.ℓ, κ), last_pos, DynamicHMC.get_final_ϵ(A), max_depth, report)
end
function DynamicHMC.tune(
                    sampler::NUTS{Tv,Tf,TR,TH}, tuner::DynamicHMC.StepsizeCovTuner, δ::Tf
                ) where {P,Tf,Tv <: Vector{Tf},TR,Tℓ<:AbstractProbabilityModel,TH<:DynamicHMC.Hamiltonian{Tℓ}}
    # @show typeof(sampler)
    regularize = tuner.regularize
    N = tuner.N
    rng = sampler.rng
    H = sampler.H
    max_depth = sampler.max_depth
    report = sampler.report

    sample, A = DynamicHMC.mcmc_adapting_ϵ(sampler, N, DynamicHMC.adapting_ϵ(sampler.ϵ, δ = δ)...)
    Σ = DynamicHMC.sample_cov(sample)
    δΣ = UniformScaling(median!(diag(Σ))) - Σ
    @. Σ += δΣ * regularize/N
    κ = DynamicHMC.GaussianKE(Σ, inv(cholesky(Symmetric(Σ)).U))
    DynamicHMC.NUTS(rng, DynamicHMC.Hamiltonian(H.ℓ, κ), last_position(sample), DynamicHMC.get_final_ϵ(A), max_depth, report)
end
function DynamicHMC.tune(
                        sampler::NUTS{Tv,Tf,TR,TH}, tuner::DynamicHMC.StepsizeTuner, δ::Tf
                    ) where {P,Tf,Tv,TR,Tℓ<:AbstractProbabilityModel,TH<:DynamicHMC.Hamiltonian{Tℓ}}
    N = tuner.N
    rng = sampler.rng
    H = sampler.H
    max_depth = sampler.max_depth
    report = sampler.report
    sample, A = DynamicHMC.mcmc_adapting_ϵ(sampler, N, DynamicHMC.adapting_ϵ(sampler.ϵ, δ = δ)...)
    DynamicHMC.NUTS(rng, H, last_position(sample), DynamicHMC.get_final_ϵ(A), max_depth, report)
end

# @inline function partition!(a::PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T}, x::T, i, j) where {N,T}
#     @inbounds while i < j
#         aᵢ = a[i]
#         while aᵢ < x
#             i += 1
#             aᵢ = a[i]
#         end
#         aⱼ = a[j]
#         while aⱼ > x
#             j -= 1
#             aⱼ = a[j]
#         end
#         # if aᵢ == aⱼ
#         #
#         #     break
#         # else
#             a[i], a[j] = a[j], a[i]
#         # end
#     end
#     i, j
# end
# @inline function quickmedian!(a::PaddedMatrices.AbstractMutableFixedSizePaddedVector{N,T}) where {N,T}
#     # println("ProbabilityModels.quickmedian")
#     # println(a)
#     x = T(0.5) * (a[1] + a[N])
#     L = 1
#     R = N
#     K = N >> 1
#     # a = MVector(as)
#     @inbounds while L < R
#         @show L, R
#         x = a[K]
#         i, j = partition!(a, x, L, R)
#         j <= K && (L = i)
#         i >= K && (R = j)
#     end
#     if iseven(N)
#         xlow = x
#         K += 1
#         R = N
#         @inbounds while L < R
#             x = a[K]
#             i, j = partition!(a, x, L, R)
#             j <= K && (L = i)
#             i >= K && (R = j)
#         end
#         x = T(0.5)*(xlow + x)
#     end
#     x
# end

function DynamicHMC.NUTS_init(rng::Random.AbstractRNG, ℓ::AbstractProbabilityModel{D};
                        q = randn(rng, PaddedMatrices.Static(D)),
                        κ = DynamicHMC.GaussianKE(PaddedMatrices.Static(D)),
                        p = rand(rng, κ),
                        max_depth = DynamicHMC.MAX_DEPTH,
                        ϵ = DynamicHMC.InitialStepsizeSearch(),
                        report = DynamicHMC.ReportIO()) where {D}
    H = DynamicHMC.Hamiltonian(ℓ, κ)
    z = DynamicHMC.phasepoint_in(H, q, p)
    if ϵ isa Float64
        ϵ64 = ϵ
    else
        ϵ64 = DynamicHMC.find_initial_stepsize(ϵ, H, z)
    end
    NUTS(rng, H, q, ϵ64, max_depth, report)
end

@generated default_tuners() = DynamicHMC.bracketed_doubling_tuner()
function NUTS_init_tune_mcmc_default(rng, ℓ, N; δ = 0.8, tuners = default_tuners(), args...)
    # if !issmall(dimension(ℓ))
    #     return NUTS_init_tune_mcmc(rng, ℓ, N; args...)
    # end
    sampler_init = DynamicHMC.NUTS_init(rng, ℓ; args...)
    sampler_tuned = stable_tune(sampler_init, tuners, δ)
    DynamicHMC.mcmc(sampler_tuned, N), sampler_tuned
end
NUTS_init_tune_mcmc_default(ℓ, N; args...) = NUTS_init_tune_mcmc_default(GLOBAL_ScalarVectorPCGs[1], ℓ, N; args...)

# function NUTS_init_tune_distributed(ℓ, N; args...)
#     @distributed vcat for i ∈ 1:nprocs()
#         rng = ScalarVectorPCG(
#             RandomNumbers.PCG.PCGStateUnique(PCG.PCG_RXS_M_XS),
#             PCG{4}(4i) #( could use i-1, but with "i" these will be distinct from the GLOBAL_vPCG )
#         )
#         samples, tuned_sampler = NUTS_init_tune_mcmc_default(rng, ℓ, N; args...)
#         samples
#     end
# end

@generated function vector_container(::PaddedMatrices.Static{N}, nthread = Base.Threads.nthreads()) where {N}
    T = Float64
    Wm1 = VectorizationBase.pick_vector_width(N, T) - 1
    L = (N + Wm1) & ~Wm1
    if 2N > VectorizationBase.REGISTER_SIZE
        return :(Vector{Vector{DynamicHMC.NUTS_Transition{Vector{Float64},Float64}}}(undef, nthread))
    else
        return :(Vector{Vector{DynamicHMC.NUTS_Transition{PaddedMatrices.ConstantFixedSizePaddedVector{$N,Float64,$L,$L},Float64}}}(undef, nthread))
    end
end


function NUTS_init_tune_threaded(ℓ, N; nchains = Base.Threads.nthreads(), args...)
    chains = vector_container(dimension(ℓ), nchains)

    Base.Threads.@threads for i ∈ 1:nchains
        rng = GLOBAL_ScalarVectorPCGs[i]
        if i == 1
            # The first chain logs.
            samples, tuned_sampler = NUTS_init_tune_mcmc_default(rng, ℓ, N; args...)
        else
            # The other chains do not, because IO is not yet threadsafe.
            samples, tuned_sampler = NUTS_init_tune_mcmc_default(rng, ℓ, N; report = DynamicHMC.ReportSilent(), args...)
        end
        chains[i] = samples
    end
    chains
end

function NUTS_init_tune_distributed(ℓ, N; nchains = nprocs()-1, args...)
    paired_res = pmap(i -> NUTS_init_tune_mcmc_default(ℓ, N; args...), 1:nchains)
    getindex.(paired_res, 1), getindex.(paired_res, 2)
end

function store_transition!(psample::Ptr{Tf}, trans::DynamicHMC.NUTS_Transition{Tv,Tf}) where {M,Tf,L,Tv <: PaddedMatrices.AbstractMutableFixedSizePaddedVector{M,Tf,L,L}}
    T_size = sizeof(Tf)
    copyto!(PaddedMatrices.PtrVector{M,Tf,L,L}(psample), trans.q)
    psample += T_size*L
    # store logdensity
    VectorizationBase.store!(psample, trans.π)
    psample += T_size
    VectorizationBase.store!(psample, reinterpret(Tf, trans.depth))
    psample += T_size
    VectorizationBase.store!(psample, reinterpret(Tf, UInt64(reinterpret(UInt32, trans.termination))))
    psample += T_size
    VectorizationBase.store!(psample, trans.a)
    psample += T_size
    VectorizationBase.store!(psample, reinterpret(Tf, trans.steps))
    psample += T_size
    psample
end

function DynamicHMC.mcmc(sampler::NUTS{Tv,Tf}, N::Int) where {M,Tf,L,Tv <: PaddedMatrices.AbstractMutableFixedSizePaddedVector{M,Tf,L,L}}
    rng = sampler.rng
    H = sampler.H
    q = sampler.q
    ϵ = sampler.ϵ
    max_depth = sampler.max_depth
    report = sampler.report
    sample = Vector{NUTS_Transition{PaddedMatrices.ConstantFixedSizePaddedVector{M,Tf,L,L},Tf}}(undef, N)
    psample = Base.unsafe_convert(Ptr{Tf}, pointer(sample))
    DynamicHMC.start_progress!(report, "MCMC"; total_count = N)
    for n ∈ 1:N
        # figure out handling of "q" / last transition versus current transition and assignment...
        trans = DynamicHMC.NUTS_transition(rng, H, q, ϵ, max_depth)
        q = trans.q
        psample = store_transition!(psample, trans)
        DynamicHMC.report!(report, n)
    end
    DynamicHMC.end_progress!(report)
    sample
end
function DynamicHMC.mcmc_adapting_ϵ(sampler::NUTS{Tv,Tf}, N::Int, A_params, A) where {M,Tf,L,Tv <: PaddedMatrices.AbstractMutableFixedSizePaddedVector{M,Tf,L,L}}
    rng = sampler.rng
    H = sampler.H
    q = sampler.q
    max_depth = sampler.max_depth
    report = sampler.report
    sample = Vector{NUTS_Transition{PaddedMatrices.ConstantFixedSizePaddedVector{M,Tf,L,L},Tf}}(undef, N)
    psample = Base.unsafe_convert(Ptr{Tf}, pointer(sample))
    DynamicHMC.start_progress!(report, "MCMC, adapting ϵ"; total_count = N)
    for n ∈ 1:N
        trans = DynamicHMC.NUTS_transition(rng, H, q, DynamicHMC.get_current_ϵ(A), max_depth)#::NUTS_Transition{MutableFixedSizePaddedVector{M,Tf,L,L},Tf}
        q = trans.q
        A = DynamicHMC.adapt_stepsize(A_params, A, trans.a)
        psample = store_transition!(psample, trans)
        # @assert all(sample[n].q .== ConstantFixedSizePaddedVector(q))
        # println("Copy succesful!")
        DynamicHMC.report!(report, n)
    end
    DynamicHMC.end_progress!(report)
    sample, A
end
