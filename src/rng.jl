
struct ScalarVectorPCG{N} <: Random.AbstractRNG
    scalar::RandomNumbers.PCG.PCGStateUnique{UInt64,Val{:RXS_M_XS},UInt64}
    vector::VectorizedRNG.PCG{N}
end
@inline Random.rand(pcg::ScalarVectorPCG) = rand(pcg.scalar)
@inline Random.randn(pcg::ScalarVectorPCG) = randn(pcg.scalar)
@inline Random.randexp(pcg::ScalarVectorPCG) = randexp(pcg.scalar)
@inline Random.rand!(pcg::ScalarVectorPCG, A::AbstractArray) = rand!(pcg.vector, A)
@inline Random.randn!(pcg::ScalarVectorPCG, A::AbstractArray) = randn!(pcg.vector, A)
@inline Random.randexp!(pcg::ScalarVectorPCG, A::AbstractArray) = randexp!(pcg.vector, A)
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
@inline Random.rng_native_52(pcg::ScalarVectorPCG) = Random.rng_native_52(pcg.scalar)

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
    P = 4
    W = VectorizationBase.pick_vector_width(Float64)
    rngs = Vector{ScalarVectorPCG{P}}(undef, N)
    seeds = rand(UInt64, P*W*N)
    myprocid = myid()-1
    Base.Threads.@threads for n ∈ 1:N
        rngs[n] = ScalarVectorPCG(
            RandomNumbers.PCG.PCGStateUnique(PCG.PCG_RXS_M_XS),
            VectorizedRNG.PCG(ntuple(w -> seeds[w+W*P*((n-1))], Val(P*W)), (n-1) + N*myprocid )
        )
    end
    rngs
end
