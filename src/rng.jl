

function threadrandinit!(pcg_vector::Vector{PtrPCG{P}}) where {P}
    N = Base.Threads.nthreads()
    W = VectorizationBase.pick_vector_width(Float64)
    myprocid = myid()-1
    local_stack_size = LOCAL_STACK_SIZE[]
    stack_ptr = STACK_POINTER_REF[]
    for n ∈ 1:N
        rng = VectorizedRNG.random_init_pcg!(PtrPCG{4}(stack_ptr), P*(n-1)*myprocid)
        if n > length(pcg_vector)
            push!(pcg_vector, rng)
        else
            pcg_vector[n] = rng
        end
        stack_ptr += local_stack_size
    end
    nothing
end

function DynamicHMC.rand_p(sp::StackPointer, rng::VectorizedRNG.AbstractPCG, κ::GaussianKineticEnergy{S,S}, q = nothing) where {S <: Diagonal}
    (sp, rp) = similar(sp, W)
    sp, randn!(rng, rp, κ.W.diag)
end

@inline DynamicHMC.rand_bool(pcg::VectorizedRNG.AbstractPCG, prob::Float64) = Float64(rand(pcg, UInt64)) < typemax(UInt64) * prob
@inline DynamicHMC.rand_bool(pcg::VectorizedRNG.AbstractPCG, prob::Float32) = Float32(rand(pcg, UInt32)) < typemax(UInt32) * prob
function DynamicHMC.random_position(pcg::VectorizedRNG.AbstractPCG, N)
    # if we really want to optimize it, we would calculate r in one pass instead of two.
    r = Vector{Float64}(undef, N)
    rand!(pcg, r, -2.0, 2.0)
    r
end
function DynamicHMC.random_position(sp::StackPointer, pcg::VectorizedRNG.AbstractPCG, ::Union{Val{N},PaddedMatrices.Static{N}}) where {N}
    # if we really want to optimize it, we would calculate r in one pass instead of two.
    sp, r = PtrVector{N,Float64}(sp)
    rand!(pcg.vector, r, -2.0, 2.0)
    sp, r
end

