

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
