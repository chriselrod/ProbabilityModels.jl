

function real_data_or_param!(g, i, default, ::Type{T} = Float64; lower = -Inf, upper = Inf) where T
    if lower == -Inf && upper == Inf
        init = :(Float{$T})
        out = :(Union{$T, $init})
    elseif lower == 0 && upper == 1
        init = :(ProbabilityFloat{$T})
        out = :(Union{$T, $init})
    elseif lower == 0 && upper == Inf
        init = :(PositiveFloat{$T})
        out = :(Union{$T, $init})
    elseif lower != -Inf && upper == Inf
        init = :(LowerBoundFloat{$T,$lower})
        out = :(Union{$T, $init})
    elseif lower == -Inf && upper != Inf
        init = :(UpperBoundFloat{$T,$upper})
        out = :(Union{$T, $init})
    else
        init = :(BoundedFloat{$T,$lower,$upper})
        out = :(Union{$T, $init})
    end
    default == nothing ? set_prop!(g, i, :default, init) : set_prop!(g, i, :default, default)
    out
end

function vector_data_or_param!(g, i, L, default, ::Type{T} = Float64; lower = -Inf, upper = Inf) where T
    if lower == -Inf && upper == Inf
        init = :(SizedVector{$L,$T})
        out = :(Union{SizedSIMDVector{$L,$T,$L,$L}, $init})
    elseif lower == 0 && upper == 1
        init = :(SizedProbabilityVector{$L,$T})
        out = :(Union{SizedSIMDVector{$L,$T,$L,$L}, $init})
    elseif lower == 0 && upper == Inf
        init = :(SizedPositiveVector{$L,$T})
        out = :(Union{SizedSIMDVector{$L,$T,$L,$L}, $init})
    elseif lower != -Inf && upper == Inf
        init = :(SizedLowerBoundVector{$L,$T,$lower})
        out = :(Union{SizedSIMDVector{$L,$T,$L,$L}, $init})
    elseif lower == -Inf && upper != Inf
        init = :(SizedUpperBoundVector{$L,$T,$upper})
        out = :(Union{SizedSIMDVector{$L,$T,$L,$L}, $init})
    else
        init = :(SizedBoundedVector{$L,$T,$lower,$upper})
        out = :(Union{SizedSIMDVector{$L,$T,$L,$L}, $init})
    end
    default == nothing ? set_prop!(g, i, :default, init) : set_prop!(g, i, :default, default)
    out
end

function matrix_data_or_param!(g, i, M, N, default, ::Type{T} = Float64; lower = -Inf, upper = Inf) where T
    R, L = SIMDArrays.calculate_L_from_size((M,N), T)
    if lower == -Inf && upper == Inf
        init = :(SizedMatrix{$M,$N,$T,$R,$L})
        out = :(Union{SizedSIMDMatrix{$M,$N,$T,$R,$L}, $init})
    elseif lower == 0 && upper == 1
        init = :(SizedProbabilityMatrix{$M,$N,$T,$R,$L})
        out = :(Union{SizedSIMDMatrix{$M,$N,$T,$R,$L}, $init})
    elseif lower == 0 && upper == Inf
        init = :(SizedPositiveMatrix{$M,$N,$T,$R,$L})
        out = :(Union{SizedSIMDMatrix{$M,$N,$T,$R,$L}, $init})
    elseif lower != -Inf && upper == Inf
        init = :(SizedLowerBoundMatrix{$M,$N,$T,$lower,$R,$L})
        out = :(Union{SizedSIMDMatrix{$M,$N,$T,$R,$L}, $init})
    elseif lower == -Inf && upper != Inf
        init = :(SizedUpperBoundMatrix{$M,$N,$T,$upper,$R,$L})
        out = :(Union{SizedSIMDMatrix{$M,$N,$T,$R,$L}, $init})
    else
        init = :(SizedBoundedMatrix{$M,$N,$T,$lower,$upper,$R,$L})
        out = :(Union{SizedSIMDMatrix{$M,$N,$T,$R,$L}, $init})
    end
    default == nothing ? set_prop!(g, i, :default, init) : set_prop!(g, i, :default, default)
    out
end
