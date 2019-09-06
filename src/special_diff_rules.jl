
# using MacroTools, DiffRules
# using MacroTools: @capture, postwalk, prewalk, @q, striplines
# DiffRules.hasdiffrule(:Base, :exp, 1)
# DiffRules.diffrule(:Base, :exp, :x)
# DiffRules.diffrule(:Base, :^, :x, :y)


const SPECIAL_DIFF_RULES = Dict{Symbol,Function}()
function exp_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    a = A[1]
    push!(first_pass.args, :($out = ProbabilityModels.SLEEFPirates.exp($a)))
    a ∈ tracked_vars || return nothing
    push!(tracked_vars, out)
    # ∂ = gensym(:∂)
    ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
    push!(first_pass.args, :($∂ = $out))
    pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($(Symbol("###seed###", out)), $∂, $(Symbol("###seed###", a)) )))
    nothing
end
SPECIAL_DIFF_RULES[:exp] = exp_diff_rule!
function vexp_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    a = A[1]
    push!(first_pass.args, :($out = ProbabilityModels.PaddedMatrices.vexp($a)))
    a ∈ tracked_vars || return nothing
    push!(tracked_vars, out)
    # ∂ = gensym(:∂)
    ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
    push!(first_pass.args, :($∂ = Diagonal($out)))
    pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($(Symbol("###seed###", out)), $∂, $(Symbol("###seed###", a)) )))
    nothing
end
SPECIAL_DIFF_RULES[:vexp] = vexp_diff_rule!
function log_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    a = A[1]
    push!(first_pass.args, :($out = ProbabilityModels.SLEEFPirates.log($a)))
    a ∈ tracked_vars || return nothing
    push!(tracked_vars, out)
    # ∂ = gensym(:∂)
    ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
    push!(first_pass.args, :($∂ = inv($a)))
    pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($(Symbol("###seed###", out)), $∂, $(Symbol("###seed###", a)) )))
    nothing
end
SPECIAL_DIFF_RULES[:log] = log_diff_rule!
function plus_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    push!(first_pass.args, :($out = +($(A...)) ))
    track_out = false
    adjout = Symbol("###seed###", out)
    for i ∈ eachindex(A)
        a = A[i]
        a ∈ tracked_vars || continue
        track_out = true
        pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($adjout, $(Symbol("###seed###", a)))))
    end
    track_out && push!(tracked_vars, out)
    nothing
end
SPECIAL_DIFF_RULES[:+] = plus_diff_rule!
# add is specifically for DistributionsParameters.Target
function add_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    push!(first_pass.args, :($out = ProbabilityModels.SIMDPirates.vadd($(A...))))
    track_out = false
    adjout = Symbol("###seed###", out)
    for i ∈ eachindex(A)
        a = A[i]
        a ∈ tracked_vars || continue
        track_out = true
        pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($adjout, $(Symbol("###seed###", a)))))
    end
    track_out && push!(tracked_vars, out)
    nothing
end
SPECIAL_DIFF_RULES[:vadd] = add_diff_rule!
function minus_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    @assert length(A) == 2
    a₁ = A[1]
    a₂ = A[2]
    push!(first_pass.args, :($out = $a₁ - $a₂ ))
    adjout = Symbol("###seed###", out)
    a₁ ∈ tracked_vars && pushfirst!(second_pass.args, :( $(Symbol("###seed###", a₁)) = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($adjout, $(Symbol("###seed###", a₁)))))
    a₂ ∈ tracked_vars && pushfirst!(second_pass.args, :( $(Symbol("###seed###", a₂)) = ProbabilityModels.PaddedMatrices.RESERVED_DECREMENT_SEED_RESERVED($adjout, $(Symbol("###seed###", a₂)))))
    track_out = (a₁ ∈ tracked_vars) || (a₂ ∈ tracked_vars)
    track_out && push!(tracked_vars, out)
    nothing
end
SPECIAL_DIFF_RULES[:-] = minus_diff_rule!
function inv_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    a = A[1]
    if a ∉ tracked_vars
        push!(first_pass.args, :($out = inv($a)))
        return nothing
    end
    push!(tracked_vars, out)
    # ∂ = gensym(:∂)
    ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
    push!(first_pass.args, :(($out, $∂) = ProbabilityModels.StructuredMatrices.∂inv($a)))
    pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($(Symbol("###seed###", out)), $∂, $(Symbol("###seed###", a)) )))
    nothing
end
SPECIAL_DIFF_RULES[:inv] = inv_diff_rule!
function inv′_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    a = A[1]
    if a ∉ tracked_vars
        push!(first_pass.args, :($out = ProbabilityModels.StructuredMatrices.inv′($a)))
        return nothing
    end
    push!(tracked_vars, out)
    # ∂ = gensym(:∂)
    ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
    push!(first_pass.args, :(($out, $∂) = ProbabilityModels.StructuredMatrices.∂inv′($a)))
    pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($(Symbol("###seed###", out)), $∂, $(Symbol("###seed###", a)) )))
    nothing
end
SPECIAL_DIFF_RULES[:inv′] = inv′_diff_rule!

function mul_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    @assert length(A) == 2
    a1 = A[1]
    a2 = A[2]
    push!(first_pass.args, :($out = $a1 * $a2))
    if (a1 ∈ tracked_vars) || (a2 ∈ tracked_vars)
        push!(tracked_vars, out)
    else
        return
    end
    adjout = Symbol("###seed###", out)
    seed2 = Symbol("###seed###", a2)
    track_tup = Expr(:tuple,)
    return_expr = Expr(:tuple,)
    for i ∈ 1:2
        a = A[i]
        if a ∈ tracked_vars
            seeda = Symbol("###seed###", a)
            ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
            pushfirst!(second_pass.args, :( $seeda = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($adjout, $∂, $seeda)))
            push!(return_expr.args, ∂)
            push!(track_tup.args, true)
        else
            push!(track_tup.args, false)
        end
    end
    pushfirst!(second_pass.args, :($(ProbabilityDistributions.return_expression(return_expr)) = ProbabilityModels.∂mul($a1, $a2, Val{$track_tup}())))
    nothing
end
SPECIAL_DIFF_RULES[:*] = mul_diff_rule!


function itp_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    ∂tup = Expr(:tuple, out)
    seedout = Symbol("###seed###", out)
    track_out = false
    track_tup = Expr(:tuple,)
    # we skip the first argument, time.
    for i ∈ 2:length(A)
        a = A[i]
        if a ∈ tracked_vars
            track_out = true
            push!(track_tup.args, true)
            ∂a = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
            seeda = Symbol("###seed###", a)
            push!(∂tup.args, ∂a)
            pushfirst!(second_pass.args, :( $seeda = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($seedout, $∂a, $seeda )))
        else
            push!(track_tup.args, false)
        end
    end
    track_out && push!(tracked_vars, out)
    push!(first_pass.args, :( $(ProbabilityDistributions.return_expression(∂tup)) = ProbabilityModels.∂ITPExpectedValue($(A...), Val{$track_tup}())))
    nothing
end
SPECIAL_DIFF_RULES[:ITPExpectedValue] = itp_diff_rule!

function hierarchical_centering_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    # fourth arg would be Domains, which are not differentiable.
    length(A) == 4 && @assert A[4] ∉ tracked_vars

    func_output = Expr(:tuple, out)
    tracked = ntuple(i -> A[i] ∈ tracked_vars, Val(3))
    any(tracked) && push!(tracked_vars, out)
    seedout = Symbol("###seed###", out)
    for i ∈ 1:3
        a = A[i]
        if tracked[i]
            ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
            push!(func_output.args, ∂)
            seeda = Symbol("###seed###", a)
            pushfirst!(second_pass.args, :( $seeda = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($seedout, $∂, $seeda )))
        end
    end
    push!(first_pass.args, :($func_output = ∂HierarchicalCentering($(A...), Val{$tracked}()) ) )
    nothing
end
SPECIAL_DIFF_RULES[:HierarchicalCentering] = hierarchical_centering_diff_rule!

function tuple_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    for i ∈ eachindex(A)
        a = A[i]
        if a ∈ tracked_vars
            push!(tracked_vars, out)
            seedout = Symbol("###seed###", out)

            #∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
            #push!(func_output.args, ∂)
            seeda = Symbol("###seed###", a)
            pushfirst!(second_pass.args, :( $seeda = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($seedout[$i], $seeda )))
        end
    end
    push!(first_pass.args, :($out = Core.tuple($(A...))))
    nothing
end
SPECIAL_DIFF_RULES[:tuple] = tuple_diff_rule!

function diagonal_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    @assert length(A) == 1
    a = A[1]
    if a ∈ tracked_vars
        push!(tracked_vars, out)
        seeda = Symbol("###seed###", a)
        seedout = Symbol("###seed###", out)
#        pushfirst!(second_pass.args, :(@show $seeda))
        pushfirst!(second_pass.args, :( $seeda = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($seedout, $seeda )))
#        pushfirst!(second_pass.args, :(@show $seedout))
#        pushfirst!(second_pass.args, :(println("diagonal diff reverse pass")))
    end
    push!(first_pass.args, :($out = LinearAlgebra.Diagonal($a)))
    nothing
end
SPECIAL_DIFF_RULES[:Diagonal] = diagonal_diff_rule!

function vec_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    @assert length(A) == 1
    a = A[1]
#    println("A: $A")
#    println("out: $out")
#    println("A[1] ∈ tracked: $(A[1] ∈ tracked_vars)")
    if a ∈ tracked_vars
        ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
        push!(tracked_vars, out)
        seeda = Symbol("###seed###", a)
        seedout = Symbol("###seed###", out)
        pushfirst!(second_pass.args, :( $seeda = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($seedout, $∂, $seeda )))
        push!(first_pass.args, :(($out,$∂) = ProbabilityModels.∂vec($a)))
    else
        push!(first_pass.args, :($out = vec($a)))
    end
#    println("out ∈ tracked: $(out ∈ tracked_vars)")
    nothing
end
SPECIAL_DIFF_RULES[:vec] = vec_diff_rule!

function reshape_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    @assert length(A) == 2
    a = A[1]
    shape = A[2]
#    println("A: $A")
#    println("out: $out")
#    println("A[1] ∈ tracked: $(A[1] ∈ tracked_vars)")
    push!(first_pass.args, :($out = reshape($a,$shape)))
    if a ∈ tracked_vars
 #       ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
        push!(tracked_vars, out)
        seeda = Symbol("###seed###", a)
        seedout = Symbol("###seed###", out)
        pushfirst!(second_pass.args, :( $seeda = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(reshape($seedout, ProbabilityModels.PaddedMatrices.maybe_static_size($a)), $seeda) ))
#        push!(first_pass.args, :(($out,$∂) = ProbabilityModels.∂vec($a)))
    end
#    println("out ∈ tracked: $(out ∈ tracked_vars)")
    nothing
end
SPECIAL_DIFF_RULES[:reshape] = reshape_diff_rule!


function cov_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    # For now, the only method is autoregressive * longitudinal model
    # so we assert that there are precisely three args.
    length(A) == 3 || throw("Please request or add support for different CovarianceMatrix functions!")
#    @assert length(A) == 3
    @assert A[3] ∉ tracked_vars
    func_output = Expr(:tuple, out)
    tracked = ntuple(i -> A[i] ∈ tracked_vars, Val(2))
    any(tracked) && push!(tracked_vars, out)
    seedout = Symbol("###seed###", out)
    for i ∈ 1:2
        a = A[i]
        if tracked[i]
            ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
            push!(func_output.args, ∂)
            seeda = Symbol("###seed###", a)
            pushfirst!(second_pass.args, :( $seeda = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($seedout, $∂, $seeda )))
        end
    end
    push!(first_pass.args, :($func_output = ProbabilityModels.DistributionParameters.∂CovarianceMatrix($(A...), Val{$tracked}()) ) )
    nothing
end
SPECIAL_DIFF_RULES[:CovarianceMatrix] = cov_diff_rule!

function getindex_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    for i ∈ 2:length(A)
        @assert A[i] ∉ tracked_vars
    end
    a = A[1]
    if a ∈ tracked_vars
        ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
        push!(tracked_vars, out)
        seeda = Symbol("###seed###", a)
        seedout = Symbol("###seed###", out)
        pushfirst!(second_pass.args, :( $seeda = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($seedout, $∂, $seeda )))
        push!(first_pass.args, :(($out, $∂) = PaddedMatrices.∂getindex($(A...))))
    elseif a isa Expr && a.head == :tuple
        # terrible hack!!!!
        # TODO: DO THIS CORRECTLY
        # That is, have check for tuple packing and unpacking.
        
    else
        push!(first_pass.args, :($out = getindex($(A...))))
    end
    nothing
end
SPECIAL_DIFF_RULES[:getindex] = getindex_diff_rule!

function rank_update_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    # This function will have to be updated once we add rank updates for things other than
    # a cholesky decomposition.
    Lsym, xsym = A[1], A[2]
    track_L = Lsym ∈ tracked_vars
    track_x = xsym ∈ tracked_vars
    track = track_L | track_x
    push!(tracked_vars, out)
    push!(first_pass.args, Expr(:(=), out, :(StructuredMatrices.rank_update($(A...)))))
    track || return
    # That is because we differentiate by differentiating the expression:
    # out = chol( L * L' + x * x' )
    seedout = Symbol("###seed###", out)
    args = Symbol[out, seedout]
    seedL = Symbol("###seed###", Lsym)
    seedLtemp = gensym(seedL)
    if track_L
        # push!(ret.args, seedLtemp)
        push!(args, Lsym)
    end
    ∂L = Symbol("###adjoint###_##∂", out, "##∂", Lsym, "##")
    seedx = Symbol("###seed###", xsym)
    seedxtemp = gensym(seedx)
    if track_x
        # push!(ret.args, seedxtemp)
        push!(args, xsym)
    end
    ∂x = Symbol("###adjoint###_##∂", out, "##∂", xsym, "##")
    seedchol = gensym(:seedchol)
    if track_L && track_x
        ret = Expr(:tuple, seedLtemp, seedxtemp)
    elseif track_L
        ret = seedLtemp
    else#if track_x
        ret = seedxtemp
    end
    q = quote
        #$seedchol = reverse_cholesky_grad($out, $seedout) # 
        $ret = StructuredMatrices.∂rank_update($(args...))
    end
    track_L && push!(q.args, :($seedL = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($seedLtemp, $seedL)))
    track_x && push!(q.args, :($seedx = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED($seedxtemp, $seedx)))
    pushfirst!(second_pass.args, q)
    nothing
end
SPECIAL_DIFF_RULES[:rank_update] = rank_update_diff_rule!
