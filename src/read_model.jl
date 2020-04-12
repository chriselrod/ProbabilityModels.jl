using ReverseDiffExpressions: Model, Variable, Func, getvar!, returns!, uses!, onevar, targetvar, addfunc!, isref
using LoopVectorization: Instruction

read_model(q::Expr, mod::Module) = read_model!(Model(mod), q)

function read_model!(m::Model, q::Expr)
    for arg ∈ q.args
        arg isa Expr || continue
        ex = arg::Expr
        if ex.head === :for
            add_loopset!(m, q)
        elseif ex.head === :block
            read_model!(m, ex)
        else
            read_line!(m, ex)
        # else
            # throw("Don't know how to handle block $ex")
        end
    end
    for (varid, v) ∈ enumerate(m.vars)
        vid = v.varid
        @show v
        @assert varid == vid + 1
        if v.initialized && !isref(v)
            # If it must already be initialized, yet it isn't a ref to a constant
            push!(m.inputvars, vid)
        end
    end
    m
end

function add_loopset!(m::Model, ex)
    throw("Not yet implemented")
    ls = LoopSet(stage1distsub!(ex), Symbol(m.mod))
end

function read_line!(m::Model, ex::Expr)
    if ex.head === :(=)
        LHS = first(ex.args)::Union{Symbol,Expr}
        # If LHS isa Expr, we assume it is a tuple being unpacked
        RHS = ex.args[2]
        read_call!(m, RHS, LHS)
    elseif ex.head === :(.=)
        read_broadcast!(m, ex)
    elseif ex.head === :call
        read_call!(m, ex)
    end
end

# Reads function arguments, returning a Variable
read_argument!(m::Model, s::Symbol)::Variable = getvar!(m, s)
function read_argument!(m::Model, ex::Expr)::Variable
    if ex.head === :call
        # To unnest the expression, we assign to LHS
        read_call!(m, ex, gensym(:LHS))
    elseif ex.head === :(.)
        read_broadcast_tempalloc!(m, ex, gensym(:LHS))
    elseif ex.head === :ref
        read_ref!(m, ex, gensym(:LHS))
    elseif ex.head === Symbol("'")
        read_call!(m, Instruction(:adjoint), ex.args, gensym(:LHS))
    else
        println(ex)
        @show ex.head === Symbol("'")
        @show ex.head, Symbol("'")
        throw("Expression not found.")
    end
end
function read_argument!(m::Model, x)::Variable
    xv = getvar!(m, gensym())
    xv.ref = x
    # xv.initialized = true
    xv
end
ReverseDiffExpressions.uses!(func::Func, m::Model, x) = uses!(func, read_argument!(m, x))

function read_ref!(m::Model, ex::Expr, LHS)
    retv = getvar!(m, LHS)
    func = Func(Instruction(:Base,:view), false, false)
    returns!(func, retv)
    foreach(arg -> uses!(func, m, arg), ex.args)
    addfunc!(m, func)
end

function read_broadcast!(m::Model, ex::Expr)
    read_broadcast!(m, first(Meta.lower(m.mod, ex).args)::Core.CodeInfo)
end
function read_broadcast!(m::Model, ci::Core.CodeInfo)
    
end
instr_from_expr(ex::Expr) = Instruction(first(ex.args)::Union{Symbol,Expr})::Instruction
function LoopVectorization.Instruction(ex::Expr)
    ex.head === :(.) || throw("Could not parse instruction $ex.")
    Instruction((ex.args[1])::Symbol, (((ex.args[2])::QuoteNode).value)::Symbol)
end

function read_sampling_statement!(m::Model, f::Instruction, ex::Expr)
    arg1 = (ex.args[2])::Symbol
    call = (ex.args[3])::Expr
    @assert call.head === :call
    v = getvar!(m, arg1)
    f = instr_from_expr(call)
    scale = if f == Instruction(:^) # Then it is weighted
        call = (call.args[2])::Expr
        f = instr_from_expr(call)
        read_argument!(m, call.args[3])
    else
        onevar(m)
    end
    func = Func(f, false, true)
    target = targetvar(m)
    returns!(func, target)
    uses!(func, v)
    for i ∈ 2:length(call.args)
        uses!(func, m, call.args[i])
    end
    uses!(func, scale)
    addfunc!(m, func)
end

# Reads a call without a return
function read_call!(m::Model, ex::Expr, ::Nothing = nothing)
    f = instr_from_expr(ex)
    if f.instr === :~
        return read_sampling_statement!(m, f, ex)
    else
        @show ex
        throw("Currently only `~` and `=` type calls are supported.")    
    end
end
# Reads a call with a return
function read_call!(m::Model, ex::Expr, LHS::Symbol)
    broadcasts = (:(.+), :(.*), :(.-), :(./), :(.*ˡ), :(.÷))
    f = instr_from_expr(ex)
    # @show ex
    if f.instr ∈ broadcasts
        return read_broadcast!(m, ex, LHS)
    # elseif f.instr === :~
        # return read_sampling_statement!(m, ex)
    elseif f.instr === :(:)
        return read_range_expr!(m, ex, LHS)
    end
    read_call!(m, f, @view(ex.args[2:end]), LHS)
end
function read_call!(m::Model, f::Instruction, args, LHS)
    func = Func(f, false, false)
    retv = getvar!(m, LHS)
    returns!(func, retv)
    foreach(arg -> uses!(func, m, arg), args)
    addfunc!(m, func)
end

function read_range_expr!(m::Model, ex::Expr, LHS)
    if length(ex.args) == 3
        read_range_args!(m, ex.args[2], ex.args[3], LHS)
    else
        @assert length(ex.args) == 4
        read_range_args!(m, ex.args[2], ex.args[3], ex.args[4], LHS)
    end
end
function read_range_args!(m::Model, l::Number, u::Number, LHS::Symbol)
    retv = getvar!(m, LHS)
    retv.ref = Expr(:call, Expr(:curly, :StaticUnitRange, l, u))
    # retv.initialized = true
    retv
end
function read_range_args!(m::Model, l, u, LHS::Symbol)
    lin = l isa Number; uin = u isa Number
    f = if lin
        :StaticUpperUnitRange
    elseif uin
        :StaticLowerUnitRange
    else
        :(:)
    end
    func = Func(Instruction(f), false, false)
    retv = getvar!(m, LHS)
    returns!(func, retv)
    uses!(func, m, l)
    uses!(func, m, u)
    addfunc!(m, func)
end
# function read_range_args!(m::Model, l, s, u, LHS::Symbol)
# end

