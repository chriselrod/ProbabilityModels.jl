using ReverseDiffExpressions: Model, Variable, Func
using LoopVectorization: Instruction

function read_model(q::Expr, mod::Module)
    m = Model(mod)
    read_model!(m, q)
end

function read_model!(m::Model, q::Expr)
    for arg ∈ q.args
        arg isa Expr || continue
        ex = arg::Expr
        if ex.head === :for
            ls = LoopSet(ex, Symbol(m.mod))
        elseif ex.head === :block
            read_model!(m, ex)
        else
            read_line!(m, ex)
        end
        else
            throw("Don't know how to handle block $ex")
        end
    end
end

function read_line!(m::Model, ex::Expr)
    if ex.head === :(=)
        LHS = first(ex.args)::Union{Symbol,Expr}
        # If LHS isa Expr, we assume it is a tuple being unpacked
        RHS = ex.args[2]
        read_call!(m, RHS)
    elseif ex.head === :(.=)
        read_broadcast!(m, ex)
    elseif ex.head === :call
        read_call!(m, ex)
    end
end

# Reads ex, returns a variable
function read_operation!(m::Model, ex::Expr)
    if ex.head === :call
    elseif ex.head === :(.)
    elseif ex.head === :ref
        read_ref!(m, ex)
    end

end

function read_broadcast!(m::Model, ex::Expr)
    read_broadcast!(m, first(Meta.lower(m.mod, ex).args)::CodeInfo)
end
function read_broadcast!(m::Model, ci::CodeInfo)
    
end
instr_from_expr(ex::Expr) = Instruction(first(ex.args)::Union{Symbol,Expr})::Instruction
function LoopVectorization.Instruction(ex::Expr)
    ex.head === :(.) || throw("Could not parse instruction $ex.")
    Instruction((ex.args[1])::Symbol, (((ex.args[2])::QuoteNode).value)::Symbol)
end

# Reads ex, returns a variable
read!(m::Model, ex::Expr)::Variable = read_operation!(m, ex)
read!(m::Model, s::Symbol)::Variable = getvar!(m, s)

function read_sampling_statement!(m::Model, f::Instruction, ex::Expr)
    arg1 = (ex.args[2])::Symbol
    call = (ex.args[3])::Expr
    @asert call.head === :call
    v = getvar!(m, arg1)
    f = instr_from_expr(call)
    scale = if f == Instruction(:^) # Then it is weighted
        call = (call.args[2])::Expr
        f = instr_from_expr(call)
        read!(m, call.args[3])
    else
        onevar(m)
    end
    func = addfunc!(m, f, false, true)
    uses!(f, v)
    for i ∈ 2:length(call.args)
        arg = read!(m, call.args[i])
        uses!(f, arg)
    end
    uses!(f, scale)
end
function read_call!(m::Model, ex::Expr, ::Nothing = nothing)
    f = instr_from_expr(ex)
    if f.instr === :~
        return read_sampling_statement!(m, f, ex)
    end

    
end
function read_call!(m::Model, ex::Expr, LHS::Symbol)
    broadcasts = (:(.+), :(.*), :(.-), :(./), :(.*ˡ), :(.÷))
    f = instr_from_expr(ex)
    if f.instr ∈ broadcasts
        return read_broadcast!(m, ex)
    elseif f.instr === :~
        return read_sampling_statement!(m, ex)
    end
    
end


