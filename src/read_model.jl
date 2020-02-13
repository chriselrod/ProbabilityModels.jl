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
            read_operation!(m, ex)
        end
        else
            throw("Don't know how to handle block $ex")
        end
    end
end

function read_operation!(m::Model, ex::Expr)
    if ex.head === :(=)
    elseif ex.head === :(.=) || ex.head === :(.)
        read_broadcast!(m, ex)
    elseif ex.head === :call
        read_call!(m, ex)
    elseif ex.head === :ref
        read_ref!(m, ex)
    end
end

function read_broadcast!(m::Model, ex::Expr)
    read_broadcast!(m, first(Meta.lower(m.mod, ex).args)::CodeInfo)
end
function read_broadcast!(m::Model, ci::CodeInfo)
    
end

function read_call!(m::Model, ex::Expr)
    broadcasts = (:(.+), :(.*), :(.-), :(./), :(.*ˡ), :(.÷))
    f = Instruction(first(ex.args))::Instruction
    if f.instr ∈ broadcasts
        return read_broadcast!(m, ex)
    elseif f.instr === :~
        return read_sampling_statement!(m, ex)
    end
    
end


