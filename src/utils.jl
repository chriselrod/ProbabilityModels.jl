using MacroTools: @capture, walk, postwalk, prewalk, isexpr, isgensym
using LightGraphs

f(x,y) = (x*y+1.0)*(x+2.0y)
fexpr = :((x*y+1.0)*(x+2.0y))
gexpr = :(g(x*y+1.0,h(x,k(2.0,y))))

# Shows expressions from the top down.
postwalk(gexpr) do x
    isexpr(x) ? (@show x) : x
end

default_tracked(x::Symbol) = !isgensym(x)
default_tracked(x) = false

tracked(x) = x ∈ (:x, :y) ? true : false

function graph_expr(expr, tracked = default_tracked)
    g = SimpleDiGraph()
    node_to_index = Dict{Symbol,Int}()
    index_to_node = Symbol[]
    # node_exprs = Expr[]
    # tracked = Int[]
    out_expr = quote end
    iter = 0
    postwalk(expr) do x
        if isexpr(x, :call)
            iter += 1
            gx = gensym(:x)
            push!(out_expr.args, :($gx = $x))
            node_to_index[gx] = iter; push!(index_to_node, gx)
            add_vertex!(g)
            for i ∈ 2:length(x.args)
                node = x.args[i]
                if node ∈ keys(node_to_index)
                    add_edge!(g, node_to_index[node], iter)
                end
            end
            return gx
        elseif isexpr(x, :(=))
            push!(out_expr.args, x)
            return x
        else
            if tracked(x) && x ∉ keys(node_to_index)
                iter += 1
                node_to_index[x] = iter; push!(index_to_node, x)
                add_vertex!(g)
            end
            return x
        end
    end
    out_expr, g, node_to_index, index_to_node
end

q, g, nti, itn = graph_expr(gexpr, tracked)
for e ∈ edges(g) println(e) end
nti
q
