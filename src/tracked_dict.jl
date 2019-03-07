
struct TrackedDict{K,V,D<:AbstractDict{K,V}} <: AbstractDict{K,V}
    d::D
    oldd::D
    reassigned::Dict{K,Tuple{V,V}}
    newlyassigned::Dict{K,V}
end
function TrackedDict(d::AbstractDict{K,V}) where {K,V}
    TrackedDict(copy(d), d, Dict{K,Tuple{V,V}}(), Dict{K,V}())
end

Base.haskey(td::TrackedDict, k) = haskey(td.d, k)
Base.get(td::TrackedDict, k, v)  = get(td.d, k, v)
Base.getindex(td::TrackedDict, k) = td.d[k]
function Base.setindex!(td::TrackedDict, v, k)
    if haskey(td.oldd, k)
        td.reassigned[k] = (td.oldd[k],v)
    else
        td.newlyassigned[k] = v
    end
    td.d[k] = v
end
