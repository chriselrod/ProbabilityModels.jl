
struct TrackedDict{K,V,D<:AbstractDict{K,V}} <: AbstractDict{K,V}
    d::D
    oldd::D
    reassigned::Dict{K,V}
end
function TrackedDict(d::AbstractDict{K,V}) where {K,V}
    TrackedDict(d, copy(d), Dict{K,V}())
end

Base.haskey(td::TrackedDict, k) = haskey(td.d, k)
Base.get(td::TrackedDict, k, v)  = get(td.d, k, v)
Base.getindex(td::TrackedDict, k) = td.d[k]
function Base.setindex!(td::TrackedDict, v, k)
    haskey(oldd, k) && (td.reassigned[k] = td.d[k])
    td.d[k] = v
end
