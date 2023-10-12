module IntiHMatricesExt

import Inti
import HMatrices

function __init__()
    @info "Loading Inti.jl HMatrices extension"
end

# HMatrices interface
function HMatrices.assemble_hmatrix(
    iop::Inti.IntegralOperator;
    atol = 0,
    rank = typemax(Int),
    rtol = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
    eta = 3,
    kwargs...,
)
    comp = HMatrices.PartialACA(; rtol, atol, rank)
    adm  = HMatrices.StrongAdmissibilityStd(eta)
    X    = [Inti.coords(x) for x in iop.target]
    Y    = [Inti.coords(y) for y in iop.source]
    Xclt = HMatrices.ClusterTree(X)
    Yclt = HMatrices.ClusterTree(Y)
    return HMatrices.assemble_hmatrix(iop, Xclt, Yclt; adm, comp, kwargs...)
end

end # module
