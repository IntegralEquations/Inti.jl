module IntiHMatricesExt

import Inti
import HMatrices

function __init__()
    return @info "Loading Inti.jl HMatrices extension"
end

# HMatrices interface
function Inti.assemble_hmatrix(
        iop::Inti.IntegralOperator;
        atol = 0,
        rank = typemax(Int),
        rtol = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
        eta = 3,
    )
    comp = HMatrices.PartialACA(; rtol, atol, rank)
    adm = HMatrices.StrongAdmissibilityStd(eta)
    X = [Inti.coords(x) for x in iop.target]
    Y = iop.target === iop.source ? X : [Inti.coords(y) for y in iop.source]
    Xclt = HMatrices.ClusterTree(X; copy_elements = false)
    Yclt = X === Y ? Xclt : HMatrices.ClusterTree(Y; copy_elements = false)
    return HMatrices.assemble_hmatrix(iop, Xclt, Yclt; adm, comp, global_index = true)
end

end # module
