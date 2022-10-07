module ForcedKernel

using KernelFunctions
using LinearAlgebra
# Write your package code here.

export GP_α,GP_pred

function GP_α(x, y, k) # Compute α for kernel ridge regression without noise
    K = kernelmatrix(k, x, x)
    eps = 1e-9
    K = K + eps * I
    CK = cholesky(Symmetric(K))
    α = CK.U \ (CK.U' \ y)
    return (α=α, k=k)
end

function GP_α(x, y, k,eps) # Compute α for kernel ridge regression with noise
    K_ = kernelmatrix(k, x, x)
    K = K_ + (abs(eps)+1e-8) * I
    CK = cholesky(Symmetric(K))
    α = CK.U \ (CK.U' \ y)
    return (α=α, k=k,K=K_)
end

function GP_pred(uu, α,xt,kk) # Compute α=(XX)^{-1}Y for GP_ODE kernel X:kernelmatrix, Y:mesured output
    K_ = kernelmatrix(kk, uu, xt)
    pp = K_ * α
    return pp
end

end
