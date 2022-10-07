using ForcedKernel
using OrdinaryDiffEq
using Plots
using Statistics
using PGFPlotsX
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using KernelFunctions
using NLsolve
using Optim

BLAS.set_num_threads(1)
function nl_4(u,p,t) # Equation of motion -- Shaw & Pierre example
    k=1.
    c1=0.003
    c2=c1/sqrt(3)
    κ=0.003

    L=@SMatrix [0 1 0 0 ;-2*k -c1-c2 k c2 ;0 0 0 1 ;k c2 -2k -c1-c2]
    du_temp=L*u
    add_vec=@SVector [0,-κ*u[1]^3+F0*cos(p[1]*t),0,F0*cos(p[1]*t)]
    du=du_temp+add_vec
    return du
end

function LS_harmonics(r, t, ω, N) # Computing Fourier coefficients of the amplitude in the measued state-variable coordinates
    # Fourier coefficients are computed in least square sence
    c = Array{Float64}(undef, 2 * N + 1)
    M = Array{Float64}(undef, 1, 2 * N + 1)
    tM = Array{Float64}(undef, 0, 2 * N + 1)
    tl = length(t)
    rr = Array{Float64}(undef, tl)
    M[1] = 1
    for j in 1:tl
        for i in 1:N
            M[1 + i] = cos(ω * t[j] * i)
            M[1 + N + i] = sin(ω * t[j] * i)
        end
        tM = vcat(tM, M)
    end
    MM = transpose(tM) * tM
    rN = transpose(tM) * r
    c = inv(MM) * rN
    for j in 1:tl
        rr[j] = c[1]
        for i in 1:N
            rr[j] += c[i + 1] * cos(ω * t[j] * i)
            rr[j] += c[i + 1 + N] * sin(ω * t[j] * i)
        end
    end
    return (coeff=c, rr=rr)
end

function fourier_diff(T::Type{<:Number}, N::Integer; order=1)
    D = zeros(T, N, N)
    n1 = (N - 1) ÷ 2
    n2 = N ÷ 2
    x = LinRange{T}(0, π, N+1)
    if order == 1
        for i in 2:N
            sgn = (one(T)/2 - iseven(i))
            D[i, 1] = iseven(N) ? sgn*cot(x[i]) : sgn*csc(x[i])
        end
    elseif order == 2
        D[1, 1] = iseven(N) ? -N^2*one(T)/12 - one(T)/6 : -N^2*one(T)/12 + one(T)/12
        for i in 2:N
            sgn = -(one(T)/2 - iseven(i))
            D[i, 1] = iseven(N) ? sgn*csc(x[i]).^2 : sgn*cot(x[i])*csc(x[i])
        end
    else
        error("Not implemented")
    end
    for j in 2:N
        D[1, j] = D[N, j-1]
        D[2:N, j] .= D[1:N-1, j-1]
    end
    return D
end
fourier_diff(N::Integer; kwargs...) = fourier_diff(Float64, N; kwargs...)

function collocation_setup(u::AbstractMatrix) # setting up collocation parameters
    return (ndim=size(u, 1), nmesh=size(u, 2), Dt=-fourier_diff(eltype(u), size(u, 2))*2π)
end

function collocation!(res, f, u, p, T, coll) # Collocation equation
    # Matrix of derivatives along the orbit
    D = reshape(u, (coll.ndim, coll.nmesh))*coll.Dt
    ii = 1:coll.ndim
    for i in 1:coll.nmesh
        # Subtract the desired derivative from the actual derivative
        res[ii] .= D[ii] .- T.*f(u[ii], p, T*(i-1)/coll.nmesh)
        ii = ii .+ coll.ndim
    end
    return res
end

function pseudo_arclength!(res, f, u, dv, pu, ds;nmesh=20) # Pseudo arclength equation for FRF
    umat = reshape(u[1:end-1], (:, nmesh))
    coll = collocation_setup(umat)
    res[1:end-1]=collocation!(res[1:end-1], f, u[1:end-1], u[end], 2π/u[end], coll)
    du = u - pu
    arclength = norm(transpose(dv) * du)
    res[end] = arclength - ds 
end

function initial_point(sp,ds,dp=0.01;nmesh=20) # Setting up ininitial point for FRF
    p1 = [sp]
    # Do initial value simulation to get a reasonable starting point
    prob = ODEProblem(nl_4, SVector(0.0, 0.0,0.0,0.0), (0.0, 100*2π/p1[1]), p1)
    odesol = solve(prob, Tsit5())
    # Refine using collocation
    t = range(0, 2π/p1[1], length=nmesh+1)[1:end-1]
    uvec = reinterpret(Float64, odesol(99*2π/p1[1] .+ t).u)
    umat = reshape(uvec, (:, nmesh))
    coll = collocation_setup(umat)
    nlsol1 = nlsolve((res, u) -> collocation!(res, nl_4, u, p1, 2π/p1[1], coll), uvec)
    p2 = [sp+dp]
    nlsol2 = nlsolve((res, u) -> collocation!(res, nl_4, u, p2, 2π/p2[1], coll), uvec)
    u1=vcat(nlsol1.zero,p1[1])
    u2=vcat(nlsol2.zero,p2[1])
    dv=(u2-u1)/norm(u2-u1)
    pu=u2
    u_pred=pu+dv*ds
    return (dv=dv,pu=pu,u_pred=u_pred)
end

function continuation_forced(num_p,sp,ds,dp=0.01;nmesh=20) # Numerical continuation of FRF
    ip=initial_point(sp,ds,dp;nmesh=nmesh)
    col_point=[zeros(5) for i=1:num_p]
    nlsol3=nlsolve((res, u) -> pseudo_arclength!(res, nl_4, u, ip.dv, ip.pu, ds;nmesh=nmesh), ip.u_pred)
    col_point[1]=nlsol3.zero
    dv=nlsol3.zero-ip.pu
    dv=dv/norm(dv)
    pu=nlsol3.zero
    u_pred=dv*ds+pu
    for i=2:num_p
        nlsol3=nlsolve((res, u) -> pseudo_arclength!(res, nl_4, u, dv, pu, ds;nmesh=nmesh), u_pred)
        if nlsol3.residual_norm>1e-8
            while nlsol3.residual_norm>1e-8
                ds=ds/1.2
                nlsol3=nlsolve((res, u) -> pseudo_arclength!(res, nl_4, u, dv, pu, ds;nmesh=nmesh), u_pred)
            end
        end
        col_point[i]=nlsol3.zero
        dv=nlsol3.zero-pu
        dv=dv/norm(dv)
        pu=nlsol3.zero
        u_pred=dv*ds+pu
        if nlsol3.iterations<2
            ds=ds*1.01
        end
    end
    return col_point
end

function continuation_forced2(num_p,sp,ds,dp=0.01;nmesh=20) # Numerical continuation of FRF
    ip=initial_point(sp,ds,dp;nmesh=nmesh)
    col_point=[zeros(5) for i=1:num_p]
    nlsol3=nlsolve((res, u) -> pseudo_arclength!(res, nl_4, u, ip.dv, ip.pu, ds), ip.u_pred)
    col_point[1]=nlsol3.zero
    dv=nlsol3.zero-ip.pu
    dv=dv/norm(dv)
    pu=nlsol3.zero
    u_pred=dv*ds+pu
    for i=2:num_p
        nlsol3=nlsolve((res, u) -> pseudo_arclength!(res, nl_4, u, dv, pu, ds), u_pred)
        col_point[i]=nlsol3.zero
        dv=nlsol3.zero-pu
        dv=dv/norm(dv)
        pu=nlsol3.zero
        u_pred=dv*ds+pu
    end
    return col_point
end

function bd_plot(num_p,sp,ds) # plotting bifurcation diagram 
    ss=continuation_forced(num_p,sp,ds)
    amp=zeros(num_p)
    ff=zeros(num_p)
    for i=1:num_p
        amp[i]=maximum(ss[i][1:end-1])
        ff[i]=ss[i][end]
    end
    plot(ff,amp,legend=:topleft)
end

function pseudo_arclength2!(res,u, dv, pu, ds,F0) # Pseudo arclength continuation of ML model
    res[1]=FRF_zero(u,F0) # Zero problem constructed from a(ρ), b(ρ)
    du = u - pu
    arclength = norm(transpose(dv) * du)
    res[2] = arclength - ds 
end

function continuation_FRF(ip,ds,F0) # Continuation of FRF from ML model
    nlsol3=nlsolve((res, u) -> pseudo_arclength2!(res,u, ip.dv, ip.pu, ds,F0), ip.u_pred)
    FRF_point=[nlsol3.zero]
    dv=nlsol3.zero-ip.pu
    dv=dv/norm(dv)
    pu=nlsol3.zero
    u_pred=dv*ds+pu
    err=0
    while err<1e-8
        nlsol3=nlsolve((res, u) -> pseudo_arclength2!(res,u, dv, pu, ds,F0), u_pred)  
        err=norm(FRF_zero(nlsol3.zero,F0))  
        FRF_point=vcat(FRF_point,[nlsol3.zero])       
        dv=nlsol3.zero-pu
        dv=dv/norm(dv)
        pu=nlsol3.zero
        u_pred=dv*ds+pu
    end
    return FRF_point
end

function FRF_plot(ip,ip2,ds,F0) # FRF plot from the ML model
    ss=continuation_FRF(ip,ds,F0)
    ss2=continuation_FRF(ip2,ds,F0)
    ss2=reverse(ss2)
    ss=vcat(ss,ss2)
    amp=zeros(length(ss))
    ff=zeros(length(ss))
    for i=1:length(ss)
        amp[i]=ss[i][1]
        ff[i]=ss[i][end]
    end
    plot!(ff,amp,legend=:topleft,label="ML model: F0=$F0")
    return(f=ff,a=amp)
end

function initial_FRF(ii,jj,ds,scale) # initial point of the FRF computation (ML model)
    ig=[amp[ii]/scale,ff[ii]]
    pu=ig
    ig2=[amp[jj]/scale,ff[jj]]
    dv=ig2-pu
    dv=dv/norm(dv)
    u_pred=dv*ds+pu
    ip=(dv=dv,u_pred=u_pred,pu=pu)
    return ip
end

##

function pseudo_arclength2!(res,u, dv, pu, ds,F0,FRF_zero) # Pseudo arclength continuation of ML model
    res[1]=FRF_zero(u,F0) # Zero problem constructed from a(ρ), b(ρ)
    du = u - pu
    arclength = norm(transpose(dv) * du)
    res[2] = arclength - ds 
end

function continuation_FRF2(ip,ds,F0,sp,s1,s2,FRF_zero) # Continuation of FRF from ML model
    nlsol3=nlsolve((res, u) -> pseudo_arclength2!(res,u, ip.dv, ip.pu, ds,F0,FRF_zero), ip.u_pred)
    FRF_point=[nlsol3.zero]
    dv=nlsol3.zero-ip.pu
    dv=dv/norm(dv)
    pu=nlsol3.zero
    u_pred=dv*ds+pu
    err=0
    s_p=1
    while s_p<sp
        nlsol3=nlsolve((res, u) -> pseudo_arclength2!(res,u, dv, pu, ds,F0,FRF_zero), u_pred)  
        err=norm(FRF_zero(nlsol3.zero,F0))  
        if err>1e-9
            ds=ds/s1
        end
        if err < 1e-8
            FRF_point=vcat(FRF_point,[nlsol3.zero])       
            dv=nlsol3.zero-pu
            dv=dv/norm(dv)
            pu=nlsol3.zero
            u_pred=dv*ds+pu
            s_p+=1
            if nlsol3.iterations<3
                ds=ds*s2
            end
        end
    end
    return FRF_point
end

function FRF_plot(ip,ds,F0,sp,s1,s2,FRF_zero) # FRF plot from the ML model
    ss=continuation_FRF2(ip,ds,F0,sp,s1,s2,FRF_zero)
    amp_=zeros(length(ss))
    ff=zeros(length(ss))
    for i=1:length(ss)
        amp_[i]=ss[i][1]
        ff[i]=ss[i][end]
    end
    plot(ff,amp_,legend=:topleft,label="ML model: F0=$F0")
    return(f=ff,a=amp_)
end

function FRF_plot!(ip,ds,F0,sp,s1,s2,FRF_zero) # FRF plot from the ML model
    ss=continuation_FRF2(ip,ds,F0,sp,s1,s2,FRF_zero)
    amp_=zeros(length(ss))
    ff=zeros(length(ss))
    for i=1:length(ss)
        amp_[i]=ss[i][1]
        ff[i]=ss[i][end]
    end
    plot!(ff,amp_,legend=:topleft,label="ML model: F0=$F0")
end

function initial_bb(ii,jj,ds,scale) # initial point of the bb-curve computation (ML model)
    ig=[rr[ii]/scale,ff[ii]]
    pu=ig
    ig2=[rr[jj]/scale,ff[jj]]
    dv=ig2-pu
    dv=dv/norm(dv)
    u_pred=dv*ds+pu
    ip=(dv=dv,u_pred=u_pred,pu=pu)
    return ip
end

function cost_fun_b(k,x_b,b_r,l_b) # Cost function to optimise hyper-parameters
    v_b=l_b[1:2]
    σ_b=l_b[3]
    ad_t_b=ARDTransform(v_b)
    xt_b=map(ad_t_b,ColVecs(x_b))
    y_b=b_r
    b_α=GP_α(xt_b, y_b, k,σ_b)
    b_ρ(xt_b,b_α,uu)=GP_pred(map(ad_t_b,ColVecs(reshape(uu,2,1))), b_α.α,xt_b,b_α.k)[1]    
    err=0.   
    for ii=1:length(x_b[1,:])
        uu=x_b[:,ii]
        err+=norm(b_ρ(xt_b,b_α,uu)[1]-b_r[ii])^2/2
    end
    err+=abs(σ_b)/2*transpose(b_α.α)*b_α.K*b_α.α
    return err
end

function cost_fun_a(k,x_a,x_b,b_r,FF,in_data,l_b,l_a) # Cost function to optimise hyper-parameters
    v_a=l_a[1:2]
    σ_a=l_a[3]

    v_b=l_b[1:2]
    σ_b=l_b[3]

    ad_t_a=ARDTransform(v_a)
    ad_t_b=ARDTransform(v_b)
    xt_b=map(ad_t_b,ColVecs(x_b))
    xt_a=map(ad_t_a,ColVecs(x_a))

    Fv=vcat(FF[1]*ones(length(ind[1])),FF[2]*ones(length(ind[2])))
    a_r=zeros(length(b_r))
    y_b=b_r
    b_α=GP_α(xt_b, y_b, k,σ_b)
    b_ρ(xt_b,b_α,uu)=GP_pred(map(ad_t_b,ColVecs(reshape(uu,2,1))), b_α.α,xt_b,b_α.k)[1]
    b_ρ_(uu)=b_ρ(xt_b,b_α,uu)
    
    for j=1:length(Fv)
        a_r[j]=-sqrt(Fv[j]^2-(b_ρ(xt_b,b_α,[amp[j],ff[j]])-ff[j])^2*amp[j]^2)
    end
    y_a=vcat(a_r,zeros(length(a_r)))
    
    a_α=GP_α(xt_a, y_a, k,σ_a)
    a_ρ(xt_a,a_α,uu)=GP_pred(map(ad_t_a,ColVecs(reshape(uu,2,1))), a_α.α,xt_a,a_α.k)[1]
    a_ρ_(uu)=a_ρ(xt_a,a_α,uu)
    FRF_zero(uu,F0)=a_ρ(xt_a,a_α,uu)^2+(b_ρ(xt_b,b_α,uu)-uu[2])^2*uu[1]^2-F0^2
    err=0.
    
    for jj=1:length(FF)
        for ii=1:length(in_data[jj][1,:])
            fr=in_data[jj][2,ii]
            uu2=[in_data[jj][:,ii][1]]
            dd=nlsolve(u->FRF_zero(vcat(u,fr),FF[jj]),uu2)
            err+=norm(dd.zero[1]-uu2[1])
        end
    end
    err+=abs(σ_a)/2*transpose(a_α.α)*a_α.K*a_α.α
    return (err=err,FRF_zero=FRF_zero,b_ρ=b_ρ_,a_ρ=a_ρ_)
end

## Compute FRF to generate data--1
F0=0.01 # Forcing amplitude
num_p=250
sp=0.97
ds=0.25
ss=continuation_forced2(num_p,sp,ds)
ss_=ss
nmesh=20
##
amp_=zeros(num_p)
freq=zeros(num_p)
for i=1:num_p
    amp_[i]=maximum(ss[i][1:end-1])
    freq[i]=ss[i][end]
end
i=[1,10,20,30,40,50,60,70,80,90,100,120,235,240,245,250] #This is the index of training data (F0=0.01)
b_r=zeros(length(i)) # output of function b(ρ)
amp=[maximum(ss[i[ii]][1:end-1]) for ii=1:length(i)]
ff=[ss[i[ii]][end] for ii=1:length(i)]

## Construct output of b(r)
fr=zeros(length(i))
for j=1:length(i)
    ii=i[j]
    fr[j]=freq[ii]
    tt=range(0, 2π/fr[j], length=nmesh+1)[1:end-1]
    res=ss[ii][1:end-1]
    r1=res[1:4:end]
    r2=res[3:4:end]
    force=cos.(fr[j]*tt)

    cs=LS_harmonics(r1, tt,fr[j], 3)
    ψ=atan(cs.coeff[2+3],cs.coeff[2])

#    ψ=atan(imag(f_f[2]),real(f_f[2]))-atan(imag(fr1[2]),real(fr1[2]))
    b_r[j]=fr[j]-cos(ψ)*F0/amp[j]
end

## Compute FRF to generate data--2
F0=0.007
num_p=300
sp=0.95
ds=0.15
ss2=continuation_forced2(num_p,sp,ds)
ss2_=ss2
##
amp2_=zeros(num_p)
freq2=zeros(num_p)
for i=1:num_p
    amp2_[i]=maximum(ss2[i][1:end-1])
    freq2[i]=ss2[i][end]
end
i2=[10,20,30,40,50,60,70,80,90,100,130,245,265,275,285,290] #This is the index of training data 
##
b_r2=zeros(length(i2))
amp2=[maximum(ss2[i2[ii]][1:end-1]) for ii=1:length(i2)]
ff2=[ss2[i2[ii]][end] for ii=1:length(i2)]
## Construct output of a(r) b(r)
fr2=zeros(length(i2))

for j=1:length(i2)
    ii=i2[j]
    fr2[j]=freq2[ii]
    tt=range(0, 2π/fr2[j], length=nmesh+1)[1:end-1]
    res=ss2[ii][1:end-1]
    r1=res[1:4:end]
    r2=res[3:4:end]
    force=cos.(fr[j]*tt)

    cs=LS_harmonics(r1, tt,fr[j], 3)
    ψ=atan(cs.coeff[2+3],cs.coeff[2])

    b_r2[j]=fr2[j]-cos(ψ)*F0/amp2[j]
end
##
σ = 1.0
l1 = 1.0
σ²=σ^2
k=σ²*with_lengthscale(SEKernel(), l1)

amp=vcat(amp,amp2)
ff=vcat(ff,ff2)
fr=vcat(fr,fr2)
b_r=vcat(b_r,b_r2) # output of b

o_b=transpose(hcat(amp,fr))
o_b2=transpose(hcat(zeros(length(fr)),fr)) # Add zeros for a(0,Ω)=0 -> ρ=0 is equilibrium
x_b=o_b # input of b
x_a=hcat(o_b,o_b2) # input of 

in_data=[o_b[:,(jj-1)*16+1:(jj-1)*16+16] for jj=1:2]
FF=[0.01,0.007]
ind=[i,i2]

## Optimise hyperparameters
l_b=[1/10,1/1,1e-8]
cost_hyp_b(l_b)=cost_fun_b(k,x_b,b_r,l_b)

p_m = Optim.optimize(cost_hyp_b, l_b, Optim.Options(; iterations=5))
l_b=p_m.minimizer

l_a=[1/10,1/1,1e-8]
cost_hyp_a(l_a)=cost_fun_a(k,x_a,x_b,b_r,FF,in_data,l_b,l_a).err
p_m = Optim.optimize(cost_hyp_a, l_a, Optim.Options(; iterations=5))
l_a=p_m.minimizer

FRF_zero=cost_fun_a(k,x_a,x_b,b_r,FF,in_data,l_b,l_a).FRF_zero
b_ρ=cost_fun_a(k,x_a,x_b,b_r,FF,in_data,l_b,l_a).b_ρ
a_ρ=cost_fun_a(k,x_a,x_b,b_r,FF,in_data,l_b,l_a).a_ρ


##
ii=1;jj=2;scale=2.5;ds=0.01
ip=initial_FRF(ii,jj,ds,scale)
##
ii=32;jj=31
ip2=initial_FRF(ii,jj,ds,scale)
##
F0=0.01 # Forcing amplitude
num_p=435
sp=0.95
ds=0.15
ss=continuation_forced(num_p,sp,ds)
nmesh=20
##
amp_=zeros(num_p)
freq=zeros(num_p)
for i=1:num_p
    amp_[i]=maximum(ss[i][1:end-1])
    freq[i]=ss[i][end]
end

##
F0=0.01
ds=0.01
ss1=FRF_plot(ip,ip2,ds,F0)
##
F0=0.007
ds=0.01
ss2=FRF_plot(ip,ip2,ds,F0)
##

a=@pgf Axis( {xlabel="Frequency (Hz)",
    ylabel = "Amplitude",
    legend_pos  = "north west",
    height="8cm",
    width="10cm",
    title = "(a)",
    ymin=0,ymax=6.0,xmin=0.99,xmax=1.01,
    mark_options = {scale=1.0},
    xtick=[0.99,1.0,1.01]
},
Plot(
    { color="red",
        no_marks,
    },
    Coordinates(freq[1:375],amp_[1:375])
),
 #  LegendEntry("Ground truth model"),
   Plot(
        { color="blue",
            no_marks
        },
        Coordinates(ss2.f,ss2.a)
    ),
#    LegendEntry("Learnt model"),
    Plot(
    { color="red",
        only_marks,
    },
    Coordinates(ff,amp)
),

#LegendEntry("Training data sets"),

    Plot(
        { color="red",
            no_marks,
        },
        Coordinates(freq2,amp2_)
    ),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(ss1.f,ss1.a)
    ),
)
@pgf a["every axis title/.style"] = "below right,at={(0,1)}";
pgfsave("./Figures/trained_FRC_num.pdf",a)
##
scale=2.0
ii=1;jj=2
ip=initial_FRF(ii,jj,ds,scale)
ii=32;jj=31
ip2=initial_FRF(ii,jj,ds,scale)
F0=0.003
ds=0.002
ss1=FRF_plot(ip,ip2,ds,F0)
plot(ss1.f,ss1.a)
num_p=290
sp=0.96
ds=0.02
ss=continuation_forced(num_p,sp,ds,F0;nmesh=20)
a1=zeros(num_p)
f1=zeros(num_p)
for i=1:num_p
    a1[i]=maximum(ss[i][1:end-1])
    f1[i]=ss[i][end]
end
##
scale=1.0
ii=1;jj=2
ip=initial_FRF(ii,jj,ds,scale)
ii=32;jj=31
ip2=initial_FRF(ii,jj,ds,scale)
F0=0.006
ds=0.002
ss2=FRF_plot(ip,ip2,ds,F0)
plot!(ss2.f,ss2.a)
num_p=500
sp=0.96
ds=0.04
ss=continuation_forced(num_p,sp,ds,F0;nmesh=40)
a2=zeros(num_p)
f2=zeros(num_p)
for i=1:num_p
    a2[i]=maximum(ss[i][1:end-1])
    f2[i]=ss[i][end]
end
f2_=f2
a2_=a2
f2=f2_[1:385];a2=a2_[1:385];
##
scale=1.
ii=1;jj=2
ip=initial_FRF(ii,jj,ds,scale)
ii=32;jj=31
ip2=initial_FRF(ii,jj,ds,scale)
F0=0.009
ds=0.002
ss3=FRF_plot(ip,ip2,ds,F0)
num_p=475
sp=0.96
ds=0.04
ss=continuation_forced(num_p,sp,ds,F0;nmesh=40)
a3=zeros(num_p)
f3=zeros(num_p)
for i=1:num_p
    a3[i]=maximum(ss[i][1:end-1])
    f3[i]=ss[i][end]
end
f3_=f3
a3_=a3
f3=f3_[1:475];a3=a3_[1:475];
##
scale=1.0
ii=1;jj=2
ip=initial_FRF(ii,jj,ds,scale)
ii=32;jj=31
ip2=initial_FRF(ii,jj,ds,scale)
F0=0.012
ds=0.001
ss4=FRF_plot(ip,ip2,ds,F0)

num_p=545
sp=0.95
ds=0.05
ss=continuation_forced(num_p,sp,ds,F0;nmesh=50)
a4=zeros(num_p)
f4=zeros(num_p)
for i=1:num_p
    a4[i]=maximum(ss[i][1:end-1])
    f4[i]=ss[i][end]
end
f4_=f4
a4_=a4
f4=f4_[1:545];a4=a4_[1:545];

## Backbone curve computation

b_zero(uu)=b_ρ(uu)[1]-uu[2]

rr=0.001:0.01:9.
f_max=zeros(length(rr))
uu2=1.000001
bb=nlsolve(u->b_zero(vcat(rr[1],u)),[uu2])
f_max[1]=bb.zero[1]

for j=2:length(rr)
    uu2=f_max[j-1]
    bb=nlsolve(u->b_zero(vcat(rr[j],u)),[uu2])
    f_max[j]=bb.zero[1]
end


b=@pgf Axis( {xlabel="Frequency (Hz)",
#            ylabel = "Amplitude",
            legend_pos  = "north west",
            height="8cm",
            width="10cm",
            title = "(b)",
            ymin=0,ymax=6.0,xmin=0.99,xmax=1.01,
            mark_options = {scale=1.0},
            xtick=[0.99,1.0,1.01]
},
Plot(
    { color="red",
        no_marks,
    },
    Coordinates(f1,a1)
),
#LegendEntry("Ground truth model"),
Plot(
    { color="blue",
        no_marks,
    },
    Coordinates(ss1.f,ss1.a)
),
#LegendEntry("Learnt model"),

    Plot(
        { color="red",
            no_marks,
        },
        Coordinates(f2,a2)
    ),
Plot(
    { color="red",
        no_marks,
    },
    Coordinates(f3,a3)
),
Plot(
    { color="red",
    no_marks,
    },
    Coordinates(f4,a4)
),

    Plot(
        { color="blue",
            no_marks
        },
        Coordinates(ss2.f,ss2.a)
    ),
    Plot(
        { color="blue",
            no_marks
        },
        Coordinates(ss3.f,ss3.a)
    ),
    Plot(
        { color="blue",
            no_marks
        },
        Coordinates(ss4.f,ss4.a)
    ),
    Plot(
        { color="black",
            no_marks,
            style ="{dashed}",
        },
        Coordinates(f_max,rr)
    ),
)
@pgf b["every axis title/.style"] = "below right,at={(0,1)}";
pgfsave("./Figures/untrained_FRC_num.pdf",b)

gp=@pgf GroupPlot(
    { group_style = { group_size="2 by 1" },
      no_markers,
    },
    a,b)

pgfsave("./Figures/num_FRF.pdf",gp)
