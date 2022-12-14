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
using ForwardDiff
using MAT

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

function vvcat(amp) # vcat vector of vectors
    aamp=amp[1]
    for ii=2:length(amp)
        aamp=vcat(aamp,amp[ii])
    end
    return aamp
end

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

function initial_FRF(ii,jj,ind,ds,scale) # initial point of the FRF computation (ML model)
    ig=[amp[ind][ii]/scale,2π*freq[ind][ii]/ω_scale]
    pu=ig
    ig2=[amp[ind][jj]/scale,2π*freq[ind][jj]/ω_scale]
    dv=ig2-pu
    dv=dv/norm(dv)
    u_pred=dv*ds+pu
    ip=(dv=dv,u_pred=u_pred,pu=pu)
    return ip
end

function initial_SC(ii,jj,ind,ds,scale) # initial point of the FRF computation (ML model)
    ig=[amp[ind][ii]/scale,amp_f[ind][ii]]
    pu=ig
    ig2=[amp[ind][jj]/scale,amp_f[ind][jj]]
    dv=ig2-pu
    dv=dv/norm(dv)
    u_pred=dv*ds+pu
    ip=(dv=dv,u_pred=u_pred,pu=pu)
    return ip
end

function initial_SC2(ii,jj,ind,ds,scale) # initial point of the FRF computation (ML model)
    j=ind
    vars = matread("./measured_data/$j.mat")
    aa=vars["exp"]
    data=aa["data"]
    amp=vec(data["x_amp"])
    amp_f=vec(data["base_amp"])
    freq=data["forcing_freq"]
    
    ig=[amp[ii]/scale,amp_f[ii]]
    pu=ig
    ig2=[amp[jj]/scale,amp_f[jj]]
    dv=ig2-pu
    dv=dv/norm(dv)
    u_pred=dv*ds+pu
    ip=(dv=dv,u_pred=u_pred,pu=pu,fr=freq,f=amp_f,h=amp)
    return ip
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

## Read measured results 

# sample frequency 5000 Hz
ind_f_l=70
skip=3
skip2=3
ind_f=1:skip:ind_f_l
f_l=length(ind_f)
start_point=5
x_amp=Vector{Any}(undef, f_l)
base_amp=Vector{Any}(undef, f_l)
x_dis=Vector{Any}(undef, f_l)
base_dis=Vector{Any}(undef, f_l)
freq=Vector{Any}(undef, f_l)
b_l=Vector{Int}(undef, f_l)

for i=1:f_l
    j=ind_f[i]
    vars = matread("./measured_data/$j.mat")
    aa=vars["exp"]
    data=aa["data"]
    x_amp[i]=vec(data["x_amp"])
    base_amp[i]=vec(data["base_amp"])
    x_dis[i]=data["x"][start_point:skip2:end]
    base_dis[i]=data["base_disp"][start_point:skip2:end]
    freq[i]=data["forcing_freq"]
    b_l[i]=length(x_dis[i])
end

ψ=[zeros(b_l[ii]) for ii=1:f_l]
ψ0=[zeros(b_l[ii]) for ii=1:f_l]
ph=[zeros(b_l[ii]) for ii=1:f_l]
ph2=[zeros(b_l[ii]) for ii=1:f_l]
amp=[zeros(b_l[ii]) for ii=1:f_l]
amp_f=[zeros(b_l[ii]) for ii=1:f_l]
cfr=[zeros(b_l[ii],500) for ii=1:f_l]
chr=[zeros(b_l[ii],500) for ii=1:f_l]
## Compute phase_lag from the measured results
t=0:0.0002:0.1-0.0002
for i=1:f_l
    for jj=1:b_l[i]    
        f=freq[i][jj]
        Om=2π*f
        ff=vec(base_dis[i][jj])[1:500]
        hh=vec(x_dis[i][jj])[1:500]
        N=1
        cf=LS_harmonics(ff, t, Om, N)
        ch=LS_harmonics(hh, t, Om, N)
        arh=sqrt(ch.coeff[2]^2+ch.coeff[2+N]^2)
        arf=sqrt(cf.coeff[2]^2+cf.coeff[2+N]^2)
        amp[i][jj]=arh
        amp_f[i][jj]=arf
        cfr[i][jj,:]=cf.rr-ones(length(cf.rr))*cf.coeff[1]
        chr[i][jj,:]=ch.rr-ones(length(ch.rr))*ch.coeff[1]
        ph[i][jj]=atan(ch.coeff[2+N],ch.coeff[2])
        ph2[i][jj]=atan(cf.coeff[2+N],cf.coeff[2])
        ψ[i][jj]=atan(ch.coeff[2+N],ch.coeff[2])-atan(cf.coeff[2+N],cf.coeff[2])
    end
end

jj=10
plot(amp_f[jj],amp[jj])
plot(chr[jj][1,:])

## Choose training data sets
train_l=sum(b_l) # Number of training points
b_r=zeros(train_l)
x_b=zeros(2,train_l)
amp_=zeros(train_l)
fr=zeros(train_l)
Fv=zeros(train_l)
ph_lag=zeros(train_l)
h_train=zeros(train_l,500)
ph_train=zeros(train_l)
ω_scale=22*2π

j=1
sF=1
for i=1:length(b_l)
    for jj=1:b_l[i]   
        b_r[j]=2π*freq[i][jj]/ω_scale-cos(ψ[i][jj])*sqrt(sF)*amp_f[i][jj]/amp[i][jj]
        ph_lag[j]=cos(ψ[i][jj])
        amp_[j]=amp[i][jj]
        fr[j]=2π*freq[i][jj]/ω_scale
        x_b[:,j]=[amp[i][jj],2π*freq[i][jj]/ω_scale]
        Fv[j]=amp_f[i][jj]
        h_train[j,:]=vec(base_dis[i][jj])[1:500]
        ph_train[j]=ph[i][jj]
        j+=1
    end
end


function cost_fun_b(k,x_b,b_r,l_b) # Cost function to optimise hyper-parameters of b(ρ,ω)
    v_b=l_b[1:2]
    σ_b=l_b[3]
    ad_t_b=ARDTransform(v_b)
    xt_b=map(ad_t_b,x_b)
    y_b=b_r
    b_α=GP_α(xt_b, y_b, k,σ_b)
    b_ρ(xt_b,b_α,uu)=GP_pred(map(ad_t_b,reshape(uu,2,1)), b_α.α,xt_b,b_α.k)[1]    
    err=0.   
    for ii=1:length(x_b[1,:])
        uu=x_b[:,ii]
        err+=norm(b_ρ(xt_b,b_α,uu)[1]-b_r[ii])^2/2
    end
    err+=abs(σ_b)/2*transpose(b_α.α)*b_α.K*b_α.α
    return err
end

function cost_fun_a(k_a,k_b,x_a,x_b,b_r,Fv,in_data,l_b,l_a) # Cost function to optimise hyper-parameters of a(ρ,ω)
    v_a=l_a[1:2]
    σ_a=l_a[3]

    v_b=l_b[1:2]
    σ_b=l_b[3]

    ad_t_a=ARDTransform(v_a)
    ad_t_b=ARDTransform(v_b)
    xt_b=map(ad_t_b,ColVecs(x_b))
    xt_a=map(ad_t_a,ColVecs(x_a))

    a_r=zeros(length(b_r))
    b_α=GP_α(xt_b, b_r, k_b,σ_b)
    b_ρ(xt_b,b_α,uu)=GP_pred(map(ad_t_b,ColVecs(reshape(uu,2,1))), b_α.α,xt_b,b_α.k)[1]
    b_ρ_(uu)=b_ρ(xt_b,b_α,uu)
    for j=1:length(Fv)
        a_r[j]=-sqrt(sF*Fv[j]^2*(1-ph_lag[j]^2))
    end
    y_a=vcat(a_r,zeros(length(a_r)+1))
    
    a_α=GP_α(xt_a, y_a, k_a,σ_a)
    a_ρ(xt_a,a_α,uu)=GP_pred(map(ad_t_a,ColVecs(reshape(uu,2,1))), a_α.α,xt_a,a_α.k)[1]
    a_ρ_(uu)=a_ρ(xt_a,a_α,uu)
    FRF_zero(uu,F0)=a_ρ_(uu)^2+(b_ρ_(uu)-uu[2])^2*uu[1]^2-sF*F0^2
    Sc_zero(uu,F0)=a_ρ_(vcat(uu[1],F0))^2+(b_ρ_(vcat(uu[1],F0))-F0)^2*uu[1]^2-sF*uu[2]^2
    err=0.

    for jj=1:length(Fv)
            uu=in_data[:,jj]
            err+=abs(a_ρ_(uu)^2+(b_ρ_(uu)-uu[2])^2*uu[1]^2-sF*Fv[jj]^2)
    end

    err2=err
    err+=abs(σ_a)/2*transpose(a_α.α)*a_α.K*a_α.α
    return (err=err,FRF_zero=FRF_zero,b_ρ=b_ρ_,a_ρ=a_ρ_,Sc=Sc_zero,err2=err2)
end

## Define kernels for kernel ridge regression
σ = 1.0
l1 = 1.0
σ²=σ^2
k_a=σ²*with_lengthscale(SqExponentialKernel(), l1)
k_b=σ²*with_lengthscale(SqExponentialKernel(), l1)
o_b2=transpose(hcat(zeros(length(x_b[1,:])),fr)) # Add zeros for a(0,Ω)=0 -> ρ=0 is equilibrium
o_b2=hcat(o_b2,zeros(2))
x_a=hcat(x_b,o_b2) # input of 

## Optimise hyperparameters
l_b=[1.0,1.0,1e-7]
cost_hyp_b(l_b)=cost_fun_b(k_b,x_b,b_r,l_b)
ForwardDiff.gradient(cost_hyp_b,l_b)

p_m = Optim.optimize(cost_hyp_b, l_b, LBFGS(),Optim.Options(iterations = 10); autodiff = :forward)
l_b=p_m.minimizer

l_a=[1.0,1.0,1e-7]
cost_hyp_a(l_a)=cost_fun_a(k_a,k_b,x_a,x_b,b_r,Fv,x_b,l_b,l_a).err
p_m = Optim.optimize(cost_hyp_a, l_a,LBFGS(),Optim.Options(iterations = 10); autodiff = :forward)
l_a=p_m.minimizer
##
FRF_zero=cost_fun_a(k_a,k_b,x_a,x_b,b_r,Fv,x_b,l_b,l_a).FRF_zero
Sc_zero=cost_fun_a(k_a,k_b,x_a,x_b,b_r,Fv,x_b,l_b,l_a).Sc
b_ρ=cost_fun_a(k_a,k_b,x_a,x_b,b_r,Fv,x_b,l_b,l_a).b_ρ
a_ρ=cost_fun_a(k_a,k_b,x_a,x_b,b_r,Fv,x_b,l_b,l_a).a_ρ
## Plot trained FRFs
#b_l[ind]

ss=Vector{Any}(undef, 24)

ind1=24;ii=1;jj=2;sp=270;ds=0.01;s1=1.01;s2=1.02;scale=1.5
ip1=initial_SC(ii,jj,ind1,ds,scale)
F0=freq[ind1][1]*2π/ω_scale
ss[1]=FRF_plot(ip1,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[1].a))*freq[ind1][1]
plot(ss[1].f,ss[1].a)
scatter!(amp_f[ind1],amp[ind1])

ind1=23;ii=1;jj=2;sp=200;ds=0.01;s1=1.01;s2=1.02;scale=1.5
ip1=initial_SC(ii,jj,ind1,ds,scale)
F0=freq[ind1][1]*2π/ω_scale
ss[2]=FRF_plot(ip1,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[2].a))*freq[ind1][1]
plot(ss[2].f,ss[2].a)
scatter!(amp_f[ind1],amp[ind1])

ind1=22;ii=1;jj=2;sp=250;ds=0.01;s1=1.01;s2=1.02;scale=1.5
ip1=initial_SC(ii,jj,ind1,ds,scale)
F0=freq[ind1][1]*2π/ω_scale
ss[3]=FRF_plot(ip1,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[3].a))*freq[ind1][1]
plot(ss[3].f,ss[3].a)
scatter!(amp_f[ind1],amp[ind1])

ind1=21;ii=1;jj=2;sp=200;ds=0.012;s1=1.01;s2=1.02;scale=1.5
ip1=initial_SC(ii,jj,ind1,ds,scale)
F0=freq[ind1][1]*2π/ω_scale
ss[4]=FRF_plot(ip1,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[4].a))*freq[ind1][1]
plot(ss[4].f,ss[4].a)
scatter!(amp_f[ind1],amp[ind1])

ind2=20;ii=1;jj=2;sp=250;ds=0.01;s1=1.01;s2=1.02;scale=1.5
ip2=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind2][1]*2π/ω_scale
ss[5]=FRF_plot(ip2,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[5].a))*freq[ind2][1]
plot(ss[5].f,ss[5].a)
scatter!(amp_f[ind2],amp[ind2])

ind2=19;ii=1;jj=2;sp=210;ds=0.011;s1=1.01;s2=1.02;scale=1.5
ip2=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind2][1]*2π/ω_scale
ss[6]=FRF_plot(ip2,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[6].a))*freq[ind2][1]
plot(ss[6].f,ss[6].a)
scatter!(amp_f[ind2],amp[ind2])

ind2=18;ii=1;jj=2;sp=160;ds=0.012;s1=1.01;s2=1.021;scale=1.5
ip2=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind2][1]*2π/ω_scale
ss[7]=FRF_plot(ip2,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[7].a))*freq[ind2][1]
plot(ss[7].f,ss[7].a)
scatter!(amp_f[ind2],amp[ind2])

ind2=17;ii=1;jj=2;sp=210;ds=0.01;s1=1.01;s2=1.02;scale=1.5
ip2=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind2][1]*2π/ω_scale
ss[8]=FRF_plot(ip2,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[8].a))*freq[ind2][1]
plot(ss[8].f,ss[8].a)
scatter!(amp_f[ind2],amp[ind2])

ind2=16;ii=1;jj=2;sp=60;ds=0.01;s1=1.01;s2=1.05;scale=1.5
ip2=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind2][1]*2π/ω_scale
ss[9]=FRF_plot(ip2,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[9].a))*freq[ind1][1]
plot(ss[9].f,ss[9].a)
scatter!(amp_f[ind2],amp[ind2])

ind2=15;ii=1;jj=2;sp=50;ds=0.01;s1=1.01;s2=1.05;scale=1.5
ip2=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind2][1]*2π/ω_scale
ss[10]=FRF_plot(ip2,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[10].a))*freq[ind1][1]
plot(ss[10].f,ss[10].a)
scatter!(amp_f[ind2],amp[ind2])

ind2=14;ii=1;jj=2;sp=50;ds=0.01;s1=1.01;s2=1.05;scale=1.5
ip2=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind2][1]*2π/ω_scale
ss[11]=FRF_plot(ip2,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[11].a))*freq[ind1][1]
plot(ss[11].f,ss[11].a)
scatter!(amp_f[ind2],amp[ind2])

ind2=13;ii=1;jj=2;sp=40;ds=0.01;s1=1.01;s2=1.07;scale=1.5
ip2=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind2][1]*2π/ω_scale
ss[12]=FRF_plot(ip2,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[12].a))*freq[ind1][1]
plot(ss[12].f,ss[12].a)
scatter!(amp_f[ind2],amp[ind2])

ind2=12;ii=1;jj=2;sp=210;ds=0.005;s1=1.01;s2=1.025;scale=1.5
ip2=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind2][1]*2π/ω_scale
ss[13]=FRF_plot(ip2,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[13].a))*freq[ind1][1]
plot(ss[13].f,ss[13].a)
scatter!(amp_f[ind2],amp[ind2])

ind2=11;ii=33;jj=32;sp=180;ds=0.005;s1=1.01;s2=1.03;scale=0.95
ip2=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind2][1]*2π/ω_scale
ss[14]=FRF_plot(ip2,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[14].a))*freq[ind1][1]
plot(ss[14].f,ss[14].a)
scatter!(amp_f[ind2],amp[ind2])

ind3=10;ii=1;jj=2;sp=250;ds=0.002;s1=1.01;s2=1.023;scale=1.5
ip3=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind3][1]*2π/ω_scale
ss[15]=FRF_plot(ip3,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[15].a))*freq[ind3][1]
plot(ss[15].f,ss[15].a)
scatter!(amp_f[ind3],amp[ind3])

ind3=9;ii=1;jj=2;sp=210;ds=0.01;s1=1.03;s2=1.03;scale=1.5
ip3=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind3][1]*2π/ω_scale
ss[16]=FRF_plot(ip3,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[16].a))*freq[ind3][1]
plot(ss[16].f,ss[16].a)
scatter!(amp_f[ind3],amp[ind3])

ind3=8;ii=1;jj=2;sp=160;ds=0.012;s1=1.03;s2=1.03;scale=1.2
ip3=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind3][1]*2π/ω_scale
ss[17]=FRF_plot(ip3,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[17].a))*freq[ind3][1]
plot(ss[17].f,ss[17].a)
scatter!(amp_f[ind3],amp[ind3])

ind3=7;ii=1;jj=2;sp=150;ds=0.01;s1=1.03;s2=1.03;scale=1.5
ip3=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind3][1]*2π/ω_scale
ss[18]=FRF_plot(ip3,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[18].a))*freq[ind3][1]
plot(ss[18].f,ss[18].a)
scatter!(amp_f[ind3],amp[ind3])

ind3=6;ii=1;jj=2;sp=220;ds=0.006;s1=1.01;s2=1.01;scale=1.5
ip3=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind3][1]*2π/ω_scale
ss[19]=FRF_plot(ip3,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[19].a))*freq[ind3][1]
plot(ss[19].f,ss[19].a)
scatter!(amp_f[ind3],amp[ind3])

ind3=5;ii=1;jj=2;sp=400;ds=0.002;s1=1.01;s2=1.01;scale=1.2
ip3=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind3][1]*2π/ω_scale
ss[20]=FRF_plot(ip3,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[20].a))*freq[ind3][1]
plot(ss[20].f,ss[20].a)
scatter!(amp_f[ind3],amp[ind3])

ind3=4;ii=1;jj=2;sp=165;ds=0.002;s1=1.01;s2=1.02;scale=1.5
ip3=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind3][1]*2π/ω_scale
ss[21]=FRF_plot(ip3,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[21].a))*freq[ind3][1]
plot(ss[21].f,ss[21].a)
scatter!(amp_f[ind3],amp[ind3])

ind3=3;ii=1;jj=2;sp=130;ds=0.002;s1=1.01;s2=1.03;scale=1.5
ip3=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind3][1]*2π/ω_scale
ss[22]=FRF_plot(ip3,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[22].a))*freq[ind3][1]
plot(ss[22].f,ss[22].a)
scatter!(amp_f[ind3],amp[ind3])

ind3=2;ii=1;jj=2;sp=95;ds=0.0035;s1=1.01;s2=1.035;scale=1.5
ip3=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind3][1]*2π/ω_scale
ss[23]=FRF_plot(ip3,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[23].a))*freq[ind3][1]
plot(ss[23].f,ss[23].a)
scatter!(amp_f[ind3],amp[ind3])

ind3=1;ii=1;jj=2;sp=40;ds=0.004;s1=1.02;s2=1.12;scale=1.2
ip3=initial_SC(ii,jj,ind2,ds,scale)
F0=freq[ind3][1]*2π/ω_scale
ss[24]=FRF_plot(ip3,ds,F0,sp,s1,s2,Sc_zero)
ff_=ones(length(ss[24].a))*freq[ind3][1]
plot(ss[24].f,ss[24].a)
scatter!(amp_f[ind3],amp[ind3])

##
ind=65;ii=1;jj=2;sp=480;ds=0.0015;s1=1.01;s2=1.02;scale=1.5
ip4=initial_SC2(ii,jj,ind,ds,scale)
F0=ip4.fr[1]*2π/ω_scale
ss4=FRF_plot(ip4,ds,F0,sp,s1,s2,Sc_zero)
scatter!(ip4.f,ip4.h)

ind=35;ii=1;jj=2;sp=400;ds=0.001;s1=1.01;s2=1.02;scale=1.5
ip5=initial_SC2(ii,jj,ind,ds,scale)
F0=ip5.fr[1]*2π/ω_scale
ss5=FRF_plot(ip5,ds,F0,sp,s1,s2,Sc_zero)
scatter!(ip5.f,ip5.h)

ind=12;ii=1;jj=2;sp=225;ds=0.001;s1=1.01;s2=1.02;scale=1.5
ip6=initial_SC2(ii,jj,ind,ds,scale)
F0=ip6.fr[1]*2π/ω_scale
ss6=FRF_plot(ip6,ds,F0,sp,s1,s2,Sc_zero)
scatter!(ip6.f,ip6.h)

##

a=@pgf Axis( {xlabel="Frequency (Hz)",
            ylabel = "Amplitude (V)",
            legend_pos  = "north west",
            height="11cm",
            width="15cm",
            ymin=0.00,ymax=2.5,xmin=16,xmax=26,
            mark_options = {scale=0.3}
},
Plot(
    { color="red",
        only_marks,
    },
    Coordinates(ff*ω_scale/2π,rr)
),
LegendEntry("Measured backbone"),
    Plot(
        { color="blue",
            no_marks,
        },
        Coordinates(bb1.f*ω_scale/2π,bb1.a)
    ),
    LegendEntry("Computed backbone (ML)")
)

pgfsave("./Figures/bb_curve.pdf",a)
##