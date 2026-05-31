#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click manuscript figure generation for the revised quantum Doppler paper.
Only NumPy/SciPy/Matplotlib are required. QuTiP is optional via the original
QZZB module; if unavailable, dense SciPy fidelity is used.
"""
import os, math, importlib.util, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent

# Import original QZZB numerical module under a safe name.
spec = importlib.util.spec_from_file_location('qzzbmod', ROOT/'quantum_doppler_qzzb_package_original.py')
qzzbmod = importlib.util.module_from_spec(spec)
sys.modules['qzzbmod'] = qzzbmod
spec.loader.exec_module(qzzbmod)

# ----------------------------- common helpers -----------------------------

def GQ(eta, NS):
    eta=np.asarray(eta,dtype=float)
    return (NS+1.0)/(1.0+2.0*(1.0-eta)*NS)

def Geff(eta, NS, Gamma, a=1.0):
    return GQ(eta,NS)*np.exp(-a*np.asarray(Gamma,dtype=float))

def gamma_boundary(eta, NS, a=1.0):
    g=GQ(eta,NS)
    return np.where(g>1, np.log(g)/a, np.nan)

def doppler_k(lambda_0=532e-9, n_g=1.000444):
    return 4*np.pi*n_g/lambda_0

def phase_wrapping_probability(sigma_phi, threshold=np.pi):
    sigma=np.maximum(np.asarray(sigma_phi,dtype=float), 1e-300)
    return 2*norm.sf(threshold/sigma)

# ----------------------------- Fig 1 -----------------------------

def solve_pr_eos_density(P_MPa, T_K):
    Tc=190.56; Pc=4.5992e6; omega=0.011; R=const.R
    P=P_MPa*1e6
    kappa=0.37464 + 1.54226*omega - 0.26992*omega**2
    alpha=(1+kappa*(1-np.sqrt(T_K/Tc)))**2
    a=0.45724*(R*Tc)**2/Pc*alpha
    b=0.07780*R*Tc/Pc
    A=a*P/(R*T_K)**2; B=b*P/(R*T_K)
    coeff=[1.0, -(1.0-B), A-3*B**2-2*B, -(A*B-B**2-B**3)]
    roots=np.roots(coeff)
    real=np.real(roots[np.abs(np.imag(roots))<1e-8])
    pos=real[real>0]
    Z=np.max(pos) if len(pos) else 1.0
    return const.Avogadro/(Z*R*T_K/P)

def rayleigh_sigma(lambda_nm):
    lam=lambda_nm*1e-9; n0=1.000444; rho0=2.68678e25; Fk=1.04
    return (24*np.pi**3)/(rho0**2*lam**4)*((n0**2-1)/(n0**2+2))**2*Fk

def return_photons(P_scan, wl, T, L, Omega_frac, eta_sys, E_pulse):
    Ntx=E_pulse/(const.h*const.c/(wl*1e-9)); sig=rayleigh_sigma(wl)
    return np.array([Ntx*sig*solve_pr_eos_density(P,T)*L*Omega_frac*eta_sys for P in P_scan])

def add_regime(ax):
    ax.axhspan(1e4, 1e8, alpha=0.15, label=r"Photon-rich ($N_{ret}>10^4$)")
    ax.axhspan(1e2, 1e4, alpha=0.15, label=r"Low-return ($10^2<N_{ret}\leq10^4$)")
    ax.axhspan(1, 1e2, alpha=0.22, label=r"Photon-starved ($1<N_{ret}\leq100$)")
    ax.axhspan(1e-4, 1, alpha=0.14, label=r"Extreme photon-starved ($N_{ret}\leq1$)")

def fig1():
    P=np.linspace(1,35,200); wls=[532,633,1064,1550]
    cases=[('Optimistic high-return case',1e-3,1e-6,0.10,(1e-2,1e6)),('Constrained photon-starved case',1e-6,1e-7,0.05,(1e-4,1e4))]
    fig,axs=plt.subplots(1,2,figsize=(15,6),sharex=True)
    for ax,(title,E,Omega,eta,ylim) in zip(axs,cases):
        add_regime(ax)
        for wl in wls:
            ax.plot(P, return_photons(P,wl,298.15,0.01,Omega,eta,E), lw=2.4, label=fr'$\lambda={wl}$ nm')
        ax.set_yscale('log'); ax.set_xlim(1,35); ax.set_ylim(*ylim)
        ax.set_xlabel('Pipeline pressure $P$ (MPa)'); ax.set_title(title, fontweight='bold')
        ax.grid(True, which='both', ls='--', alpha=0.35)
    axs[0].set_ylabel(r'Expected return photons $N_{ret}$ per pulse')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.18,0.5), fontsize=9)
    fig.suptitle('Photon budget vs pressure and wavelength under two optical constraints', fontweight='bold')
    fig.tight_layout()
    fig.savefig(ROOT/'fig1_photon_budget.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# ----------------------------- Fig 2 -----------------------------
def fig2():
    NS=np.logspace(0,3,300)
    scenarios=[(1,0.10,1e-6,'$M=1,\ \eta=0.10,\ \tau=1\,\mu s$','-'),(10,0.10,1e-6,'$M=10,\ \eta=0.10,\ \tau=1\,\mu s$','--'),(100,0.30,1e-6,'$M=100,\ \eta=0.30,\ \tau=1\,\mu s$','-.'),(1000,0.50,10e-6,'$M=1000,\ \eta=0.50,\ \tau=10\,\mu s$',':')]
    fig,(a,b)=plt.subplots(1,2,figsize=(14.5,6))
    for M,eta,tau,label,ls in scenarios:
        sig=1/np.sqrt(4*M*eta*NS); sv=sig/(tau*doppler_k())
        a.plot(NS,sig,ls=ls,lw=2.4,label=label); b.plot(NS,sv,ls=ls,lw=2.4,label=label)
    for ax in (a,b):
        ax.axvspan(1,100,alpha=0.15,label=r'Low-photon signal regime ($1<N_S\leq100$)')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, which='both', ls='--', alpha=0.4); ax.legend(fontsize=8)
    a.axhline(np.pi,ls='-',lw=1.4,alpha=0.55,label=r'Phase wrapping scale ($\pi$)')
    a.set_xlabel(r'Transmitted signal photons $N_S$'); a.set_ylabel(r'Phase standard deviation $\sigma_{\hat\phi}$ (rad)')
    b.set_xlabel(r'Transmitted signal photons $N_S$'); b.set_ylabel(r'Velocity standard deviation $u(v)$ (m/s)')
    a.set_title('Classical coherent-state SQL phase error', fontweight='bold'); b.set_title('Velocity error propagated from phase SQL',fontweight='bold')
    fig.suptitle('Classical SQL error propagation in the low-photon signal regime', fontweight='bold')
    fig.tight_layout(); fig.savefig(ROOT/'fig2_classical_sql.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# ----------------------------- Fig 3 -----------------------------
def fig3():
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    eta=np.linspace(0.01,1,600); NS_list=[10,30,100]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15.5,6.5))
    for NS in NS_list:
        ax1.plot(eta,GQ(eta,NS),lw=2.5,label=fr'$N_S={NS}$')
    ax1.axhline(1,ls='--',lw=2,label=r'$G_Q=1$ equal-signal')
    ax1.axvline(0.5,ls=':',lw=1.8,label=r'equal-signal threshold $\eta=0.5$')
    ax1.axvline(0.75,ls='-.',lw=1.8,label=r'equal-total large-$N_S$ threshold $\eta\simeq0.75$')
    ax1.axvspan(0.5,1,alpha=0.10); ax1.axvspan(0.01,0.5,alpha=0.08)
    ax1.set(xlim=(0.01,1),ylim=(0,6),xlabel=r'System transmittance $\eta$',ylabel=r'Advantage ratio')
    ax1.set_title(r'Zoomed pure-loss advantage near $G_Q=1$', fontweight='bold')
    ax1.grid(True,ls='--',alpha=0.4); ax1.legend(fontsize=8,loc='upper left')
    ins=inset_axes(ax1,width='36%',height='36%',loc='upper right',borderpad=1.3)
    for NS in NS_list: ins.plot(eta,GQ(eta,NS),lw=1.3)
    ins.axhline(1,ls='--',lw=1); ins.axvline(0.5,ls=':',lw=1); ins.axvline(0.75,ls='-.',lw=1)
    ins.set_xlim(0.01,1); ins.set_ylim(0,105); ins.set_title('full range',fontsize=8); ins.tick_params(labelsize=7); ins.grid(True,ls='--',alpha=0.25)
    eg,ng=np.meshgrid(np.linspace(0.05,1,300),np.linspace(1,150,300)); grid=GQ(eg,ng)
    im=ax2.pcolormesh(eg,ng,grid,shading='auto',vmin=0,vmax=6)
    c=fig.colorbar(im,ax=ax2); c.set_label(r'$G_Q$ (values $>6$ clipped)')
    cs=ax2.contour(eg,ng,grid,levels=[1],linewidths=2.2); ax2.clabel(cs,fmt={1:'$G_Q=1$'},fontsize=9)
    ax2.axvline(0.5,ls=':',lw=1.8); ax2.axvline(0.75,ls='-.',lw=1.8)
    ax2.set(xlim=(0.05,1),ylim=(1,150),xlabel=r'System transmittance $\eta$',ylabel=r'Signal-mode photons $N_S$')
    ax2.set_title(r'Topology of $G_Q(\eta,N_S)$',fontweight='bold')
    fig.suptitle('Pure-loss TMSV advantage window', fontweight='bold')
    fig.tight_layout(); fig.savefig(ROOT/'fig3_pure_loss_tmsv.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# ----------------------------- Fig 4 -----------------------------
def fig4():
    eta=np.linspace(0.30,1,360); gam=np.linspace(0,5,330); E,G=np.meshgrid(eta,gam)
    NS=100; ge=Geff(E,NS,G)
    var_cs=1/(4*1*E*NS); var_t=var_cs/np.maximum(ge,1e-300); pwrap=phase_wrapping_probability(np.sqrt(var_t))
    fig,axs=plt.subplots(1,3,figsize=(18,5.4))
    im=axs[0].pcolormesh(E,G,ge,shading='auto',vmin=0,vmax=4)
    fig.colorbar(im,ax=axs[0],label=r'$G_{eff}$ (clipped at 4)')
    cs=axs[0].contour(E,G,ge,levels=[1],linewidths=2.5); axs[0].clabel(cs,fmt={1:r'$G_{eff}=1$'},fontsize=9)
    axs[0].axvline(0.5,ls=':',lw=1.7); axs[0].set(xlabel=r'Transmittance $\eta$',ylabel=r'Diffusion $\Gamma$',title='(a) Local survival map')
    for NSi in [10,30,100]: axs[1].plot(eta,gamma_boundary(eta,NSi),lw=2.4,label=fr'$N_S={NSi}$')
    axs[1].axvline(0.5,ls=':',lw=1.7); axs[1].set(xlim=(0.30,1),ylim=(0,5),xlabel=r'Transmittance $\eta$',ylabel=r'$\Gamma_{max}=\ln G_Q$',title='(b) Analytic boundary')
    axs[1].legend(fontsize=8); axs[1].grid(True,ls='--',alpha=0.35)
    im2=axs[2].pcolormesh(E,G,np.log10(np.maximum(pwrap,1e-12)),shading='auto',vmin=-12,vmax=0)
    fig.colorbar(im2,ax=axs[2],label=r'$\log_{10}P_{wrap}$')
    cs2=axs[2].contour(E,G,ge,levels=[1],colors='white',linewidths=2.2); axs[2].clabel(cs2,fmt={1:r'$G_{eff}=1$'},fontsize=9)
    cs3=axs[2].contour(E,G,pwrap,levels=[1e-6,1e-3,1e-2],linestyles='--',linewidths=1.2)
    axs[2].clabel(cs3,fmt={1e-6:'1e-6',1e-3:'1e-3',1e-2:'1e-2'},fontsize=8)
    pts=[('A',0.90,0.50),('B',0.90,1.50),('C',0.90,2.00)]
    for lab,x,y in pts:
        axs[2].plot(x,y,'o',ms=6); axs[2].text(x+0.015,y+0.08,lab,fontsize=10,fontweight='bold')
    axs[2].set(xlabel=r'Transmittance $\eta$',ylabel=r'Diffusion $\Gamma$',title='(c) Wrapping risk and QZZB points')
    for ax in axs: ax.grid(True,ls='--',alpha=0.25)
    fig.suptitle('Loss--diffusion survival, phase wrapping, and QZZB guardrail bridge',fontweight='bold')
    fig.tight_layout(); fig.savefig(ROOT/'fig4_survival_wrapping_bridge.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# ----------------------------- QZZB figs and toy scaling -----------------------------
def qzzb_figures():
    cfg=qzzbmod.ToyConfig(N_S_toy=1.5,ncut=8,M=1,W=np.pi,tau_points=21)
    qzzbmod.run_qzzb_toy_guardcheck(str(ROOT),cfg)
    qzzbmod.run_fig5_qzzb_guarded_rmse(str(ROOT),cfg)
    qzzbmod.run_idler_loss_sensitivity(str(ROOT),cfg)
    qzzbmod.run_phase_wrapping_risk_map(str(ROOT),cfg)
    qzzbmod.run_cutoff_convergence(str(ROOT),cfg)
    # toy NS scaling: intentionally lightweight
    cases=[('A local-valid',0.9,0.5),('B transition',0.9,1.5),('C stop-extrap.',0.9,2.0)]
    ns_values=[1.5,2.0,3.0]
    fig,ax=plt.subplots(figsize=(7.4,5.2))
    rows=[]
    for label,eta_s,Gamma in cases:
        vals=[]
        for ns in ns_values:
            # heuristic cutoff high enough for a stable but quick diagnostic
            ncut=8 if ns<=1.5 else 10
            rho=qzzbmod.build_noisy_tmsv_density(ns,ncut,eta_signal=eta_s,eta_idler=1.0,Gamma=Gamma)
            qzzb,*_=qzzbmod.qzzb_bound_for_state(rho,ncut,W=np.pi,tau_points=21)
            ns_eff=qzzbmod.truncated_signal_photon_number(ns,ncut)
            _, var_sur, ge = qzzbmod.local_phase_variances(eta_s, ns_eff, Gamma, M=1, a=1.0)
            vals.append(qzzb)
            rows.append((label,ns,ncut,ns_eff,eta_s,Gamma,float(ge),float(var_sur),float(qzzb),max(float(var_sur),float(qzzb))))
        ax.plot(ns_values,vals,marker='o',lw=2,label=label)
    ax.set_xlabel(r'Toy signal photons $N_S^{toy}$'); ax.set_ylabel(r'QZZB phase-variance lower bound')
    ax.set_title('Toy photon-number scaling diagnostic',fontweight='bold'); ax.grid(True,ls='--',alpha=0.35); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT/'toy_ns_scaling_guardrail.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    with open(ROOT/'toy_ns_scaling_guardrail.csv','w',encoding='utf-8') as f:
        f.write('case,N_S_toy,ncut,N_S_cutoff_effective,eta_signal,Gamma,G_eff_surrogate,Var_TMSV_surrogate,QZZB_phase_variance,Guarded_phase_variance\n')
        for r in rows: f.write(','.join(map(str,r))+'\n')


def main():
    fig1(); fig2(); fig3(); fig4(); qzzb_figures()
    print('All manuscript figures regenerated in', ROOT)

if __name__=='__main__':
    main()
