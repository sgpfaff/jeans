import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
S="/tmp/claude-1003/-geir-data-scr-gabrielspace-jeans/c2b40140-c765-436f-8e08-af5382a50138/scratchpad"
MODEL,HALO="SIDM1b",1
ch=np.load(f"{S}/chain2_{MODEL}_{HALO}.npy"); bl=np.load(f"{S}/blobs2_{MODEL}_{HALO}.npy")
burn=400
fl=ch[burn:].reshape(-1,ch.shape[-1]); r1f=bl[burn:].reshape(-1)
ok=np.isfinite(r1f)&np.isfinite(fl).all(axis=1); fl,r1f=fl[ok],r1f[ok]
print(f"{len(fl)} live post-burn samples ({(~ok).sum()} dropped)")
data=np.column_stack([fl[:,0], fl[:,1], 10**fl[:,2], r1f, 10**fl[:,3]])
labels=[r"$\log_{10}M_{200}$", r"$c$", r"$\sigma/m$  [cm$^2$/g]",
        r"$r_1/R_{200}$  (derived)", "intrinsic scatter [dex]"]
LM_TRUE=np.log10(1.630e14)
fig=corner.corner(data, labels=labels, quantiles=[0.16,0.5,0.84], show_titles=True,
                  title_fmt=".3f", title_kwargs={"fontsize":9},
                  label_kwargs={"fontsize":10}, bins=40, color="C0")
n=data.shape[1]
for col,val,cc in ((0,LM_TRUE,"C2"),(2,1.0,"C3")):
    fig.axes[col*n+col].axvline(val,color=cc,lw=2)
    for row in range(col+1,n): fig.axes[row*n+col].axvline(val,color=cc,lw=1.3,alpha=.8)
    for c2 in range(col): fig.axes[col*n+c2].axhline(val,color=cc,lw=1.3,alpha=.8)
fig.axes[2].legend(handles=[plt.Line2D([],[],color="C3",lw=2,label=r"truth $\sigma/m=1$"),
                            plt.Line2D([],[],color="C2",lw=2,label=r"sim $M_{200}$")],
                   fontsize=9, loc="center", frameon=False)
fig.suptitle("EAGLE-50 SIDM1b halo 1 — recovering the cross-section from the DM profile\n"
             r"sampling $\sigma/m$ and solving for $r_1$;  recovered $1.42^{+0.12}_{-0.09}$ "
             r"against a true $1$ cm$^2$/g", fontsize=11, y=1.02)
out=f"{S}/corner_recovery_halo1.png"; fig.savefig(out,dpi=110,bbox_inches="tight")
print("wrote",out)
sm=10**fl[:,2]
print("  sigma/m 16/50/84:", np.round(np.percentile(sm,[16,50,84]),4))
print("  P(within factor 2 of truth):", f"{np.mean((sm>0.5)&(sm<2.0)):.1%}")
print("  P(sigma/m > 0.5):", f"{np.mean(sm>0.5):.1%}")
