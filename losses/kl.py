def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    return 0.5 * torch.sum(
        logvar_p - logvar_q +
        (torch.exp(logvar_q) + (mu_q - mu_p)**2) / torch.exp(logvar_p) - 1,
        dim=-1
    )

def loss_fn(out):
    recon = F.mse_loss(out["x_pred"], out["x_target"], reduction="none")
    recon = recon.mean(dim=[1,2])

    kl = kl_divergence(
        out["mu_q"], out["logvar_q"],
        out["mu_p"], out["logvar_p"]
    )

    return (recon + kl).mean()