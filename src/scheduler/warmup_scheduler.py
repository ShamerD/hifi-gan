class LinearWarmupScheduler:
    def __init__(self, d_model, total_steps, warmup_steps):
        self.d_model_coeff = d_model ** (-0.5)
        self.total_steps = total_steps
        self.warmup_coef = warmup_steps ** (-1.5)

    def __call__(self, step):
        return self.d_model_coeff * min((step + 1) ** (-0.5), (step + 1) * self.warmup_coef)
