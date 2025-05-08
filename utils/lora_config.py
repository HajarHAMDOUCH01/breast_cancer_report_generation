def LoraConfig class:
    def __init__(self, r=16, lora_alpha=32, lora_modules = ["qkv"], lora_dropout=0.1, task_type="CAUSAL_LM", bias="none"):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.target_modules = ["qkv"]  # ou ["query", "key", "value"] selon la structure
        self.task_type = task_type

    def to_dict(self):
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "lora_dropout": self.lora_dropout,
            "task_type": self.task_type,
        }
        