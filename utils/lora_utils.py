from peft import get_peft_model, LoraConfig, TaskType

def inject_lora_into_vit_model(vit_model, lora_config):
    """
    Inject LoRA into the model.
    Args:
        model: The base model to inject LoRA into.
        lora_config: The LoRA configuration.
    Returns:
        The model with LoRA injected.
    """
    if lora_config is not None:
        vit_model_with_lora = get_peft_model(vit_model, lora_config)
        return vit_model_with_lora