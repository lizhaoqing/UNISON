"""Compatibility shims for diffusers + transformers version mismatches."""


def patch_transformers_deepspeed() -> None:
    """Patch transformers.deepspeed for diffusers EMAModel.

    Must be called AFTER importing diffusers (diffusers reloads transformers).
    """
    import transformers

    if not hasattr(transformers, "deepspeed"):
        import transformers.integrations.deepspeed as deepspeed

        transformers.deepspeed = deepspeed
