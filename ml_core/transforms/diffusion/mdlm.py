"""MDLM (Masked Discrete Language Model) transform functions."""

import torch


class ToBool:
    """Convert tensor to boolean dtype."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to boolean dtype.

        :param tensor: Input tensor.
        :return: Tensor converted to torch.bool.
        """
        return tensor.bool()


class SampleMask:
    """Sample a random mask for masked language modeling.

    Samples k ~ U[1, L] and then samples mask ~ B(k/L), where:
    - k is the number of tokens to mask (sampled uniformly from 1 to sequence length L)
    - Each position has probability k/L of being masked (Bernoulli distribution)
    """

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Sample a random mask for the input.

        :param input_ids: Input token IDs tensor of shape [batch_size, seq_length].
        :return: Boolean mask tensor of shape [batch_size, seq_length] where True means mask.
        """
        batch_size, seq_length = input_ids.shape

        # Sample k ~ U[1, L] for each batch element
        k = torch.randint(1, seq_length + 1, (batch_size,), device=input_ids.device)

        # Sample mask ~ B(k/L) for each batch element
        # Generate random values and compare with k/L probability
        random_values = torch.rand(batch_size, seq_length, device=input_ids.device)
        mask = random_values < (k.float() / seq_length).unsqueeze(1)

        return mask


class MaskInputIds:
    """Apply a mask to input_ids by replacing masked positions with mask_token_id.

    Takes either mask_token_id directly or extracts it from a tokenizer.
    """

    def __init__(self, mask_token_id: int = None, tokenizer=None):
        """Initialize the mask function.

        :param mask_token_id: Token ID to use for masking. If None, must provide tokenizer.
        :param tokenizer: Tokenizer from which to extract mask_token_id. Used if mask_token_id is
            None.
        """
        if mask_token_id is None and tokenizer is None:
            raise ValueError("Either mask_token_id or tokenizer must be provided")

        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        else:
            self.mask_token_id = tokenizer.mask_token_id

    def __call__(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to input_ids.

        :param input_ids: Input token IDs tensor of shape [batch_size, seq_length].
        :param mask: Boolean mask tensor of shape [batch_size, seq_length] where True means mask.
        :return: Masked input_ids tensor of same shape as input.
        """
        # Clone input_ids to avoid modifying the original
        masked_input_ids = input_ids.clone()

        # Replace masked positions with mask_token_id
        masked_input_ids[mask] = self.mask_token_id

        return masked_input_ids


class MaskLoss:
    """Apply ignore_index to non-masked positions in targets for loss computation.

    This ensures the loss is only computed on masked positions, not the entire sequence.
    """

    def __init__(self, ignore_index: int = -100):
        """Initialize the mask loss function.

        :param ignore_index: Index to use for positions that should be ignored in loss computation.
            Default is -100, which is the standard ignore_index for CrossEntropyLoss.
        """
        self.ignore_index = ignore_index

    def __call__(self, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply ignore_index to non-masked positions in target.

        :param target: Target token IDs tensor of shape [batch_size, seq_length].
        :param mask: Boolean mask tensor of shape [batch_size, seq_length] where True means mask.
        :return: Modified target tensor with ignore_index at non-masked positions.
        """
        # Set non-masked positions to ignore_index
        masked_target = torch.where(mask, target, self.ignore_index)

        return masked_target
