from .split_expression import split_expression

def reward_function(original, guessed):
    # Remove padding 'P' from the guessed string
    guessed = guessed.replace('P', '')
    
    # Check for presence of '?' in guessed
    if '?' in guessed:
        return -1  # Immediate penalty if '?' is present
    
    # Split original and guessed strings into tokens
    original_tokens = split_expression(original)
    guessed_tokens = split_expression(guessed)
    
    # Calculate reward based on correctness
    correct_count = sum(1 for orig_token, guess_token in zip(original_tokens, guessed_tokens) if orig_token == guess_token)
    total_tokens = len(original_tokens)
    
    # Ensure the reward is calculated correctly
    if total_tokens == 0:
        return 0
    
    # Scale reward based on correctness
    reward = correct_count / total_tokens
    
    # Apply additional penalties for incorrect guesses
    incorrect_count = total_tokens - correct_count
    reward -= 0.5 * incorrect_count / total_tokens
    
    return reward
