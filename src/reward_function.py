def reward_function(original_token: str, guess: str) -> float:
    """Calculate the reward based on the original token and the guessed value."""
    # Check if both are operators
    operators = "+-*/="
    if original_token in operators and guess in operators:
        return 1.0 if original_token == guess else 0.5
    
    # Check if both are integers
    try:
        original_value = int(original_token)
        guess_value = int(guess)
        difference = abs(original_value - guess_value)
        return max(0, 1 - (difference / (abs(original_value) + 1)))
    except ValueError:
        return -1.0  # Penalty for non-integer guesses when original is integer
