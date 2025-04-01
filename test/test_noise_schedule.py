from src.noise_schedule import noise_schedule


def test_specific_examples():
    for tokens in [512, 100, 10, 5, 4, 3, 2, 1]:
        schedule = list(noise_schedule(tokens))
        assert schedule[0] == 1
        assert sum(schedule) == tokens
        for i in range(1, len(schedule)):
            assert schedule[i] >= schedule[i-1], f"Failed with tokens={tokens}, schedule={schedule}"

def test_zero_tokens():
    schedule = list(noise_schedule(0))
    assert schedule == []

def test_negative_values():
    schedule = list(noise_schedule(-100))
    assert schedule == []
