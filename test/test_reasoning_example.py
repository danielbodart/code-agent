from src.reasoning_example import ReasoningExample


def test_creation():
    example = ReasoningExample("Question", "Answer")
    
    assert example.question == "Question"
    assert example.answer == "Answer"

def test_iteration():
    example = ReasoningExample("Question", "Answer")

    assert list(example) == ["Question", "Answer"]

def test_str():
    example = ReasoningExample("A", "D")
    
    assert str(example) == "A[SEP]D"
