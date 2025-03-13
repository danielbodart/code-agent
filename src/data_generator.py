from hypothesis import strategies as st
from hypothesis.errors import NonInteractiveExampleWarning
import warnings

warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)

def generate_addition_example():
    a = st.integers(min_value=0, max_value=9).example()
    b = st.integers(min_value=0, max_value=9).example()
    return f"{a}+{b}={a + b}"
