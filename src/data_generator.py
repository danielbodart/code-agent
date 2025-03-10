from hypothesis import strategies as st
from hypothesis.errors import NonInteractiveExampleWarning
import warnings

warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)

class DataGenerator:
    @staticmethod
    def generate_addition_example():
        a = st.integers().example()
        b = st.integers().example()
        return f"{a} + {b} = {a + b}"
