import re

def split_expression(expression):
    # Use regex to split by operators while keeping numbers (including multi-digit) together
    return re.findall(r'\d+|[^\d\s]', expression