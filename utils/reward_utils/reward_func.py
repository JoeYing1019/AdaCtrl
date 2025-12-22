try:
    from ..openmathinst_utils import extract_answer, math_equal
except:
    from utils.openmathinst_utils import extract_answer, math_equal

from math_verify import verify, parse
from typing import Union


def math_equal_ray(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    check_antlr_version: bool = True
) -> bool:
    return math_equal(prediction, reference, include_percentage, tolerance, timeout, check_antlr_version)


def verify_ray(
    gold, 
    target, 
    float_rounding: int=6,
    numeric_precision: int=15,
    strict: bool=True,
    timeout_seconds: int=3
) -> bool:
    return verify(gold, target, float_rounding, numeric_precision, strict, timeout_seconds)


def reward_func(data_source, solution_str, ground_truth, extra_info) -> float:
    omi_pred = None
    omi_correct = False
    mathv_pred = None
    mathv_correct = False
    try:
        omi_pred = extract_answer(solution_str, extract_from_boxed=True)
        omi_correct = math_equal_ray(omi_pred, ground_truth, check_antlr_version=False)
    except Exception:
        omi_correct = False

    acc = omi_correct
    score = 1.0 if acc else -1.0

    return {
        "score": score,
        "acc": acc,
        "ground_truth": str(ground_truth),
        "pred": str(omi_pred),
        "omi_correct": omi_correct,
    }


if __name__ == "__main__":
    solution = "To find the quotient when $x^6 - 3$ is divided by $x + 1,$ we can use polynomial long division. Let's go through the steps:\n\n1. Set up the division: $x^6 - 3 \\div (x + 1)$\n2. Divide the leading term of the dividend ($x^6$) by the leading term of the divisor ($x$), which gives $x^5$.\n3. Multiply the entire divisor ($x + 1$) by $x^5$ to get $x^6 + x^5$.\n4. Subtract this result from the original dividend: $(x^6 - 3) - (x^6 + x^5) = -x^5 - 3$.\n5. Repeat the process with the new dividend $-x^5 - 3$: divide the leading term $-x^5$ by $x$, which gives $-x^4$.\n6. Multiply the entire divisor ($x + 1$) by $-x^4$ to get $-x^5 - x^4$.\n7. Subtract this result from the new dividend: $(-x^5 - 3) - (-x^5 - x^4) = x^4 - 3$.\n8. Continue this process until the degree of the remainder is less than the degree of the divisor.\n\nFollowing these steps, we get the quotient as $x^5 - x^4 + x^3 - x^2 + x - 1$ and the remainder as $-2$. Therefore, the quotient when $x^6 - 3$ is divided by $x + 1$ is $\\boxed{x^5 - x^4 + x^3 - x^2 + x - 1}$."
    print(reward_func(None, solution, "x^5 - x^4 + x^3 - x^2 + x - 1", None))