import numpy as np
import math
import os

from hill_functions import *

from models import ff_ode_model_RS


def three_bit_sipo(Y, T, params, calc_input, calc_clear):
    (
        a1,
        not_a1,
        q1,
        not_q1,
        a2,
        not_a2,
        q2,
        not_q2,
        a3,
        not_a3,
        q3,
        not_q3,
        R1,
        S1,
        R2,
        S2,
        R3,
        S3,
    ) = Y

    clk = get_clock(T)

    d1 = calc_input(T)
    d2 = q1
    d3 = q2

    R1, R2, R3 = [calc_clear(T)] * 3

    Y_FF1 = [a1, not_a1, q1, not_q1, d1, clk, R1, S1]
    Y_FF2 = [a2, not_a2, q2, not_q2, d2, clk, R2, S2]
    Y_FF3 = [a3, not_a3, q3, not_q3, d3, clk, R3, S3]

    dY1 = ff_ode_model_RS(Y_FF1, T, params)
    dY2 = ff_ode_model_RS(Y_FF2, T, params)
    dY3 = ff_ode_model_RS(Y_FF3, T, params)

    dY = np.append(
        np.append(np.append(dY1, dY2), dY3),
        [
            R1,
            S1,
            R2,
            S2,
            R3,
            S3,
        ],
    )

    return dY
