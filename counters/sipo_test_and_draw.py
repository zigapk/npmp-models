from models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sipo_model import three_bit_sipo


def calc_input(T, period=24, amp=100):
    # one for two clocks and then 0 for two clocks

    nth_period = np.floor(T / period)

    return 0 * (nth_period % 2) + amp * ((nth_period + 1) % 2)
    # return 100


def calc_clear(T, period=24, amp=100):
    if T > period * 10 and T < period * 11:
        return amp
    return amp if T < period else 0


"""
    TESTING
"""

# simulation parameters
t_end = 400
N = 1000


# # model parameters
alpha1 = 34.73  # protein_production
alpha2 = 49.36  # protein_production
alpha3 = 32.73  # protein_production
alpha4 = 49.54  # protein_production
delta1 = 1.93  # protein_degradation
delta2 = 0.69  # protein_degradation
Kd = 10.44  # Kd
n = 4.35  # hill
deltaE = delta1
KM = 1.0


def main():
    params_ff = (alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n, deltaE, KM)

    # three-bit counter with external clock
    # a1, not_a1, q1, not_q1, a2, not_a2, q2, not_q2, a3, not_a3, q3, not_q3
    Y0 = np.array([0] * 18)  # initial state
    T = np.linspace(0, t_end, N)  # vector of timesteps

    # numerical interation
    Y = odeint(three_bit_sipo, Y0, T, args=(params_ff, calc_input, calc_clear))

    Y_reshaped = np.split(Y, Y.shape[1], 1)

    # plotting the results
    Q1 = Y_reshaped[2] + 1
    not_Q1 = Y_reshaped[3]
    Q2 = Y_reshaped[6] + 2
    not_Q2 = Y_reshaped[7]
    Q3 = Y_reshaped[10] + 3
    not_Q3 = Y_reshaped[11]

    data_in = np.array([calc_input(x) for x in np.linspace(0, t_end, N)])
    data_clear = np.array([calc_clear(x) for x in np.linspace(0, t_end, N)])

    plt.plot(T, Q1, label="q1")
    plt.plot(T, Q2, label="q2")
    plt.plot(T, Q3, label="q3")
    # plt.plot(T, not_Q1, label='not q1')
    # plt.plot(T, not_Q2, label='not q2')
    plt.plot(T, data_in, "--", linewidth=2, label="input", color="green", alpha=0.5)
    plt.plot(T, data_clear, "--", linewidth=2, label="clear", color="red", alpha=0.5)

    plt.plot(T, get_clock(T), "--", linewidth=2, label="CLK", color="black", alpha=0.25)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
