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
N = int(2.5 * t_end)  # number of timesteps


# # model parameters
alpha1 = 33.00188789843441
alpha2 = 46.049242873562875
alpha3 = 18.091178106515247
alpha4 = 46.756580213281744
delta1 = 1.2310854733181351
delta2 = 0.36911726271103895
Kd = 7.458684990153726
n = 4.35  # hill
deltaE = delta1
KM = 1.0


def clock_driven_sipo(T):
    params_ff = (alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n, deltaE, KM)

    # three-bit counter with external clock
    # a1, not_a1, q1, not_q1, a2, not_a2, q2, not_q2, a3, not_a3, q3, not_q3
    Y0 = np.array([0] * 18)  # initial state

    # numerical interation
    Y = odeint(three_bit_sipo, Y0, T, args=(params_ff, calc_input, calc_clear))
    return Y


def input_from_different_module_sipo(T):
    params_ff = (alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n, deltaE, KM)
    # three-bit counter with external clock
    # a1, not_a1, q1, not_q1, a2, not_a2, q2, not_q2, a3, not_a3, q3, not_q3
    Y = np.zeros((N, 18))
    for idx in range(1, N):
        Y0 = Y[idx - 1, :]
        t = T[idx - 1 : idx + 1]
        x = calc_input(t[0])
        Y[idx, :] = odeint(
            three_bit_sipo, Y0, t, args=(params_ff, calc_input, calc_clear, x)
        )[1, :]

    return Y


def main():
    # three-bit counter with external clock
    # a1, not_a1, q1, not_q1, a2, not_a2, q2, not_q2, a3, not_a3, q3, not_q3
    # Y0 = np.array([0] * 18)  # initial state
    T = np.linspace(0, t_end, N)  # vector of timesteps
    Y = input_from_different_module_sipo(T)
    # Y = clock_driven_sipo(T)
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

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )  # Specify your desired width and height here

    # Plotting
    ax.plot(T, Q1, label="q1")
    ax.plot(T, Q2, label="q2")
    ax.plot(T, Q3, label="q3")
    # ax.plot(T, not_Q1, label='not q1')
    # ax.plot(T, not_Q2, label='not q2')
    ax.plot(T, data_in, "--", linewidth=2, label="input", color="green", alpha=0.5)
    ax.plot(T, data_clear, "--", linewidth=2, label="clear", color="red", alpha=0.5)
    ax.plot(T, get_clock(T), "--", linewidth=2, label="CLK", color="black", alpha=0.25)

    # Legend
    ax.legend()

    # Display the plot
    # plt.show()

    # Save the figure
    fig.savefig("plot.pdf", dpi=300)
    # save as pdf in a huge size
    plt.show()


if __name__ == "__main__":
    main()
