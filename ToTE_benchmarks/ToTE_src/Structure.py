import random


class POD:
    def __init__(self, R, s):
        self.R = R
        self.s = s

    def get_bandwidth(self, other_pod):
        return min(self.s, other_pod.s)


def generate_matrix(N, d_range):
    matrix = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                matrix[(i, j)] = 0.0
            else:
                matrix[(i, j)] = random.uniform(d_range[0], d_range[1])
    return matrix


def init_structure(N, R_range, s_range, d_range,seed=42):
    random.seed(seed)
    pods = [POD(random.randint(R_range[0], R_range[1]), random.uniform(s_range[0], s_range[1])) for _ in range(N)]
    d_wave_matrix = generate_matrix(N, d_range)
    s_matrix = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                pass
            else:
                s_matrix[(i, j)] = pods[i].get_bandwidth(pods[j])

    return pods, d_wave_matrix, s_matrix


if __name__ == '__main__':
    N = 10  # Number of pods
    R_range = [3, 5]  # Range of uplink port counts
    s_range = [1, 2]  # Range of uplink bandwidth
    d_range = [2, 3]  # Demand range
    pods, d_wave, s_matrix = init_structure(N, R_range, s_range, d_range)

    print(1)
