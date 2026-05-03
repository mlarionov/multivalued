import numpy as np
from scipy.optimize import brentq

def f(y):
    return y + 0.3 * np.sin(2 * np.pi * y)

def get_y_roots(x_target):
    # Roots of f'(y) are approx 0.339 and 0.661
    intervals = [(0, 0.339), (0.339, 0.661), (0.661, 1)]
    roots = []
    for a, b in intervals:
        fa = f(a) - x_target
        fb = f(b) - x_target
        if fa * fb < 0:
            root = brentq(lambda y: f(y) - x_target, a, b)
            roots.append(root)
        elif abs(fa) < 1e-9:
            roots.append(a)
        elif abs(fb) < 1e-9:
            roots.append(b)
    
    # Clean up duplicates (e.g. at boundaries) and sort
    return sorted(list(set(np.round(roots, 8))))

def generate_dataset(num_samples=2000):
    x_values = np.linspace(-0.2, 1.2, num_samples)
    dataset = []
    for x in x_values:
        y_targets = get_y_roots(x)
        if y_targets:
            dataset.append((x, y_targets))
    return dataset

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = generate_dataset(500)
    xs = []
    ys = []
    for x, y_list in dataset:
        for y in y_list:
            xs.append(x)
            ys.append(y)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=2, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Generated Multivalued Dataset')
    plt.savefig('dataset_check.png')
    print(f"Generated {len(dataset)} samples. Plot saved to dataset_check.png")
