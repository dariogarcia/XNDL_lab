    # Generate input data
    theta = np.random.uniform(0, 2*np.pi, size=(1000, 1))
    r = np.sqrt(np.random.uniform(0, 1, size=(1000, 1)))
    X = np.concatenate((r*np.cos(theta), r*np.sin(theta)), axis=1)
    y = np.zeros((1000, 1))
    y[np.where(r <= 0.5)] = 1
    y += np.random.normal(0, 0.1, size=y.shape)

    # Split into training and val sets
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]