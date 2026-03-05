import matplotlib.pyplot as plt


def plot_forecast(true, pred, title="Forecast"):
    plt.figure(figsize=(10, 4))
    plt.plot(true, label="True")
    plt.plot(pred, label="Predicted")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_decomposition(original, trend, seasonal):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(original)
    plt.title("Original")

    plt.subplot(3, 1, 2)
    plt.plot(trend)
    plt.title("Trend")

    plt.subplot(3, 1, 3)
    plt.plot(seasonal)
    plt.title("Seasonal")

    plt.tight_layout()
    plt.show()
