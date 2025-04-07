import matplotlib.pyplot as plt


class ShowPlot():
    def __init__(self):
        self.figuresize = (12,6)

    def lineplot(self, x, y, title, x_label, y_label, subplot=False):
        """
        Show a line plot with x, y and their labels
        :param subplot creates two subplots with the same x and different y, else creates a standard x/y plot
        """
        plt.figure(figsize=self.figuresize)

        if subplot:
            plt.plot(x, label=f'Train {y_label}')
            plt.plot(y, label=f'Validation {y_label}')
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()

        else:
            plt.plot(x, y)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)

        plt.show()
