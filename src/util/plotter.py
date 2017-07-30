import matplotlib.pylab as pl
import numpy as np


class Plotter:
    def __init__(self, max_iterations, figure_name, error_name):
        self.figure_name = figure_name
        self.error_name = error_name

        #control information
        self.error_training = []
        self.error_validation = []
        self.accuracy_training = []
        self.accuracy_validation = []

        self.has_legend = False

        self.iteration_current = 1
        self.iteration_max = max_iterations

        pl.ion()

    def update_control_information(self, accuracy_training, accuracy_validation,
                                    total_training_error, total_validation_error):
        self.error_training.append(total_training_error)
        self.error_validation.append(total_validation_error)
        self.accuracy_training.append(accuracy_training)
        self.accuracy_validation.append(accuracy_validation)

    def update_plot(self):
        """
        Updates the runtime plot with the new accuracy and error values.
        If the legend has not yet been added to the plot, it will also be
        initialized.
        """
        if self.iteration_current == self.iteration_max - 1:
            pl.savefig(self.figure_name + ".png")
        x = range(self.iteration_current)
        pl.xlabel(u"Epochs")
        pl.figure(1)
        sp1 = pl.subplot(211)
        pl.xlim(0, self.iteration_max)
        pl.ylim(0, 1.0)
        pl.plot(x, self.accuracy_validation, 'r-', label='validation accuracy')
        pl.plot(x, self.accuracy_training, 'g-', label='training accuracy')

        sp2 = pl.subplot(212)
        pl.xlim(0, self.iteration_max)
        pl.ylim(0, np.max(self.error_training))
        pl.plot(x, self.error_training, 'g-', label=(self.error_name + ' training error'))
        pl.plot(x, self.error_validation, 'r-', label=(self.error_name + ' validation error'))
        if not self.has_legend:
            # Now add the legend with some customizations.
            sp1.legend(loc='upper right')
            sp2.legend(loc='upper right')
            self.has_legend = True
        self._display_plot()
        self.iteration_current += 1

    @staticmethod
    def _display_plot():
        pl.show()
        pl.pause(0.01)