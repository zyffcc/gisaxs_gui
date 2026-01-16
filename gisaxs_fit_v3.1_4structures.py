# -*- coding: utf-8 -*-
"""
Gisaxs_fit
Version for Python 3 using PyQt 5
by Christoph Schaffer
Modified by Florian Jung, Christian Weindl, Lennart Reb, Julian Heger

"""

import os
# print(os.environ['VIRTUAL_ENV'])
import sys
import timeit

import PyQt5
import lmfit
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import uic
from scipy.special import jv as bessel

__VERSION__ = '3.1'

# todo compare with original function to verify the maths is correct

qtCreatorFile = "WINDOW_gisaxs_model_v3.5_4structures.ui"
Ui_MainWindow, QtBaseClass = PyQt5.uic.loadUiType(qtCreatorFile)


# noinspection PyUnresolvedReferences
class GisaxsModeler(PyQt5.QtWidgets.QMainWindow, Ui_MainWindow):
    """
    GISAXS modeler by Christoph Schaffer
    """

    def __init__(self, parent=None):
        PyQt5.QtWidgets.QWidget.__init__(self, parent)
        Ui_MainWindow.__init__(self)  # self = WINDOW.Ui_MainWindow_GISAXS_MODEL()
        self.setupUi(self)

        # set window Title
        self.setWindowTitle('{0} - Version {1}'.format(self.windowTitle(), __VERSION__))
        plt.ion()  # draws all pyplot commands immediately , plt.ioff() turns interactive behaviour off
        self.fig, self.ax, self.leg, self.fig2, self.ax2 = object, object, None, object, object
        self.path = '.'
        self.filename_data = ''
        self.parameter_file_path = ''
        self.fit_in_progress = False
        self.fit_step = 0

        # eval bessel (todo add other form factors)
        print("Preparing Bessel function..", end='')
        self._evaluate_bessel_on_grid()
        print("done.")
        self.num_of_points_graph = 50
        self.num_of_points_gauss = 32
        self.ROI_MIN = 0
        self.ROI_MAX = 100000  # set this to arbitrarily big value without knowing yet what is biggest --> last index!
        self.xmin, self.xmax = 1e-4, 1
        self.R1Sum = 0
        self.R2Sum = 0
        self.R3Sum = 0
        self.R4Sum = 0
        self.R1Int = 0
        self.R2Int = 0
        self.R3Int = 0
        self.R4Int = 0
        self.R1Mean = 0
        self.R2Mean = 0
        self.R3Mean = 0
        self.R4Mean = 0
        self.chi_squared = 0.
        self.chi_squared_old = 1.

        # Initialize parameters (single parameters)
        self.Br = 0.001
        self.Nu = 5
        self.Int_resolution = 2
        self.Int_resolution_exp = 4

        self.Dc1 = 0
        self.Omega1 = 0.5
        self.R1 = 90
        self.Sigma1 = 0.3
        self.Int1 = 2.0
        self.Int1_exp = 2

        self.Dc2 = 0
        self.Omega2 = 0.5
        self.R2 = 30
        self.Sigma2 = 0.3
        self.Int2 = 2.0
        self.Int2_exp = 3

        self.Dc3 = 0
        self.Omega3 = 0.5
        self.R3 = 10.0
        self.Sigma3 = 0.3
        self.Int3 = 2.0
        self.Int3_exp = 4

        self.Dc4 = 0
        self.Omega4 = 0.5
        self.R4 = 10.0
        self.Sigma4 = 0.3
        self.Int4 = 2.0
        self.Int4_exp = 4

        self.Background = 10.0

        self.read_errors = False

        # create parameter array
        self.parameters = self.setup_parameter_array()
        # initialize the gui with the above defined values
        self.put_parameters_to_gui(self.parameters)

        self.data = None
        self.fit = None
        self.fit_result = None
        self.old_params = None
        self.old_red_chisqr = 1e12

        self.x = None
        self.chi_points = None
        self.create_x_chi_grid()

        self.resolution_peak = np.zeros(self.num_of_points_graph)
        self.Formfactor1 = np.zeros(self.num_of_points_graph)
        self.Formfactor2 = np.zeros(self.num_of_points_graph)
        self.Formfactor3 = np.zeros(self.num_of_points_graph)
        self.Formfactor4 = np.zeros(self.num_of_points_graph)
        self.Structure_Factor1 = np.zeros(self.num_of_points_graph)
        self.Structure_Factor2 = np.zeros(self.num_of_points_graph)
        self.Structure_Factor3 = np.zeros(self.num_of_points_graph)
        self.Structure_Factor4 = np.zeros(self.num_of_points_graph)

        # connect buttons
        self.button_BROWSE_INPUT_DAT.clicked.connect(self.get_data_from_gui_input_path)
        self.lineEdit_INPUT_DATA.textChanged.connect(self.read_input_data_file)
        self.checkBox_INCLUDE_LEFT_SIDE.setChecked(True)  # default to switch on left side mirror
        self.checkBox_INCLUDE_LEFT_SIDE.stateChanged.connect(self.read_input_data_file)
        self.button_BROWSE_PARAMETER.clicked.connect(self.set_parameter_path_and_load_parameters)
        self.button_SAVE.clicked.connect(self.save_input_parameter_file)
        self.button_TOGGLE_PLOT.clicked.connect(self.calc_and_plot)
        self.button_SAVE_PNG.clicked.connect(self.save_png_and_export_fit)
        self.button_FIT_LEASTSQ.clicked.connect(self.fit_leastsq)
        self.button_FIT_PARS.clicked.connect(self.fit_pars)
        self.pushButton_PARAMETER_gen.clicked.connect(self.get_input_parameter_file_from_data_file)

        self.pushButton_emcee.clicked.connect(self.fit_emcee)
        self.pushButton_save_dist.clicked.connect(self.save_distribution)

        self.checkBox_lookup.setChecked(True)
        self.comboBox_LEGEND_POS.setCurrentIndex(3)  # set legend position default to lower left

        self.checkBox_real_time.setChecked(False)
        self.checkBox_real_time.stateChanged.connect(self._activate_spin_boxes_for_plotting_real_time)

        # model lines
        self.vlineleft = None
        self.vlineright = None
        self.line_res = None
        self.line_ff1 = None
        self.line_ff2 = None
        self.line_ff3 = None
        self.line_ff4 = None
        self.line_sf1 = None
        self.line_sf2 = None
        self.line_sf3 = None
        self.line_sf4 = None
        self.line_bg = None
        self.line_fit = None

        # fitting parameter constraints
        self.Sigma1_min = 0.2
        self.Sigma2_min = 0.2
        self.Sigma3_min = 0.2
        self.Sigma4_min = 0.2
        self.Sigma1_max = 0.5#1
        self.Sigma2_max = 0.5#1
        self.Sigma3_max = 0.5#1
        self.Sigma4_max = 0.5

        self.Omega1_min = 0.2
        self.Omega2_min = 0.2
        self.Omega3_min = 0.2
        self.Omega4_min = 0.2
        self.Omega1_max = 0.5
        self.Omega2_max = 0.5
        self.Omega3_max = 0.5
        self.Omega4_max = 0.5

        self.D1_min = 2*self.R1#30
        self.D1_max = 300
        self.D2_min = 2*self.R2#10
        self.D2_max = 150
        self.D3_min = 2*self.R3#3
        self.D3_max = 60
        self.D4_min = 2*self.R3#3
        self.D4_max = 60

        self.R1_min = 20
        self.R1_max = 300
        self.R2_min = 5
        self.R2_max = 150
        self.R3_min = 1
        self.R3_max = 50
        self.R4_min = 1
        self.R4_max = 50

        self.put_fit_parameters_to_gui()

        self._update_plot_settings()

    def _activate_spin_boxes_for_plotting_real_time(self):

        boxes_update_fit = [self.doubleSpinBox_RESOLUTION_WIDTH.valueChanged,
                            self.doubleSpinBox_RESOLUTION_INT.valueChanged,
                            self.doubleSpinBox_RESOLUTION_INT_EXP.valueChanged,
                            self.doubleSpinBox_RESOLUTION_EXPONENT.valueChanged,
                            self.doubleSpinBox_BACKGROUND.valueChanged,
                            self.doubleSpinBox_INT1.valueChanged, self.doubleSpinBox_INT1_EXP.valueChanged,
                            self.doubleSpinBox_INT2.valueChanged, self.doubleSpinBox_INT2_EXP.valueChanged,
                            self.doubleSpinBox_INT3.valueChanged, self.doubleSpinBox_INT3_EXP.valueChanged,
                            self.doubleSpinBox_INT4.valueChanged, self.doubleSpinBox_INT4_EXP.valueChanged,
                            self.doubleSpinBox_DC1.valueChanged,
                            self.doubleSpinBox_DC2.valueChanged,
                            self.doubleSpinBox_DC3.valueChanged,
                            self.doubleSpinBox_DC4.valueChanged,
                            self.doubleSpinBox_OMEGA1.valueChanged,
                            self.doubleSpinBox_OMEGA2.valueChanged,
                            self.doubleSpinBox_OMEGA3.valueChanged,
                            self.doubleSpinBox_OMEGA4.valueChanged,
                            self.doubleSpinBox_R1.valueChanged,
                            self.doubleSpinBox_R2.valueChanged,
                            self.doubleSpinBox_R3.valueChanged,
                            self.doubleSpinBox_R4.valueChanged,
                            self.doubleSpinBox_SIGMA1.valueChanged,
                            self.doubleSpinBox_SIGMA2.valueChanged,
                            self.doubleSpinBox_SIGMA3.valueChanged,
                            self.doubleSpinBox_SIGMA4.valueChanged,
                            self.doubleSpinBox_NUM_OF_POINTS_GAUSS.valueChanged,
                            self.doubleSpinBox_NUM_OF_POINTS_GRAPH.valueChanged,
                            self.spinBox_ROI_MIN.valueChanged,
                            self.spinBox_ROI_MAX.valueChanged,
                            self.checkBox_SPHERES1.toggled,
                            self.checkBox_SPHERES2.toggled,
                            self.checkBox_SPHERES3.toggled,
                            self.checkBox_SPHERES4.toggled,
                            self.checkBox_NORM_FF.toggled,
                            ]

        boxes_calc_plot = [self.checkBox_LEGEND.toggled,
                           self.comboBox_LEGEND_POS.currentTextChanged,
                           # self.lineEdit_INPUT_DATA.textChanged,  # call these explicitely in functions, else creates
                           # self.lineEdit_PARAMETER.textChanged,  # async. cause plot refreshes b4 new data is loaded
                           ]

        if self.checkBox_real_time.isChecked():
            for box_changed in boxes_update_fit:
                box_changed.connect(self.update_fit)
            for box_changed in boxes_calc_plot:
                box_changed.connect(self.calc_and_plot)
            self.calc_and_plot()
            print("Started real-time plotting.")
            self.checkBox_lookup.setChecked(True)  # to accelerate for real-time plotting

        else:
            for box in boxes_update_fit:
                box.disconnect()
            for box in boxes_calc_plot:
                box.disconnect()
            print("Stopped real-time plotting.")

    def update_fit(self):
        if not isinstance(self.fig, plt.Figure):
            return None
            # self.calc_and_plot()
        self.calc_components_from_gui()
        self.get_fit_from_components()
        self._update_model_lines()

    def create_x_chi_grid(self):
        if self.data is None:
            self.x = np.logspace(np.log10(self.xmin), np.log10(self.xmax), int(self.num_of_points_graph))
            self.chi_points = np.zeros(self.num_of_points_graph)
        else:
            try:
                mini, maxi = 10 ** self.data[:, 0].min(), 10 ** self.data[:, 0].max()
                self.x = np.logspace(np.log10(mini), np.log10(maxi), int(self.num_of_points_graph))
                self.chi_points = np.array([abs(self.x - d).argmin() for d in 10 ** self.data[:, 0]])
            except ValueError:
                print("Problems setting up evaluation grid - is your data correct?")

    def set_parameters_from_gui(self):

        self.num_of_points_gauss = int(self.doubleSpinBox_NUM_OF_POINTS_GAUSS.value())
        self.num_of_points_graph = int(self.doubleSpinBox_NUM_OF_POINTS_GRAPH.value())
        self.Background = self.doubleSpinBox_BACKGROUND.value()
        self.Br = self.doubleSpinBox_RESOLUTION_WIDTH.value()
        self.Nu = self.doubleSpinBox_RESOLUTION_EXPONENT.value()
        self.Int_resolution = self.doubleSpinBox_RESOLUTION_INT.value()
        self.Int_resolution_exp = int(self.doubleSpinBox_RESOLUTION_INT_EXP.value())
        self.Dc1 = self.doubleSpinBox_DC1.value()
        self.Dc2 = self.doubleSpinBox_DC2.value()
        self.Dc3 = self.doubleSpinBox_DC3.value()
        self.Dc4 = self.doubleSpinBox_DC4.value()
        self.Omega1 = self.doubleSpinBox_OMEGA1.value()
        self.Omega2 = self.doubleSpinBox_OMEGA2.value()
        self.Omega3 = self.doubleSpinBox_OMEGA3.value()
        self.Omega4 = self.doubleSpinBox_OMEGA4.value()
        self.R1 = self.doubleSpinBox_R1.value()
        self.R2 = self.doubleSpinBox_R2.value()
        self.R3 = self.doubleSpinBox_R3.value()
        self.R4 = self.doubleSpinBox_R4.value()
        self.Sigma1 = self.doubleSpinBox_SIGMA1.value()
        self.Sigma2 = self.doubleSpinBox_SIGMA2.value()
        self.Sigma3 = self.doubleSpinBox_SIGMA3.value()
        self.Sigma4 = self.doubleSpinBox_SIGMA4.value()
        self.Int1 = self.doubleSpinBox_INT1.value()
        self.Int2 = self.doubleSpinBox_INT2.value()
        self.Int3 = self.doubleSpinBox_INT3.value()
        self.Int4 = self.doubleSpinBox_INT4.value()
        self.Int1_exp = int(self.doubleSpinBox_INT1_EXP.value())
        self.Int2_exp = int(self.doubleSpinBox_INT2_EXP.value())
        self.Int3_exp = int(self.doubleSpinBox_INT3_EXP.value())
        self.Int4_exp = int(self.doubleSpinBox_INT4_EXP.value())

        self.ROI_MIN = int(self.spinBox_ROI_MIN.value())
        self.ROI_MAX = int(self.spinBox_ROI_MAX.value())

        # fit limits
        self.D1_min = 2*self.R1#self.doubleSpinBox_D1_min.value()
        self.D1_max = self.doubleSpinBox_D1_max.value()
        self.D2_min = 2*self.R2#self.doubleSpinBox_D2_min.value()
        self.D2_max = self.doubleSpinBox_D2_max.value()
        self.D3_min = 2*self.R3#self.doubleSpinBox_D3_min.value()
        self.D3_max = self.doubleSpinBox_D3_max.value()
        self.D4_min = 2*self.R4#self.doubleSpinBox_D4_min.value()
        self.D4_max = self.doubleSpinBox_D4_max.value()

        self.R1_min = self.doubleSpinBox_R1_min.value()
        self.R1_max = self.doubleSpinBox_R1_max.value()
        self.R2_min = self.doubleSpinBox_R2_min.value()
        self.R2_max = self.doubleSpinBox_R2_max.value()
        self.R3_min = self.doubleSpinBox_R3_min.value()
        self.R3_max = self.doubleSpinBox_R3_max.value()
        self.R4_min = self.doubleSpinBox_R4_min.value()
        self.R4_max = self.doubleSpinBox_R4_max.value()

        self.Omega1_min = self.doubleSpinBox_OMEGA1_min.value()
        self.Omega1_max = self.doubleSpinBox_OMEGA1_max.value()
        self.Omega2_min = self.doubleSpinBox_OMEGA2_min.value()
        self.Omega2_max = self.doubleSpinBox_OMEGA2_max.value()
        self.Omega3_min = self.doubleSpinBox_OMEGA3_min.value()
        self.Omega3_max = self.doubleSpinBox_OMEGA3_max.value()
        self.Omega4_min = self.doubleSpinBox_OMEGA4_min.value()
        self.Omega4_max = self.doubleSpinBox_OMEGA4_max.value()

        self.Sigma1_min = self.doubleSpinBox_SIGMA1_min.value()
        self.Sigma1_max = self.doubleSpinBox_SIGMA1_max.value()
        self.Sigma2_min = self.doubleSpinBox_SIGMA2_min.value()
        self.Sigma2_max = self.doubleSpinBox_SIGMA2_max.value()
        self.Sigma3_min = self.doubleSpinBox_SIGMA3_min.value()
        self.Sigma3_max = self.doubleSpinBox_SIGMA3_max.value()
        self.Sigma4_min = self.doubleSpinBox_SIGMA4_min.value()
        self.Sigma4_max = self.doubleSpinBox_SIGMA4_max.value()

    def get_data_from_gui_input_path(self):
        filename = PyQt5.QtWidgets.QFileDialog.getOpenFileName(self, 'Open File',
                                                               directory=self.path,
                                                               filter="Data file (*.dat);;Data file log(values) ("
                                                                      "*.gen);;All files (*.*)")[0]
        # print(filename)
        try:
            self.lineEdit_INPUT_DATA.setText(filename)
        except TypeError:
            # unpack tuple whatever why - maybe new with pyqt5?
            self.lineEdit_INPUT_DATA.setText(filename[0])

        self.filename_data = filename
        print('Set data file path: {0}'.format(self.filename_data))
        self.path = os.path.dirname(str(filename))
        # self.read_input_data_file()

    def read_input_data_file(self):  # reads out data!!!
        """
        Input file: separator/delimiter = '\t', comments = '#'
        *.gen:  file contains log10(q), log10(I)
        *.dat:  file contains q, I  (also for all other extensions)
        :return:
        """

        try:
            self.filename_data = str(self.lineEdit_INPUT_DATA.text())
            if self.filename_data.startswith('file://'):
                self.filename_data = self.filename_data[7:-2]
            self.lineEdit_INPUT_DATA.setText(self.filename_data)
            print('Loading data: {0}'.format(self.filename_data))
            self.data = np.genfromtxt(self.filename_data, comments='#', missing_values=np.nan, delimiter='\t')

            # check if data has more than 3 columns and if the 3rd column contains data
            # for dpdak linecuts the 3rd column contains only nan values - avoid this
            if np.shape(self.data)[1] > 2:
                if any(~np.isnan(self.data[:,2])):  # if any error value that is not nan?
                    self.read_errors = True
                    print('Using errorbars.')
            else:
                self.read_errors = False

            # sorting data according to q values
            # don't used for *.gen files, they contain only one side of the cut with log10 values (opposite sorting)
            if self.checkBox_INCLUDE_LEFT_SIDE.isChecked() and not self.filename_data.lower().endswith('gen'):
                self.data[:, 0] = np.abs(self.data[:, 0])  # get absolute values
                sort_idx = np.argsort(self.data[:, 0])  # get sort_idx of values
                self.data = self.data[sort_idx, :]  # sort data according to absolute q_values

            # get rid of invalid entries from your file...
            # delete all data entries where q or I = 0
            # todo could shorten this with nonzero ..
            print('Preparing data: ', end='\r')
            self.data = np.delete(self.data, np.where(self.data[:, 0] == 0.0), 0)
            self.data = np.delete(self.data, np.where(self.data[:, 1] == 0.0), 0)
            print('Preparing data: deleted zero-entries', end='\r')

            self.data = np.delete(self.data, np.where(np.isnan(self.data[:, 0])), 0)
            self.data = np.delete(self.data, np.where(np.isnan(self.data[:, 1])), 0)
            print('Preparing data: deleted zero-entries, nan-entries', end='\r')

            self.data = np.delete(self.data, np.where(np.isinf(self.data[:, 0])), 0)
            self.data = np.delete(self.data, np.where(np.isinf(self.data[:, 1])), 0)
            print('Preparing data: deleted zero-entries, nan-entries, inf-entries', end='\r')
            #            print where(self.data<0)
            if not self.filename_data.lower().endswith('gen'):
                self.data = np.delete(self.data, np.where(self.data[:, 0] <= 0), 0)
                self.data = np.delete(self.data, np.where(self.data[:, 1] <= 0), 0)
                self.data = np.log10(self.data)
            self.label_CURRENT_INPUT_DATA.setText(
                'Current Input Data: {0}'.format(self.filename_data))

            self.create_x_chi_grid()

            self.ROI_MAX = self.data.size
            self.spinBox_ROI_MAX.setValue(self.ROI_MAX)
            print('Preparing data: deleted zero-entries, nan-entries, inf-entries, and set ROI.')

            if self.checkBox_real_time.isChecked():
                self.calc_and_plot()

        except (OSError, FileNotFoundError):
            self.label_CURRENT_INPUT_DATA.setText('File does not exist!')
            print('Could not load data from {0}'.format(self.filename_data))
        except ValueError:
            self.label_CURRENT_INPUT_DATA.setText('Wrong file type?')
            print('Could not load data from {0}.'.format(self.filename_data))

    def get_input_parameter_file_from_data_file(self):
        filename = self.filename_data.rsplit('.', 1)[0] + '.par'
        self.lineEdit_PARAMETER.setText(filename)

    def set_parameter_path_and_load_parameters(self):
        filename = PyQt5.QtWidgets.QFileDialog.getOpenFileName(self, 'Select Parameter File',
                                                               directory=self.path,
                                                               filter="Parameter file (*.par);;All files (*.*)")[0]

        self.lineEdit_PARAMETER.setText(filename)
        # print('Set parameter file path: {0}'.format(filename))
        self.parameter_file_path = str(self.lineEdit_PARAMETER.text())
        # print('reading parameter input file...')
        try:
            self.label_CURRENT_INPUT_PARAMETER.setText(
                'Current Input Parameter File: {0}'.format(self.parameter_file_path))
            parameters = np.genfromtxt(self.parameter_file_path, comments='#',
                                       dtype='float64')

            # make parameters to self.parameters, update gui with them and make them also to the single self parameters
            self.parameters = parameters

            self.put_parameters_to_gui(self.parameters)
            self.put_parameters_to_self(self.parameters)

            self.put_fit_parameters_to_gui()

            print('Loaded parameters: {0}'.format(
                self.parameter_file_path))

            if self.checkBox_real_time.isChecked():
                self.calc_and_plot()

        except (FileNotFoundError, IOError, IndexError, UnicodeDecodeError):
            # IOError if file not found? and IndexError if not array with desired index
            print(f"Could not load parameters from {self.parameter_file_path}.")
            self.label_CURRENT_INPUT_PARAMETER.setText(
                'No parameter file loaded.')

    def put_fit_parameters_to_gui(self):

        self.doubleSpinBox_R1_min.setValue(self.R1_min)
        self.doubleSpinBox_R1_max.setValue(self.R1_max)
        self.doubleSpinBox_SIGMA1_min.setValue(self.Sigma1_min)
        self.doubleSpinBox_SIGMA1_max.setValue(self.Sigma1_max)
        self.doubleSpinBox_D1_min.setValue(self.D1_min)
        self.doubleSpinBox_D1_max.setValue(self.D1_max)
        self.doubleSpinBox_OMEGA1_min.setValue(self.Omega1_min)
        self.doubleSpinBox_OMEGA1_max.setValue(self.Omega1_max)

        self.doubleSpinBox_R2_min.setValue(self.R2_min)
        self.doubleSpinBox_R2_max.setValue(self.R2_max)
        self.doubleSpinBox_SIGMA2_min.setValue(self.Sigma2_min)
        self.doubleSpinBox_SIGMA2_max.setValue(self.Sigma2_max)
        self.doubleSpinBox_D2_min.setValue(self.D2_min)
        self.doubleSpinBox_D2_max.setValue(self.D2_max)
        self.doubleSpinBox_OMEGA2_min.setValue(self.Omega2_min)
        self.doubleSpinBox_OMEGA2_max.setValue(self.Omega2_max)

        self.doubleSpinBox_R3_min.setValue(self.R3_min)
        self.doubleSpinBox_R3_max.setValue(self.R3_max)
        self.doubleSpinBox_SIGMA3_min.setValue(self.Sigma3_min)
        self.doubleSpinBox_SIGMA3_max.setValue(self.Sigma3_max)
        self.doubleSpinBox_D3_min.setValue(self.D3_min)
        self.doubleSpinBox_D3_max.setValue(self.D3_max)
        self.doubleSpinBox_OMEGA3_min.setValue(self.Omega3_min)
        self.doubleSpinBox_OMEGA3_max.setValue(self.Omega3_max)


        self.doubleSpinBox_R4_min.setValue(self.R4_min)
        self.doubleSpinBox_R4_max.setValue(self.R4_max)
        self.doubleSpinBox_SIGMA4_min.setValue(self.Sigma4_min)
        self.doubleSpinBox_SIGMA4_max.setValue(self.Sigma4_max)
        self.doubleSpinBox_D4_min.setValue(self.D4_min)
        self.doubleSpinBox_D4_max.setValue(self.D4_max)
        self.doubleSpinBox_OMEGA4_min.setValue(self.Omega4_min)
        self.doubleSpinBox_OMEGA4_max.setValue(self.Omega4_max)

    def save_input_parameter_file(self):
        self.set_parameters_from_gui()  # update parameter variables
        self.parameter_file_path = str(self.lineEdit_PARAMETER.text())
        self.label_CURRENT_INPUT_PARAMETER.setText(
            'Current Input Parameter File: {0}'.format(self.parameter_file_path))

        try:
            f = open(self.parameter_file_path, 'w')
            f.write("# parameters:\n" +
                    "# num_of_points_graph, num_of_points_gauss, Br, Nu, Int_resolution, Int_resolution_exp\n" +
                    "# Dc1, Omega1, R1, Sigma1, Int1, Int1_exp\n" +
                    "# Dc2, Omega2, R2, Sigma2, Int2, Int2_exp\n" +
                    "# Dc3, Omega3, R3, Sigma3, Int3, int3_exp\n" +
                    "# Dc4, Omega4, R4, Sigma4, Int4, int4_exp\n" +
                    "# Background, 0, 0, 0, ROI_MIN, ROI_MAX\n" +
                    "# \n")

            self.parameters = self.setup_parameter_array()

            for a in self.parameters:
                f.write(str(a[0]) + '\t' + str(a[1]) + '\t' + str(a[2]) + '\t' +
                        str(a[3]) + '\t' + str(a[4]) + '\t' + str(a[5]) + '\n')

            self.calc_components_from_gui()

            f.write('# Mean Radii:\n# <R1>\t<R2>\t<R3>\t<R4>\n# {0}\t{1}\t{2}'.format(
                self.R1Mean, self.R2Mean, self.R3Mean, self.R4Mean))

            f.close()
            print('Saved parameters to: {0}'.format(self.parameter_file_path))
        except FileNotFoundError:
            print(f"Could not save file - check parameter file path: {self.parameter_file_path}")

    def put_parameters_to_self(self, parameters):

        self.num_of_points_graph = parameters[0, 0]
        self.num_of_points_gauss = parameters[0, 1]
        self.Br = parameters[0, 2]
        self.Nu = parameters[0, 3]
        self.Int_resolution = parameters[0, 4]
        self.Int_resolution_exp = parameters[0, 5]

        self.Dc1 = parameters[1, 0]
        self.Omega1 = parameters[1, 1]
        self.R1 = parameters[1, 2]
        self.Sigma1 = parameters[1, 3]
        self.Int1 = parameters[1, 4]
        self.Int1_exp = parameters[1, 5]

        self.Dc2 = parameters[2, 0]
        self.Omega2 = parameters[2, 1]
        self.R2 = parameters[2, 2]
        self.Sigma2 = parameters[2, 3]
        self.Int2 = parameters[2, 4]
        self.Int2_exp = parameters[2, 5]

        self.Dc3 = parameters[3, 0]
        self.Omega3 = parameters[3, 1]
        self.R3 = parameters[3, 2]
        self.Sigma3 = parameters[3, 3]
        self.Int3 = parameters[3, 4]
        self.Int3_exp = parameters[3, 5]

        self.Dc4 = parameters[4, 0]
        self.Omega4 = parameters[4, 1]
        self.R4 = parameters[4, 2]
        self.Sigma4 = parameters[4, 3]
        self.Int4 = parameters[4, 4]
        self.Int4_exp = parameters[4, 5]

        self.Background = parameters[5, 0]
        self.ROI_MIN = int(parameters[5, 4])
        self.ROI_MAX = int(parameters[5, 5])

    def put_parameters_to_gui(self, parameters):
        """Only model parameters"""
        self.doubleSpinBox_NUM_OF_POINTS_GRAPH.setValue(parameters[0, 0])
        self.doubleSpinBox_NUM_OF_POINTS_GAUSS.setValue(parameters[0, 1])
        self.doubleSpinBox_RESOLUTION_WIDTH.setValue(parameters[0, 2])
        self.doubleSpinBox_RESOLUTION_EXPONENT.setValue(parameters[0, 3])
        self.doubleSpinBox_RESOLUTION_INT.setValue(parameters[0, 4])
        self.doubleSpinBox_RESOLUTION_INT_EXP.setValue(parameters[0, 5])

        self.doubleSpinBox_DC1.setValue(parameters[1, 0])
        self.doubleSpinBox_OMEGA1.setValue(parameters[1, 1])
        self.doubleSpinBox_R1.setValue(parameters[1, 2])
        self.doubleSpinBox_SIGMA1.setValue(parameters[1, 3])
        self.doubleSpinBox_INT1.setValue(parameters[1, 4])
        self.doubleSpinBox_INT1_EXP.setValue(parameters[1, 5])

        self.doubleSpinBox_DC2.setValue(parameters[2, 0])
        self.doubleSpinBox_OMEGA2.setValue(parameters[2, 1])
        self.doubleSpinBox_R2.setValue(parameters[2, 2])
        self.doubleSpinBox_SIGMA2.setValue(parameters[2, 3])
        self.doubleSpinBox_INT2.setValue(parameters[2, 4])
        self.doubleSpinBox_INT2_EXP.setValue(parameters[2, 5])

        self.doubleSpinBox_DC3.setValue(parameters[3, 0])
        self.doubleSpinBox_OMEGA3.setValue(parameters[3, 1])
        self.doubleSpinBox_R3.setValue(parameters[3, 2])
        self.doubleSpinBox_SIGMA3.setValue(parameters[3, 3])
        self.doubleSpinBox_INT3.setValue(parameters[3, 4])
        self.doubleSpinBox_INT3_EXP.setValue(parameters[3, 5])

        self.doubleSpinBox_DC4.setValue(parameters[4, 0])
        self.doubleSpinBox_OMEGA4.setValue(parameters[4, 1])
        self.doubleSpinBox_R4.setValue(parameters[4, 2])
        self.doubleSpinBox_SIGMA4.setValue(parameters[4, 3])
        self.doubleSpinBox_INT4.setValue(parameters[4, 4])
        self.doubleSpinBox_INT4_EXP.setValue(parameters[4, 5])

        self.doubleSpinBox_BACKGROUND.setValue(parameters[5, 0])
        self.spinBox_ROI_MIN.setValue(parameters[5, 4])
        self.spinBox_ROI_MAX.setValue(parameters[5, 5])

    def setup_parameter_array(self):
        """ take the currently saved class parameters and return parameters filled with it"""
        parameters = np.array([
            [self.num_of_points_graph, self.num_of_points_gauss, self.Br,
             self.Nu, self.Int_resolution, self.Int_resolution_exp],
            [self.Dc1, self.Omega1, self.R1,
             self.Sigma1, self.Int1, self.Int1_exp],
            [self.Dc2, self.Omega2, self.R2,
             self.Sigma2, self.Int2, self.Int2_exp],
            [self.Dc3, self.Omega3, self.R3,
             self.Sigma3, self.Int3, self.Int3_exp],
            [self.Dc4, self.Omega4, self.R4,
             self.Sigma4, self.Int4, self.Int4_exp],
            [self.Background, 0, 0, 0, self.ROI_MIN, self.ROI_MAX]
        ])
        return parameters

    @staticmethod
    def _calc_resolution_peak(x, Br, Nu):
        # print('evaluating new resolution peak ... ', end='')
        resolution_peak = 1 / (1 + (x / Br) ** Nu)
        return resolution_peak

    @staticmethod
    def _calc_form_factor(x, num_of_points_gauss, R, Sigma, spheres, bessel_grid_eval, grid, norm_ff, lookup):
        """
        faster method for calculating FormFactor, using numpy-arrays instead of for-loops
        """
        # t1_start = timeit.default_timer()

        Sigma = R * Sigma

        # create x-array with (n, 1) dimensions
        x = x[:, np.newaxis]

        if Sigma != 0.:

            igrid = np.arange(0, num_of_points_gauss + 1)
            Rvar = (R - 3.0 * Sigma * (1.0 - 2.0 * igrid / num_of_points_gauss))

            # get idx where Rvar > 0
            idx = np.where(Rvar > 0)
            # reduce Rvar to positive values
            Rvar = Rvar[idx]
            # weight (y - value) for each Rvar
            Weight = np.exp(-((Rvar - R) ** 2) / (2.0 * Sigma ** 2))
            Weight /= Weight.sum()
        else:
            Rvar = R
            Weight = 1.

        if spheres:
            #  prefactor 64 * np.pi**3 according to http://gisaxs.com/index.php/Form_Factor:Sphere
            Formfactor = Weight * ((np.sin(x * Rvar) - x * Rvar * np.cos(x * Rvar)) / x ** 3) ** 2
        else:
            if lookup:
                try:
                    # prefactor 16*np.pi**2 * height according to lazzari 2002
                    Formfactor = Weight * (
                            Rvar * GisaxsModeler._lookup_bessel_from_grid(bessel_grid_eval, grid, Rvar * x) / x) ** 2
                except IndexError:
                    print('LIMITS: ', np.min(Rvar * x), np.max(Rvar * x))
                    Formfactor = Weight * (
                            Rvar * bessel(1, Rvar * x) / x) ** 2  # fall back to slower normal calculation
            else:
                Formfactor = Weight * (Rvar * bessel(1, Rvar * x) / x) ** 2

        RSum = np.sum(Weight)
        RInt = np.sum(Rvar * Weight)
        Formfactor = np.sum(Formfactor, axis=1)

        if norm_ff:
            Formfactor = Formfactor / Formfactor.max()
        else:
            # intensities have no absolute definitions. this multiplicator is introduced to not have negative exponents.
            mag_scale = 6
            if not spheres:
                Formfactor = Formfactor * 10 ** -mag_scale
            else:
                Formfactor = Formfactor * 10 ** -(mag_scale + 3)  # scale further down to achieve comparable intensities

        RMean = RInt / RSum

        return Formfactor, RMean

    @staticmethod
    def _calc_structure_factor(x, Dc, Omega):
        # t1_start = timeit.default_timer()

        if Dc == 0.:
            return np.ones(shape=x.shape)
        else:
            # all calculation is for qz=0!
            # todo check maths!
            omega = Omega * Dc

            # reasonable value of omega: up to 300
            arg = np.pi * omega ** 2 * x ** 2
            arg[arg > 100] = 100  # limiting this to 100 limits exponential below to ~ 10**86 to avoid overflow warning
            exp = np.exp(arg)
            expsq = exp ** 2
            Structure_Factor = -(1 - expsq) / (1 + expsq - 2 * exp * np.cos(x * Dc))
            # Structure_Factor[np.isnan(Structure_Factor)] = 1  # brute way to ensure definedness
            # t1_stop = timeit.default_timer()
            # print('elapsed time calc_struc_factor:', t1_stop - t1_start, 's')
            return Structure_Factor

    @staticmethod
    def _calc_model_components(x, n_pts_gauss, Br, Nu, grid, bessel, normalize, R1, Sigma1, sphere1, R2,
                               Sigma2, sphere2, R3, Sigma3, sphere3, R4, Sigma4, sphere4, Dc1, Omega1, Dc2, Omega2, Dc3, Omega3, Dc4, Omega4, lookup):

        resolution_peak = GisaxsModeler._calc_resolution_peak(x, Br, Nu)

        ff1, R1Mean = GisaxsModeler._calc_form_factor(x, n_pts_gauss, R1, Sigma1, sphere1, bessel, grid, normalize,
                                                      lookup)
        ff2, R2Mean = GisaxsModeler._calc_form_factor(x, n_pts_gauss, R2, Sigma2, sphere2, bessel, grid, normalize,
                                                      lookup)
        ff3, R3Mean = GisaxsModeler._calc_form_factor(x, n_pts_gauss, R3, Sigma3, sphere3, bessel, grid, normalize,
                                                      lookup)
        ff4, R4Mean = GisaxsModeler._calc_form_factor(x, n_pts_gauss, R4, Sigma4, sphere4, bessel, grid, normalize,
                                                      lookup)

        sf1 = GisaxsModeler._calc_structure_factor(x, Dc1, Omega1)
        sf2 = GisaxsModeler._calc_structure_factor(x, Dc2, Omega2)
        sf3 = GisaxsModeler._calc_structure_factor(x, Dc3, Omega3)
        sf4 = GisaxsModeler._calc_structure_factor(x, Dc4, Omega4)

        return resolution_peak, ff1, R1Mean, ff2, R2Mean, ff3, R3Mean, ff4, R4Mean, sf1, sf2, sf3, sf4

    def calc_full_model(self, x, I1, R1, Sigma1, D1, Omega1, I2, R2, Sigma2, D2, Omega2, I3, R3, Sigma3, D3, Omega3, I4, R4, Sigma4, D4, Omega4,
                        I_res, cbg, gauss_points, lookup):
        # for normal call use self.num_of_points_gauss, for fitting use less!
        # todo could make remaining self params here also static
        # t1_start = timeit.default_timer()
        resolution_peak, F1, R1Mean, F2, R2Mean, F3, R3Mean, F4, R4Mean, S1, S2, S3, S4 = self._calc_model_components(
            x, gauss_points, self.Br, self.Nu, self.grid, self.bessel_grid_eval,
            self.checkBox_NORM_FF.isChecked(), R1, Sigma1, self.checkBox_SPHERES1.isChecked(), R2, Sigma2,
            self.checkBox_SPHERES2.isChecked(), R3, Sigma3, self.checkBox_SPHERES3.isChecked(), R4, Sigma4, self.checkBox_SPHERES4.isChecked(), D1, Omega1, D2, Omega2,
            D3, Omega3, D4, Omega4, lookup)

        resolution = I_res * resolution_peak

        return I1 * F1 * S1 + I2 * F2 * S2 + I3 * F3 * S3 + I4 * F4 * S4 + resolution + cbg

    def calc_components_from_gui(self):
        self.set_parameters_from_gui()
        # update x and chi_grid
        self.create_x_chi_grid()
        normalize = self.checkBox_NORM_FF.isChecked()

        [self.resolution_peak, self.Formfactor1, self.R1Mean, self.Formfactor2, self.R2Mean, self.Formfactor3,
         self.R3Mean, self.Formfactor4, self.R4Mean, self.Structure_Factor1, self.Structure_Factor2, self.Structure_Factor3, self.Structure_Factor4
         ] = self._calc_model_components(
            self.x, self.num_of_points_gauss, self.Br, self.Nu, self.grid,
            self.bessel_grid_eval, normalize, self.R1, self.Sigma1, self.checkBox_SPHERES1.isChecked(), self.R2,
            self.Sigma2, self.checkBox_SPHERES2.isChecked(), self.R3, self.Sigma3, self.checkBox_SPHERES3.isChecked(),
            self.R4, self.Sigma4, self.checkBox_SPHERES4.isChecked(),
            self.Dc1, self.Omega1, self.Dc2, self.Omega2, self.Dc3, self.Omega3, self.Dc4, self.Omega4, self.checkBox_lookup.isChecked()
        )

    def get_fit_from_components(self):

        self.fit = ((self.Int_resolution * 10 ** self.Int_resolution_exp * self.resolution_peak) +
                    (self.Int1 * 10 ** self.Int1_exp * self.Formfactor1 * self.Structure_Factor1) +
                    (self.Int2 * 10 ** self.Int2_exp * self.Formfactor2 * self.Structure_Factor2) +
                    (self.Int3 * 10 ** self.Int3_exp * self.Formfactor3 * self.Structure_Factor3) +
                    (self.Int4 * 10 ** self.Int4_exp * self.Formfactor4 * self.Structure_Factor4))
        self.fit += self.Background

    def calc_and_plot(self):

        self.calc_components_from_gui()
        self.get_fit_from_components()

        self.plot_graph()

    def plot_graph(self, save_pdf=False):
        """Does the full work plotting workd"""
        print('setting up new plot ...')

        if not isinstance(self.fig, plt.Figure):
            self.fig = plt.figure('Gisaxs-Fit', figsize=(10, 8))
            self.ax = self.fig.add_subplot(111)

        self.ax.cla()

        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlabel('q [nm$^{-1}$]', fontsize=20)
        self.ax.set_ylabel('Int. [a.u.]', fontsize=20)
        self.ax.tick_params(axis='both', which='both', direction='in', labelsize=18, bottom=True, top=True,
                            left=True, right=True)

        self.ax.set_title(os.path.basename(self.filename_data), fontsize=20)
        self.fig.tight_layout(w_pad=0.05)

        # from here the ax content is changed but figure stays same

        if self.data is not None:
            self.get_chisquared()

            # todo test errorbar data
            if not self.read_errors:
                yerr = np.sqrt(10 ** self.data[:, 1])
                xerr = np.zeros(shape=self.data[:, 0].shape)
            else:
                yerr = 10 ** self.data[:, 2]
                xerr = 10 ** self.data[:, 3]

            markers, caps, bars = self.ax.errorbar(x=10 ** self.data[:, 0], y=10 ** self.data[:, 1], xerr=xerr,
                                                   alpha=0.7,
                                                   yerr=yerr, fmt='ok', mfc='gray',
                                                   label=r'$\chi^2$ = {0:.4e},'.format(self.chi_squared) +
                                                         r' $\frac{{\chi^2_{{new}}}}{{\chi^2_{{old}}}}$  = {0:.4e}'.format(
                                                             self.chi_squared / self.chi_squared_old))

            # loop through bars and caps and set the alpha value
            [bar.set_alpha(0.3) for bar in bars]

            self.chi_squared_old = self.chi_squared

            self.ax.set_ylim(
                [max(1, 0.5 * 10 ** self.data[:, 1].min()), 1.5 * 10 ** self.data[:, 1].max()])  # adjust lower lim
            # todo (re)-set here also x limit -> as soon as data exists it makes up the self.x grid , use it!
            # todo good limits, what about plt.margins?
            # self.ax.set_xlim([self.xmin, self.xmax])
            # self.ax.set_xlim([0.0016367, 1.99])

        # plot model
        self._plot_model_lines()

        if self.checkBox_LEGEND.isChecked():
            # prop: change font size
            location = str(self.comboBox_LEGEND_POS.itemText(
                self.comboBox_LEGEND_POS.currentIndex()))
            self.leg = self.ax.legend(loc=location, prop={'size': 11})
            # leg.get_frame().set_facecolor('white')
            self.leg.get_frame().set_linewidth(0.0)
            self.leg.get_frame().set_alpha(0.9)  # make it nearly opaque
            self.leg.get_frame().set_facecolor('aliceblue')
        else:
            leg = self.ax.get_legend()
            if leg is not None:
                self.ax.get_legend().remove()

        self.fig.canvas.draw()

        print('Mean Radii:\nR1 = {0:.2f} nm\n'.format(self.R1Mean) +
              'R2 = {0:.2f} nm\n'.format(self.R2Mean) +
              'R3 = {0:.2f} nm\n'.format(self.R3Mean) +
              'R4 = {0:.2f} nm'.format(self.R4Mean))

        if save_pdf is True:
            pdf_name = self.filename_data.rsplit('.', 1)[0] + '.pdf'
            self.fig.savefig(pdf_name, dpi=300, bbox_inches='tight')
            print('PDF-file saved as {0}'.format(pdf_name))
        self.fig.show()
        print('plotting done')

    def _plot_model_lines(self):

        try:
            self.vlineleft = self.ax.vlines(x=10 ** self.data[int(self.ROI_MIN), 0], ymin=1,
                                            ymax=10 ** self.data[int(self.ROI_MIN), 1], linestyle='--')
            self.vlineright = self.ax.vlines(x=10 ** self.data[min(self.data[:, 0].size - 1,
                                                                   int(self.ROI_MAX)), 0],
                                             ymin=1, ymax=10 ** self.data[min(self.data[:, 1].size - 1,
                                                                              int(self.ROI_MAX)), 1],
                                             linestyle='--')
        except TypeError:
            pass  # if no data defined yet

        self.line_res, = self.ax.plot(self.x,
                                      self.Int_resolution * 10 ** self.Int_resolution_exp * self.resolution_peak,
                                      '--g',
                                      label=r'Res. Func.: w = {0}, I$_R$ = {1}e{2}, $\nu$ = {3}'.format(
                                          round(self.Br, 4),
                                          round(self.Int_resolution, 1),
                                          self.Int_resolution_exp,
                                          round(self.Nu, 2)))
        self.line_ff1, = self.ax.plot(self.x, self.Formfactor1 * self.Int1 * 10 ** self.Int1_exp, color='blue',
                                      label=r'FF: $<R_1>$ = {0} nm, $\sigma_1$ = {1} %, I$_1$ = {2}e{3}'.format(
                                          round(self.R1Mean, 1),
                                          round(100 * self.Sigma1, 0),
                                          round(self.Int1, 1), self.Int1_exp))
        self.line_ff2, = self.ax.plot(self.x, self.Formfactor2 * self.Int2 * 10 ** self.Int2_exp, color='orange',
                                      label=r'FF: $<R_2>$ = {0} nm, $\sigma_2$ = {1} %, I$_2$ = {2}e{3}'
                                            ''.format(round(self.R2Mean, 1), round(100 * self.Sigma2, 0),
                                                      round(self.Int2, 1),
                                                      self.Int2_exp))
        self.line_ff3, = self.ax.plot(self.x, self.Formfactor3 * self.Int3 * 10 ** self.Int3_exp, color='magenta',
                                      label=r'FF: $<R_3>$ = {0} nm, $\sigma_3$ = {1} %, I$_3$ = {2}e{3}'.format(
                                          round(self.R3Mean, 1),
                                          round(100 * self.Sigma3, 0),
                                          round(self.Int3, 1), self.Int3_exp))
        self.line_ff4, = self.ax.plot(self.x, self.Formfactor4 * self.Int4 * 10 ** self.Int4_exp, color='purple',
                                      label=r'FF: $<R_4>$ = {0} nm, $\sigma_4$ = {1} %, I$_4$ = {2}e{3}'.format(
                                          round(self.R4Mean, 1),
                                          round(100 * self.Sigma4, 0),
                                          round(self.Int4, 1), self.Int4_exp))
        self.line_sf1, = self.ax.plot(self.x, 10 * self.Structure_Factor1, '--', color='blue',
                                      label=r'Distance: Dc1 = {0} nm, $\Omega_1$ = {1} %'.format(round(self.Dc1, 1),
                                                                                                 round(
                                                                                                     100 * self.Omega1,
                                                                                                     0)))
        self.line_sf2, = self.ax.plot(self.x, 10 * self.Structure_Factor2, '--', color='orange',
                                      label=r'Distance: Dc2 = {0} nm, $\Omega_2$ = {1} %'.format(round(self.Dc2, 1),
                                                                                                 round(
                                                                                                     100 * self.Omega2,
                                                                                                     0)))
        self.line_sf3, = self.ax.plot(self.x, 10 * self.Structure_Factor3, '--', color='magenta',
                                      label=r'Distance: Dc3 = {0} nm, $\Omega_3$ = {1} %'.format(round(self.Dc3, 2),
                                                                                                 round(
                                                                                                     100 * self.Omega3,
                                                                                                     0)))
        self.line_sf4, = self.ax.plot(self.x, 10 * self.Structure_Factor4, '--', color='purple',
                                      label=r'Distance: Dc4 = {0} nm, $\Omega_4$ = {1} %'.format(round(self.Dc4, 2),
                                                                                                 round(
                                                                                                     100 * self.Omega4,
                                                                                                     0)))
        self.line_bg, = self.ax.plot(self.x, np.full(self.x.shape, self.Background), '--', color='grey',
                                     label=r'Background = {} '.format(self.Background))
        self.line_fit, = self.ax.plot(self.x, self.fit, 'r', zorder=9, lw=3, alpha=0.7)

    def _update_model_lines(self):

        try:
            y = 1
            xl = 10 ** self.data[int(self.ROI_MIN), 0]
            ylmax = 10 ** self.data[int(self.ROI_MIN), 1]
            self.vlineleft.set_segments([np.array([[xl, y], [xl, ylmax]]), np.array([[xl, y], [xl, ylmax]])])

            xr = 10 ** self.data[min(self.data[:, 0].size - 1, int(self.ROI_MAX)), 0]
            yrmax = 10 ** self.data[min(self.data[:, 1].size - 1, int(self.ROI_MAX)), 1]
            self.vlineright.set_segments([np.array([[xr, y], [xr, yrmax]]), np.array([[xr, y], [xr, yrmax]])])

            self.ax.draw_artist(self.vlineleft)
            self.ax.draw_artist(self.vlineright)

            # update also chi-sq label
            self.get_chisquared()
            if self.leg is not None:
                lab = r'$\chi^2$ = {0:.4e},'.format(
                    self.chi_squared) + r' $\frac{{\chi^2_{{new}}}} {{\chi^2_{{old}}}}$  = {0:.4e}'.format(
                    self.chi_squared / self.chi_squared_old)
                self.leg.texts[8].set_text(lab)

        except (AttributeError, TypeError):
            pass  # if no data defined yet

        self.line_res.set_data(self.x, self.Int_resolution * 10 ** self.Int_resolution_exp * self.resolution_peak)
        self.ax.draw_artist(self.line_res)

        self.line_ff1.set_data(self.x, self.Formfactor1 * self.Int1 * 10 ** self.Int1_exp)
        self.ax.draw_artist(self.line_ff1)

        self.line_ff2.set_data(self.x, self.Formfactor2 * self.Int2 * 10 ** self.Int2_exp)
        self.ax.draw_artist(self.line_ff2)

        self.line_ff3.set_data(self.x, self.Formfactor3 * self.Int3 * 10 ** self.Int3_exp)
        self.ax.draw_artist(self.line_ff3)

        self.line_ff4.set_data(self.x, self.Formfactor4 * self.Int4 * 10 ** self.Int4_exp)
        self.ax.draw_artist(self.line_ff4)

        self.line_sf1.set_data(self.x, 10 * self.Structure_Factor1)
        self.ax.draw_artist(self.line_ff1)

        self.line_sf2.set_data(self.x, 10 * self.Structure_Factor2)
        self.ax.draw_artist(self.line_sf2)

        self.line_sf3.set_data(self.x, 10 * self.Structure_Factor3)
        self.ax.draw_artist(self.line_sf3)

        self.line_sf4.set_data(self.x, 10 * self.Structure_Factor4)
        self.ax.draw_artist(self.line_sf4)

        self.line_bg.set_data(self.x, np.full(self.x.shape, self.Background))
        self.ax.draw_artist(self.line_bg)

        self.line_fit.set_data(self.x, self.fit)
        self.ax.draw_artist(self.line_fit)

        # update legend
        if self.leg is not None:
            self.leg.texts[0].set_text(
                r'Res. Func.: w = {}, I$_R$ = {}e{}, $\nu$ = {}'.format(
                    round(self.Br, 4), round(self.Int_resolution, 1), self.Int_resolution_exp, round(self.Nu), 2)
            )
            self.leg.texts[1].set_text(
                r'FF: $<R_1>$ = {0} nm, $\sigma_1$ = {1} %, I$_1$ = {2}e{3}'.format(
                    round(self.R1Mean, 1),
                    round(100 * self.Sigma1, 0),
                    round(self.Int1, 1), self.Int1_exp)
            )
            self.leg.texts[2].set_text(
                r'FF: $<R_2>$ = {0} nm, $\sigma_2$ = {1} %, I$_2$ = {2}e{3}'
                ''.format(round(self.R2Mean, 1), round(100 * self.Sigma2, 0), round(self.Int2, 1),
                          self.Int2_exp)
            )
            self.leg.texts[3].set_text(
                r'FF: $<R_3>$ = {0} nm, $\sigma_3$ = {1} %, I$_3$ = {2}e{3}'.format(round(self.R3Mean, 1),
                                                                                    round(100 * self.Sigma3, 0),
                                                                                    round(self.Int3, 1), self.Int3_exp)
            )
            self.leg.texts[4].set_text(
                r'FF: $<R_4>$ = {0} nm, $\sigma_4$ = {1} %, I$_4$ = {2}e{3}'.format(round(self.R4Mean, 1),
                                                                                    round(100 * self.Sigma4, 0),
                                                                                    round(self.Int4, 1), self.Int4_exp)
            )
            self.leg.texts[5].set_text(
                r'Distance: Dc1 = {0} nm, $\Omega_1$ = {1} %'.format(round(self.Dc1, 1),
                                                                     round(
                                                                         100 * self.Omega1,
                                                                         0))
            )
            self.leg.texts[6].set_text(
                r'Distance: Dc2 = {0} nm, $\Omega_2$ = {1} %'.format(round(self.Dc2, 1),
                                                                     round(100 * self.Omega2,
                                                                           0))
            )

            self.leg.texts[7].set_text(
                r'Distance: Dc3 = {0} nm, $\Omega_3$ = {1} %'.format(round(self.Dc3, 2),
                                                                     round(
                                                                         100 * self.Omega3,
                                                                         0))
            )

            self.leg.texts[8].set_text(
                r'Distance: Dc4 = {0} nm, $\Omega_4$ = {1} %'.format(round(self.Dc4, 2),
                                                                     round(
                                                                         100 * self.Omega4,
                                                                         0))
            )
            self.leg.texts[9].set_text(r'Background = {} '.format(self.Background))

    def save_distribution(self):

        if self.checkBox_NORM_FF.isChecked():
            print("Form factors are normalized, and this we won't save for you.. please read a theory book first.")
            return None

        bool_sum = self.checkBox_SPHERES1.isChecked() + self.checkBox_SPHERES2.isChecked() + self.checkBox_SPHERES3.isChecked() + self.checkBox_SPHERES4.isChecked()
        if not (bool_sum == 0 or bool_sum == 4):
            print("Not possible for mixed cylindrical and spherical form factors.")
            return None
        if self.checkBox_SPHERES1.isChecked():
            form = 'spherical'
        else:
            form = 'cylindrical'

        # get currently written parameter name from gui and kidnap that name for saving here
        parameter_file_path = str(self.lineEdit_PARAMETER.text())
        body = parameter_file_path.rstrip('par').rstrip('.')

        if parameter_file_path == '':  # if that does not work fall back to data path
            parameter_file_path = str(self.lineEdit_INPUT_DATA.text())
            body = parameter_file_path.rstrip('dat').rstrip('.')

        save_name = body + '_distribution.txt'  # use txt that files have different extensions from .dat .par .gen
        save_name_values = body + '_values.txt'

        def calc_dist(x, Rm, Int, Int_exp, Sigma):
            return Int * 10 ** Int_exp * np.exp(-((x - Rm) ** 2.) / (2. * (Sigma * Rm) ** 2.)) # newer version
            # return Int * 10 ** Int_exp * np.exp(-((x - Rm) ** 2.) / (2. * (Sigma * Rm) ** 2.))*Rm**2 # could

        R_dist = np.arange(0, 1000, 0.1)  # older -which correct?
        N = calc_dist(R_dist, self.R1Mean, self.Int1, self.Int1_exp, self.Sigma1) + \
            calc_dist(R_dist, self.R2Mean, self.Int2, self.Int2_exp, self.Sigma2) + \
            calc_dist(R_dist, self.R3Mean, self.Int3, self.Int3_exp, self.Sigma3) + \
            calc_dist(R_dist, self.R4Mean, self.Int4, self.Int4_exp, self.Sigma4)

        file = open(save_name, 'w')
        file.write(f'# Using {form} form factors.\n')
        file.write('# R' + '\t' + 'Intensity' + '\n' + '# nm' + '\t' + '[a.u.]' + '\n')
        for i in range(len(R_dist)):
            file.write(str(np.round(R_dist[i], 1)) + '\t' + str(np.round(N[i], 3)) + '\n')
        file.close()

        file = open(save_name_values, 'w')
        file.write(f'# Using {form} form factors.\n')
        file.write('# R' + '\t' + 'Intensity' + '\t' + 'Sigma' + '\t' + 'Distance' + '\n'
                   '# nm' + '\t' + '[a.u.]' + '\t' + '...' + '\t' + 'nm' + '\n')
        file.write(
            str(np.round(self.R1Mean, 2)) + '\t' + str(np.round(self.Int1 * 10 ** self.Int1_exp)) + '\t' + str(
                self.Sigma1) + '\t' + str(np.round(self.Dc1, 2)) + '\n' +
            str(np.round(self.R2Mean, 2)) + '\t' + str(np.round(self.Int2 * 10 ** self.Int2_exp)) + '\t' + str(
                self.Sigma2) + '\t' + str(np.round(self.Dc2, 2)) + '\n' +
            str(np.round(self.R3Mean, 2)) + '\t' + str(np.round(self.Int3 * 10 ** self.Int3_exp)) + '\t' + str(
                self.Sigma3) + '\t' + str(np.round(self.Dc3, 2)) + '\n' +
            str(np.round(self.R4Mean, 2)) + '\t' + str(np.round(self.Int4 * 10 ** self.Int4_exp)) + '\t' + str(
                self.Sigma4) + '\t' + str(np.round(self.Dc4, 2)) + '\n')
        file.close()
        print("saved {} and {}.".format(save_name, save_name_values))

    def _evaluate_bessel_on_grid(self, mini=0, maxi=600, step=0.0001):
        self.mini, self.maxi, self.step = mini, maxi, step
        self.grid = np.arange(mini, maxi, step)
        self.bessel_grid_eval = bessel(1, self.grid)

    @staticmethod
    def _lookup_bessel_from_grid(bessel_grid_eval, grid, array: np.ndarray):
        # we want to rescale the array values that they become ints for accessing the bessel table via indices
        # the array is where we want to evaluate the bessel
        mini, maxi, sizi = grid[0], grid[-1], grid.size
        ind_array = (np.round((array - mini) / (maxi - mini) * sizi)).astype(np.int32)
        return bessel_grid_eval[ind_array]

    def save_png_and_export_fit(self):
        t1_start = timeit.default_timer()

        png_name = self.parameter_file_path.rsplit('.', 1)[0] + '.png'
        try:
            self.fig.savefig(png_name, dpi=300, bbox_inches='tight')
        except AttributeError:
            print("Cannot save None type - create figure first!")
        print('PNG-file saved as {0}'.format(png_name))
        fit_name = self.parameter_file_path.rsplit('.', 1)[0] + '_fit.gen'
        with open(fit_name, 'w') as f:
            for i in range(int(self.num_of_points_graph)):
                # np.log10(self.x.. and self.fit...) deleted to reduce calculations in origin
                f.write(str((self.x[i])) + '\t' +
                        str((self.fit[i])) + '\n')
            f.close()
        print('fitting curve saved as {0}'.format(fit_name))
        t1_stop = timeit.default_timer()
        # print('elapsed time save png:', t1_stop - t1_start, 's')

    def get_chisquared(self):

        # todo test if this is stable now
        # chi points has len like self.data and gives indices within len of eval data to select the grid data points
        # that are closest data point to the actual cut data self.cut

        # points has length <= len of chi points in roi which is equal self.data[:,0] in roi
        try:
            points = np.array(self.chi_points[int(self.ROI_MIN):int(self.ROI_MAX)], dtype='uint16')
            data = 10 ** self.data[int(self.ROI_MIN):int(self.ROI_MAX), 1]
            fit = self.fit[points]
            self.chi_squared = ((fit - data) ** 2).sum() / points.size

            # print('Chi^2 in ROI = {0}'.format(self.chi_squared))
        except ValueError:
            # might occur if fit is plotted, then data loaded, then plotted again, cause self.x changed!
            print('Chi^2 could not be determined.')

    @staticmethod
    def convert_to_log(number):
        # get a and b of: number = a*10**b
        try:
            b = int(np.log10(number))
            a = 10 ** (np.log10(number) - int(np.log10(number)))
            f_ = [a, b]
            return f_
        except OverflowError:
            return [0, 0]

    def fit_leastsq(self):
        if self.data is None:
            print("Not data specified yet - cannot fit")
            return None

        self.set_parameters_from_gui()

        xfit = 10 ** self.data[int(self.ROI_MIN):int(self.ROI_MAX), 0]
        yfit = 10 ** self.data[int(self.ROI_MIN):int(self.ROI_MAX), 1]

        self._setup_lmfit_model()

        t1_start = timeit.default_timer()

        if self.read_errors:
            yweights = 10 ** -self.data[int(self.ROI_MIN):int(self.ROI_MAX), 2]
        else:
            yweights = 1. / np.sqrt(yfit)

        self.fit_result = self.mod.fit(yfit, self.pars, method='least_squares', x=xfit, weights=yweights,
                                       max_nfev=10000,)
        print(self.fit_result.fit_report())

        print("Reduced Chisquare after/b4: {}\nReduced Chisquare absolute:\t{:3.3e}".format(
            self.fit_result.redchi / self.old_red_chisqr, self.fit_result.redchi))
        self.old_red_chisqr = self.fit_result.redchi

        t1_stop = timeit.default_timer()
        print('Finished fitting! It took:', round(t1_stop - t1_start, 3), 's')

        if self.checkBox_real_time.isChecked():
            self.fit_pars()  # emulate what happens if one clicks insert fit parameter to update plot

        if self.checkBox_PLOT_FIT.isChecked():
            self.plot_fit(xfit, self.mod.eval(self.pars, x=xfit))

    def fit_emcee(self):
        if self.fit_result is None:
            print("No fit result available - run classical fit first.")
            return None
        try:
            import emcee
        except ImportError:
            print("Install emcee version 3 package first.\nAlso recommended to install tqdm package for progress bar."
                  "\nAlso recommended to install pandas package for chain analysis.\nAlso install corner package for"
                  "beautiful corner plots.")
            return None

        print("This is highly experimental - use with caution.")
        print("This shall help to get errors on fit parameters. This is not for fitting. "
              "Make sure your fit is perfect before you do this.")

        xfit = 10 ** self.data[int(self.ROI_MIN):int(self.ROI_MAX), 0]
        yfit = 10 ** self.data[int(self.ROI_MIN):int(self.ROI_MAX), 1]

        if self.read_errors:
            yweights = 10 ** -self.data[int(self.ROI_MIN):int(self.ROI_MAX), 2]
        else:
            yweights = 1. / np.sqrt(yfit)

        emcee_kwargs = {'nwalkers': 3*self.fit_result.nvarys, 'steps': 2500, 'burn': 500}
        self.result_emcee = self.mod.fit(yfit, self.pars, method='emcee', x=xfit, weights= yweights,
                                         fit_kws=emcee_kwargs)

        p_varys = list(self.result_emcee.flatchain.columns)
        truths = [self.result_emcee.params.valuesdict()[el] for el in p_varys]

        # if self.checkBox_PLOT_FIT.isChecked():
        import corner
        emcee_corner = corner.corner(self.result_emcee.flatchain, labels=p_varys, truths=truths)
        plt.show()

        print(self.result_emcee.fit_report())

        highest_prob = np.argmax(self.result_emcee.lnprob)
        hp_loc = np.unravel_index(highest_prob, self.result_emcee.lnprob.shape)
        mle_soln = self.result_emcee.chain[hp_loc]
        for i, par in enumerate(p_varys):
            self.result_emcee.params[par].value = mle_soln[i]

        print('\nComparing leastsq with emcee   ')
        print('-------------------------------------------------')
        print('Parameter   ClassicFit  MedianValue   Uncertainty')
        fmt = '{:10s}    {:8.2f}     {:8.2f}      {:8.2f}'.format
        for name in p_varys:
            print(fmt(name, self.fit_result.params[name].value, self.result_emcee.params[name].value,
                      self.result_emcee.params[name].stderr))

        # todo why is it different median above and below?
        print('\nError estimates from emcee:')
        print('-----------------------------------------------------------------')
        print('Parameter      -2sigma    -1sigma     median    +1sigma    +2sigma')

        for name in p_varys:
            quantiles = np.percentile(self.result_emcee.flatchain[name],
                                      [2.275, 15.865, 50, 84.135, 97.275])
            median = quantiles[2]
            err_m2 = quantiles[0] - median
            err_m1 = quantiles[1] - median
            err_p1 = quantiles[3] - median
            err_p2 = quantiles[4] - median
            fmt = '  {:6s}   {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:12.4f}'.format
            print(fmt(name, err_m2, err_m1, median, err_p1, err_p2))

        # todo save important results as txt
        # todo how get clean contours emcee? get local trajectory distribution

    def _setup_lmfit_model(self):

        self.mod = lmfit.Model(self.calc_full_model, independent_vars=["x"],
                               param_names=("I1", "R1", "Sigma1", "D1", "Omega1",
                                            "I2", "R2", "Sigma2", "D2", "Omega2",
                                            "I3", "R3", "Sigma3", "D3", "Omega3",
                                            "I4", "R4", "Sigma4", "D4", "Omega4",
                                            "I_res", "cbg", "gauss_points", "lookup"))
        # general points
        self.mod.set_param_hint('gauss_points', value=self.num_of_points_gauss, vary=False)
        self.mod.set_param_hint('lookup', value=self.checkBox_lookup.isChecked(), vary=False)

        # note: the np isclose fixes the respective parameter if the fit limits are close enough together

        # create the remaining parameters with boolean decision if they should be varied and limits from above
        # boolean decision explanation:
        # - structure factor parameters should not fitted if Dc is set to 0, fix also omega in that case
        # - fit values only if min-max are substantially different, if they are not np.isclose
        # - fit component only if intensity is non-zero
        # - add a tiny value to min limits to avoid that lmfit throws error for min=max (happens independent of
        # vary=False)
        form1 = bool(self.Int1)
        strc1 = bool(self.Dc1)
        self.mod.set_param_hint('I1', min=0., max=1.e12, value=self.Int1 * 10 ** self.Int1_exp, vary=form1)
        self.mod.set_param_hint('R1', min=self.R1_min + 1e-8, max=self.R1_max, value=self.R1,
                                vary=form1 & ~np.isclose(self.R1_min, self.R1_max))
        self.mod.set_param_hint('Sigma1', min=self.Sigma1_min + 1e-8, max=self.Sigma1_max, value=self.Sigma1,
                                vary=form1 & ~np.isclose(self.Sigma1_min, self.Sigma1_max))
        #self.mod.set_param_hint('D1', min=min(self.D1_min + 1e-8, self.Dc1), max=self.D1_max, value=self.Dc1,
        #                        vary=form1 & strc1 & ~np.isclose(self.D1_min, self.D1_max))
        self.mod.set_param_hint('D1', min=2*self.R1 + 1e-8, max=self.D1_max, value=self.Dc1,
                                vary=form1 & strc1 & ~np.isclose(self.D1_min, self.D1_max))
        self.mod.set_param_hint('Omega1', min=self.Omega1_min + 1e-8, max=self.Omega1_max, value=self.Omega1,
                                vary=form1 & strc1 & ~np.isclose(self.Omega1_min, self.Omega1_max))

        form2 = bool(self.Int2)
        strc2 = bool(self.Dc2)
        self.mod.set_param_hint('I2', min=0., max=1.e12, value=self.Int2 * 10 ** self.Int2_exp, vary=form2)
        self.mod.set_param_hint('R2', min=self.R2_min + 1e-8, max=self.R2_max, value=self.R2,
                                vary=form2 & ~np.isclose(self.R2_min, self.R2_max))
        self.mod.set_param_hint('Sigma2', min=self.Sigma2_min + 1e-8, max=self.Sigma2_max, value=self.Sigma2,
                                vary=form2 & ~np.isclose(self.Sigma2_min, self.Sigma2_max))
        #self.mod.set_param_hint('D2', min=min(self.D2_min + 1e-8, self.Dc2), max=self.D2_max, value=self.Dc2,
        #                        vary=form2 & strc2 & ~np.isclose(self.D2_min, self.D2_max))
        self.mod.set_param_hint('D2', min=2*self.R2 + 1e-8, max=self.D2_max, value=self.Dc2,
                                vary=form2 & strc2 & ~np.isclose(self.D2_min, self.D2_max))
        self.mod.set_param_hint('Omega2', min=self.Omega2_min + 1e-8, max=self.Omega2_max, value=self.Omega2,
                                vary=form2 & strc2 & ~np.isclose(self.Omega2_min, self.Omega2_max))

        form3 = bool(self.Int3)
        strc3 = bool(self.Dc3)
        self.mod.set_param_hint('I3', min=0., max=1.e12, value=self.Int3 * 10 ** self.Int3_exp, vary=form3)
        self.mod.set_param_hint('R3', min=self.R3_min + 1e-8, max=self.R3_max, value=self.R3,
                                vary=form3 & ~np.isclose(self.R3_min, self.R3_max))
        self.mod.set_param_hint('Sigma3', min=self.Sigma3_min + 1e-8, max=self.Sigma3_max, value=self.Sigma3,
                                vary=form3 & ~np.isclose(self.Sigma3_min, self.Sigma3_max))
        #self.mod.set_param_hint('D3', min=min(self.D3_min + 1e-8, self.Dc3), max=self.D3_max, value=self.Dc3,
        #                        vary=form3 & strc3 & ~np.isclose(self.D3_min, self.D3_max))
        self.mod.set_param_hint('D3', min=2*self.R3 + 1e-8, max=self.D3_max, value=self.Dc3,
                                vary=form3 & strc3 & ~np.isclose(self.D3_min, self.D3_max))
        self.mod.set_param_hint('Omega3', min=self.Omega3_min + 1e-8, max=self.Omega3_max, value=self.Omega3,
                                vary=form3 & strc3 & ~np.isclose(self.Omega3_min, self.Omega3_max))

        form4 = bool(self.Int4)
        strc4 = bool(self.Dc4)
        self.mod.set_param_hint('I4', min=0., max=1.e12, value=self.Int4 * 10 ** self.Int4_exp, vary=form4)
        self.mod.set_param_hint('R4', min=self.R4_min + 1e-8, max=self.R4_max, value=self.R4,
                                vary=form4 & ~np.isclose(self.R4_min, self.R4_max))
        self.mod.set_param_hint('Sigma4', min=self.Sigma4_min + 1e-8, max=self.Sigma4_max, value=self.Sigma4,
                                vary=form4 & ~np.isclose(self.Sigma4_min, self.Sigma4_max))
        #self.mod.set_param_hint('D3', min=min(self.D3_min + 1e-8, self.Dc3), max=self.D3_max, value=self.Dc3,
        #                        vary=form3 & strc3 & ~np.isclose(self.D3_min, self.D3_max))
        self.mod.set_param_hint('D4', min=2*self.R4 + 1e-8, max=self.D4_max, value=self.Dc4,
                                vary=form4 & strc4 & ~np.isclose(self.D4_min, self.D4_max))
        self.mod.set_param_hint('Omega4', min=self.Omega4_min + 1e-8, max=self.Omega4_max, value=self.Omega4,
                                vary=form4 & strc4 & ~np.isclose(self.Omega4_min, self.Omega4_max))

        self.mod.set_param_hint('cbg', min=1, max=1000, value=self.Background,
                                vary=self.checkBox_FIT_background.isChecked())
        self.mod.set_param_hint('I_res', min=0, max=1e12, value=self.Int_resolution * 10 ** self.Int_resolution_exp,
                                vary=self.checkBox_fit_resolution.isChecked())

        self.pars = self.mod.make_params()
        print(self.pars.pretty_print())

    def plot_fit(self, xfit, yfit):

        # delete axes on fig2 if existing
        if isinstance(self.fig2, plt.Figure):
            self.fig2.clf()

        # for matplotlib 3.5.1, figure assignments changed
        self.fig2 = self.fit_result.plot(fig=self.fig2)  # creates fig2 if not None and plots on it
        self.fig2.canvas.manager.set_window_title('fitting window')
        self.fig2.set_size_inches(8, 8)
        ax1, ax2 = self.fig2.get_axes()
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.get_lines()[0].set_zorder(0)  # put data to the ground
        ax2.get_lines()[1].set_zorder(9)  # put fit to the top
        ax2.plot(xfit, yfit, label="init", color='green')  # plot init above data below fit
        ax2.legend()
        ax1.set_title('residual and fit to data')

        # self.fig2.show()

    def fit_pars(self):

        if self.fit_result is None:
            print("No fit parameters existing yet")
            return None

        self.Background = self.fit_result.params['cbg'].value
        self.doubleSpinBox_BACKGROUND.setValue(self.Background)
        self.Int_resolution = (self.convert_to_log(self.fit_result.params['I_res'].value))[0]
        self.doubleSpinBox_RESOLUTION_INT.setValue(self.Int_resolution)
        self.Int_resolution_exp = (self.convert_to_log(self.fit_result.params['I_res'].value))[1]
        self.doubleSpinBox_RESOLUTION_INT_EXP.setValue(self.Int_resolution_exp)
        self.Dc1 = self.fit_result.params['D1'].value
        self.doubleSpinBox_DC1.setValue(self.Dc1)
        self.Dc2 = self.fit_result.params['D2'].value
        self.doubleSpinBox_DC2.setValue(self.Dc2)
        self.Dc3 = self.fit_result.params['D3'].value
        self.doubleSpinBox_DC3.setValue(self.Dc3)
        self.Dc4 = self.fit_result.params['D4'].value
        self.doubleSpinBox_DC4.setValue(self.Dc4)
        self.Omega1 = self.fit_result.params['Omega1'].value
        self.doubleSpinBox_OMEGA1.setValue(self.Omega1)
        self.Omega2 = self.fit_result.params['Omega2'].value
        self.doubleSpinBox_OMEGA2.setValue(self.Omega2)
        self.Omega3 = self.fit_result.params['Omega3'].value
        self.doubleSpinBox_OMEGA3.setValue(self.Omega3)
        self.Omega4 = self.fit_result.params['Omega4'].value
        self.doubleSpinBox_OMEGA4.setValue(self.Omega4)
        self.R1 = self.fit_result.params['R1'].value
        self.doubleSpinBox_R1.setValue(self.R1)
        self.R2 = self.fit_result.params['R2'].value
        self.doubleSpinBox_R2.setValue(self.R2)
        self.R3 = self.fit_result.params['R3'].value
        self.doubleSpinBox_R3.setValue(self.R3)
        self.R4 = self.fit_result.params['R4'].value
        self.doubleSpinBox_R4.setValue(self.R4)
        self.Sigma1 = self.fit_result.params['Sigma1'].value
        self.doubleSpinBox_SIGMA1.setValue(self.Sigma1)
        self.Sigma2 = self.fit_result.params['Sigma2'].value
        self.doubleSpinBox_SIGMA2.setValue(self.Sigma2)
        self.Sigma3 = self.fit_result.params['Sigma3'].value
        self.doubleSpinBox_SIGMA3.setValue(self.Sigma3)
        self.Sigma4 = self.fit_result.params['Sigma4'].value
        self.doubleSpinBox_SIGMA4.setValue(self.Sigma4)
        self.Int1 = (self.convert_to_log(self.fit_result.params['I1'].value))[0]
        self.doubleSpinBox_INT1.setValue(self.Int1)
        self.Int1_exp = (self.convert_to_log(self.fit_result.params['I1'].value))[1]
        self.doubleSpinBox_INT1_EXP.setValue(self.Int1_exp)
        self.Int2 = (self.convert_to_log(self.fit_result.params['I2'].value))[0]
        self.doubleSpinBox_INT2.setValue(self.Int2)
        self.Int2_exp = (self.convert_to_log(self.fit_result.params['I2'].value))[1]
        self.doubleSpinBox_INT2_EXP.setValue(self.Int2_exp)
        self.Int3 = (self.convert_to_log(self.fit_result.params['I3'].value))[0]
        self.doubleSpinBox_INT3.setValue(self.Int3)
        self.Int3_exp = (self.convert_to_log(self.fit_result.params['I3'].value))[1]
        self.doubleSpinBox_INT3_EXP.setValue(self.Int3_exp)
        self.Int4 = (self.convert_to_log(self.fit_result.params['I4'].value))[0]
        self.doubleSpinBox_INT4.setValue(self.Int4)
        self.Int4_exp = (self.convert_to_log(self.fit_result.params['I4'].value))[1]
        self.doubleSpinBox_INT4_EXP.setValue(self.Int4_exp)

    @staticmethod
    def _update_plot_settings():

        plt.rcParams['figure.raise_window'] = False
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['xtick.major.width'] = 1.5
        plt.rcParams['ytick.major.width'] = 1.1  #1.5
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.minor.width'] = 0.001     #1

    def closeEvent(self, event):
        """
        catches closeEvent which is sent by calling self.close
        by pressing EXIT or red X
        """
        self.save_temp()  # saving the latest parameter for safety
        print("Thanks for using our software. Cheers!")
        plt.close()  # close plot window
        event.accept()  # exit application

    def save_temp(self):
        """save parameters in temporary file if they changed"""

        self.set_parameters_from_gui()  # update parameter variables
        self.parameters = self.setup_parameter_array()  # update parameter array

        # get current parameter_file_path
        parameter_file_path = str(self.lineEdit_PARAMETER.text())

        if parameter_file_path == '':  # if that does not work fall back to data path
            parameter_file_path = str(self.lineEdit_INPUT_DATA.text())

        body = parameter_file_path.split('.')[0]

        save_name = body + '_parameters.temp'

        f = open(save_name, 'w')
        f.write("# parameters:\n" +
                "# num_of_points_graph, num_of_points_gauss, Br, Nu, Int_resolution, Int_resolution_exp\n" +
                "# Dc1, Omega1, R1, Sigma1, Int1, Int1_exp\n" +
                "# Dc2, Omega2, R2, Sigma2, Int2, Int2_exp\n" +
                "# Dc3, Omega3, R3, Sigma3, Int3, int3_exp\n" +
                "# Dc4, Omega4, R4, Sigma4, Int4, int4_exp\n" +
                "# Background, 0, 0, 0, ROI_MIN, ROI_MAX\n" +
                "# \n")

        for a in self.parameters:
            f.write(str(a[0]) + '\t' + str(a[1]) + '\t' + str(a[2]) + '\t' +
                    str(a[3]) + '\t' + str(a[4]) + '\t' + str(a[5]) + str(a[6]) + '\n')

        f.close()
        print('Backup: saved latest parameters to {0}'.format(save_name))


def Main():
    print('Welcome to Gisaxs-Fit!')
    print('This is version {0}'.format(__VERSION__))

    app = PyQt5.QtWidgets.QApplication(sys.argv)
    my_app = GisaxsModeler()
    my_app.show()
    # just to be sure it's on top
    my_app.activateWindow()
    my_app.raise_()

    ret = app.exec_()
    sys.exit(ret)


if __name__ == "__main__":
    Main()
