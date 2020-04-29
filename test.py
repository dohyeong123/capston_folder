#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Random Pskmod Constel
# Generated: Wed Apr 15 11:18:13 2020
##################################################

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from PyQt5 import Qt, QtCore
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import blocks
from gnuradio import channels
from gnuradio import analog
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import qtgui
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from gnuradio.filter import pfb
from gnuradio import filter
from grc_gnuradio import blks2 as grc_blks2
from optparse import OptionParser
import sip
import sys
from gnuradio import qtgui
from cap_transmitters import transmitters
from cap_source_alphabet import source_alphabet
import numpy as np
import numpy.fft, cPickle, gzip
import math
import start2
import mapper
class random_pskmod_constel(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Random Pskmod Constel")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Random Pskmod Constel")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "random_pskmod_constel")

        if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
            self.restoreGeometry(self.settings.value("geometry").toByteArray())
        else:
            self.restoreGeometry(self.settings.value("geometry", type=QtCore.QByteArray))

        ##################################################
        # Variables
        ##################################################
        self.nfilts = nfilts = 32
        self.ntaps = ntaps = nfilts*11*8
        self.bw = bw = 0.35
        self.samp_rate = samp_rate = 200000

        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(nfilts, nfilts, 1.0, bw, ntaps)

        self.mod = mod = 1
        '''
        for alphabet_type in transmitters.keys():
            for mod_type in enumerate(transmitters[alphabet_type]):   
                if mod_type.modname == "BPSK":
                    print("running test", mod_type)
                    self.mod1=mod_type()
                if mod_type.modname == "QPSK":
                    print("running test", mod_type)
                    self.mod2=mod_type()
        self.QPSK = self.mod2
        self.BPSK = self.mod1
        '''
        self.QPSK = mapper.mapper(mapper.QPSK, ([0,1,3,2]))
        self.BPSK = mapper.mapper(mapper.BPSK, ([0,1]))
        ##################################################
        # Blocks
        ##################################################
        self._mod_options = (0, 1, )
        self._mod_labels = ('BPSK', 'QPSK', )
        self._mod_group_box = Qt.QGroupBox("mod")
        self._mod_box = Qt.QVBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._mod_button_group = variable_chooser_button_group()
        self._mod_group_box.setLayout(self._mod_box)
        for i, label in enumerate(self._mod_labels):
        	radio_button = Qt.QRadioButton(label)
        	self._mod_box.addWidget(radio_button)
        	self._mod_button_group.addButton(radio_button, i)
        self._mod_callback = lambda i: Qt.QMetaObject.invokeMethod(self._mod_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._mod_options.index(i)))
        self._mod_callback(self.mod)
        self._mod_button_group.buttonClicked[int].connect(
        	lambda i: self.set_mod(self._mod_options[i]))
        self.top_grid_layout.addWidget(self._mod_group_box, 0, 0, 1, 1)
        [self.top_grid_layout.setRowStretch(r,1) for r in range(0,1)]
        [self.top_grid_layout.setColumnStretch(c,1) for c in range(0,1)]
        self.selector = grc_blks2.selector(
        	item_size=gr.sizeof_gr_complex*1,
        	num_inputs=2,
        	num_outputs=1,
        	input_index=mod,
        	output_index=0,
        )
        self.qtgui_time_sink_x_1_0 = qtgui.time_sink_c(
        	1024, #size
        	samp_rate, #samp_rate
        	"post_real_data", #name
        	1 #number of inputs
        )
        self.qtgui_time_sink_x_1_0.set_update_time(0.10)
        self.qtgui_time_sink_x_1_0.set_y_axis(-5, 5)

        self.qtgui_time_sink_x_1_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_1_0.enable_tags(-1, True)
        self.qtgui_time_sink_x_1_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1_0.enable_autoscale(False)
        self.qtgui_time_sink_x_1_0.enable_grid(False)
        self.qtgui_time_sink_x_1_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_1_0.enable_control_panel(False)
        self.qtgui_time_sink_x_1_0.enable_stem_plot(False)

        if not True:
          self.qtgui_time_sink_x_1_0.disable_legend()

        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
                  "magenta", "yellow", "dark red", "dark green", "blue"]
        styles = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in xrange(2):
            if len(labels[i]) == 0:
                if(i % 2 == 0):
                    self.qtgui_time_sink_x_1_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_1_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_1_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_0_win = sip.wrapinstance(self.qtgui_time_sink_x_1_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_1_0_win)
        self.qtgui_time_sink_x_1 = qtgui.time_sink_c(
        	1024, #size
        	samp_rate, #samp_rate
        	"pre_real_data", #name
        	1 #number of inputs
        )
        self.qtgui_time_sink_x_1.set_update_time(0.10)
        self.qtgui_time_sink_x_1.set_y_axis(-5, 5)

        self.qtgui_time_sink_x_1.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_1.enable_tags(-1, True)
        self.qtgui_time_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1.enable_autoscale(False)
        self.qtgui_time_sink_x_1.enable_grid(False)
        self.qtgui_time_sink_x_1.enable_axis_labels(True)
        self.qtgui_time_sink_x_1.enable_control_panel(False)
        self.qtgui_time_sink_x_1.enable_stem_plot(False)

        if not True:
          self.qtgui_time_sink_x_1.disable_legend()

        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
                  "magenta", "yellow", "dark red", "dark green", "blue"]
        styles = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in xrange(2):
            if len(labels[i]) == 0:
                if(i % 2 == 0):
                    self.qtgui_time_sink_x_1.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_1.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_win = sip.wrapinstance(self.qtgui_time_sink_x_1.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_1_win)
        self.qtgui_const_sink_x_0_0_0_0 = qtgui.const_sink_c(
        	1024, #size
        	"pre_real_data", #name
        	1 #number of inputs
        )
        self.qtgui_const_sink_x_0_0_0_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0_0_0_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0_0_0_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0_0_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0_0_0.enable_grid(False)
        self.qtgui_const_sink_x_0_0_0_0.enable_axis_labels(True)

        if not True:
          self.qtgui_const_sink_x_0_0_0_0.disable_legend()

        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
                  "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]
        for i in xrange(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_0_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_0_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0_0_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_0_0_0_win)
        self.qtgui_const_sink_x_0_0_0 = qtgui.const_sink_c(
        	1024, #size
        	"post_real_data", #name
        	1 #number of inputs
        )
        self.qtgui_const_sink_x_0_0_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0_0_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0_0_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0_0.enable_grid(False)
        self.qtgui_const_sink_x_0_0_0.enable_axis_labels(True)

        if not True:
          self.qtgui_const_sink_x_0_0_0.disable_legend()

        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
                  "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]
        for i in xrange(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_0_0_win)
        self.pfb_arb_resampler_xxx_0 = filter.pfb_arb_resampler_ccf(8, rrc_taps)
        '''
        self.digital_constellation_modulator_0_0 = digital.generic_mod(
          constellation=QPSK,
          differential=True,
          samples_per_symbol=8,
          pre_diff_code=True,
          excess_bw=0.35,
          verbose=False,
          log=False,
          )
        self.digital_constellation_modulator_0 = digital.generic_mod(
          constellation=BPSK,
          differential=True,
          samples_per_symbol=8,
          pre_diff_code=True,
          excess_bw=0.35,
          verbose=False,
          log=False,
          )
        '''
        self.src= blocks.vector_source_b(map(int, np.random.randint(0, 2, 1000)), True)
        fD = 1
        delays = [0]
        mags = [1]
        #noise_amp = 0
        ntaps = 8
        noise_amp = 0.1  
        self.channels_dynamic_channel_model_0 = channels.dynamic_channel_model( 200e3, 0.01, 1e2, 0.01, 1e3, 8, fD, True, 4, delays, mags, ntaps, noise_amp, 0x1337 )
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, 255, 1000)), True)
        self.snk2 = blocks.file_sink(gr.sizeof_gr_complex*1, '/root/workspace/capston_folder/after_channel.dat', False)
        self.snk2.set_unbuffered(False)
        ##################################################
        # Connections
        ##################################################
        self.connect((self.src), (self.BPSK, 0))
        self.connect((self.src), (self.QPSK, 0))
        self.connect((self.channels_dynamic_channel_model_0, 0), (self.qtgui_const_sink_x_0_0_0, 0))
        self.connect((self.channels_dynamic_channel_model_0, 0), (self.qtgui_time_sink_x_1_0, 0))
        self.connect((self.channels_dynamic_channel_model_0, 0), (self.snk2))
        self.connect((self.BPSK, 0), (self.selector, 0))
        self.connect((self.QPSK, 0), (self.selector, 1))
        self.connect((self.pfb_arb_resampler_xxx_0, 0), (self.channels_dynamic_channel_model_0, 0))
        self.connect((self.pfb_arb_resampler_xxx_0, 0), (self.qtgui_const_sink_x_0_0_0_0, 0))
        self.connect((self.pfb_arb_resampler_xxx_0, 0), (self.qtgui_time_sink_x_1, 0))
        self.connect((self.selector, 0), (self.pfb_arb_resampler_xxx_0, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "top_block")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_nfilts(self):
        return self.nfilts

    def set_nfilts(self, nfilts):
        self.nfilts = nfilts
        self.set_ntaps(self.nfilts*11*8)

    def get_ntaps(self):
        return self.ntaps

    def set_ntaps(self, ntaps):
        self.ntaps = ntaps

    def get_bw(self):
        return self.bw

    def set_bw(self, bw):
        self.bw = bw

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_time_sink_x_1_0.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate)
        self.channels_dynamic_channel_model_0.set_samp_rate(self.samp_rate)

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps
        self.pfb_arb_resampler_xxx_0.set_taps((self.rrc_taps))

    def get_mod(self):
        return self.mod

    def set_mod(self, mod):
        self.mod = mod
        self._mod_callback(self.mod)
        self.selector.set_input_index(int(self.mod))

    def get_QPSK(self):
        return self.QPSK

    def set_QPSK(self, QPSK):
        self.QPSK = QPSK

    def get_BPSK(self):
        return self.BPSK

    def set_BPSK(self, BPSK):
        self.BPSK = BPSK
##################################################
# Main
##################################################
def main(top_block_cls=random_pskmod_constel, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()
    tb.start()
    tb.show()
    
    def quitting():
        tb.stop()
        tb.wait()
    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()
    
    cat = start2.Cat()
    
    ## IQ data를 shaping하는 과정에 있어서 sample_length를 설정한 후 input dataset을 dictionary형태로 만들기 위한 key를 'QPSK'로 설정한다
    cat.generate(128, 'QPSK')
    
    ## test_set 생성하기
    cat.train_test()
    
    ## CNN model 설정
    #cat.cnn_model()
   
    ## Training하기
    #cat.training_session(5,1024)
    
    ## 이미 trainging을 마친 model 불러오기
    cat.load_model('/root/workspace/RF-Signal-Model/weight_4layers.wts.h5')
   
    ## model에 testset에서 batch 만큼 넣어서 정확도 평가하기
    cat.score(1)
   
    ## matrix plot 하기
    cat.plot_matrix()
    
    
if __name__ == '__main__':
    main()
