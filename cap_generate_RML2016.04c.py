from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from PyQt5 import Qt
from PyQt5 import Qt, QtCore
from gnuradio import blocks
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import qtgui
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import numpy
import sip
import sys
from gnuradio import qtgui



#!/usr/bin/env python
from cap_transmitters import transmitters
from cap_source_alphabet import source_alphabet
from test import random_pskmod_constel
#import timeseries_slicer
#import analyze_stats
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft, cPickle, gzip
import math
'''
Generate dataset with dynamic channel model across range of SNRs
'''

apply_channel = True
output = {}
min_length = 9e9
snr_vals = range(12,13,1)
#for i in range(10):
for alphabet_type in transmitters.keys():
    print(alphabet_type)
    for i,mod_type in enumerate(transmitters[alphabet_type]):
        print("running test", i,mod_type)
        print("mod_type_name : ", mod_type.modname)
        tx_len = int(10e3)
        src = source_alphabet(alphabet_type, tx_len, True)
        mod = mod_type()

        snk = blocks.vector_sink_c()
        tb = gr.top_block()
        # connect blocks
        tb.connect(src, mod)

        random_pskmod_constel(mod)
        tb.run()


        print("test2")
 
