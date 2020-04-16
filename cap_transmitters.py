#!/usr/bin/env python
import time, math
from scipy.signal import get_window
from gnuradio import gr, blocks, digital, analog, filter
from gnuradio.filter import firdes
import mapper
sps = 8
ebw = 0.35

class transmitter_mapper(gr.hier_block2):
    def __init__(self, modtype, symvals, txname, samples_per_symbol=2, excess_bw=0.35):
        gr.hier_block2.__init__(self, txname,
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.mod = mapper.mapper(modtype, symvals)
        # pulse shaping filter
        nfilts = 32
        ntaps = nfilts * 11 * int(samples_per_symbol)    # make nfilts filters of ntaps each
        rrc_taps = filter.firdes.root_raised_cosine(
            nfilts,          # gain
            nfilts,          # sampling rate based on 32 filters in resampler
            1.0,             # symbol rate
            excess_bw, # excess bandwidth (roll-off factor)
            ntaps)
        self.rrc_filter = filter.pfb_arb_resampler_ccf(samples_per_symbol, rrc_taps)
        self.connect(self, self.mod, self.rrc_filter, self)
        #self.rate = const.bits_per_symbol()

class transmitter_bpsk(transmitter_mapper):
    modname = "BPSK"
    def __init__(self):
        transmitter_mapper.__init__(self, mapper.BPSK,
            [0,1], "transmitter_bpsk", sps, ebw)

class transmitter_qpsk(transmitter_mapper):
    modname = "QPSK"
    def __init__(self):
        transmitter_mapper.__init__(self, mapper.QPSK,
            [0,1,3,2], "transmitter_qpsk", sps, ebw)



transmitters = {
    "discrete":[transmitter_qpsk],
    
    }
