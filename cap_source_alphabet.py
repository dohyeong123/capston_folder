#!/usr/bin/env python
from gnuradio import gr, blocks
import mediatools
import numpy as np

class source_alphabet(gr.hier_block2):
    def __init__(self, dtype="discrete", limit=10000, randomize=False):
        if(dtype == "discrete"):
            gr.hier_block2.__init__(self, "source_alphabet",
                gr.io_signature(0,0,0),
                gr.io_signature(1,1,gr.sizeof_char))

            ##using gnuradio random block
            self.src = blocks.vector_source_b(map(int, np.random.randint(0, 255, 1000)), True)
            
            self.convert = blocks.packed_to_unpacked_bb(1, gr.GR_LSB_FIRST)
            #self.convert = blocks.packed_to_unpacked_bb(8, gr.GR_LSB_FIRST);
            self.limit = blocks.head(gr.sizeof_char*1, limit)
            self.connect(self.src,self.convert)
            last = self.convert

            # whiten our sequence with a random block scrambler (optionally)
            if(randomize):
                rand_len = 256
                rand_bits = np.random.randint(2, size=rand_len)
                self.randsrc = blocks.vector_source_b(rand_bits, True)
                self.xor = blocks.xor_bb()
                self.connect(self.randsrc,(self.xor,1))
                self.connect(last, self.xor)
                last = self.xor


        # connect head or not, and connect to output
        if(limit==None):
            self.connect(last, self)
        else:
            self.connect(last, self.limit, self)


if __name__ == "__main__":
    print "QA..."

    # Test discrete source
    tb = gr.top_block()
    src = source_alphabet("discrete", 1000)
    snk = blocks.vector_sink_b()
    tb.run()


