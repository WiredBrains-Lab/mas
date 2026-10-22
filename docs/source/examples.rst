Examples
========

PSD and Notch Detection
-----------------------

.. code-block:: python

   import numpy as np
   from dsp.power_spectrum import process_single_channel_notches

   fs = 1_000
   t = np.arange(0, 2.0, 1/fs)
   # 60 Hz interference + noise
   x = 0.2*np.sin(2*np.pi*60*t) + 0.05*np.random.randn(t.size)
   params = {
       'fs': fs,
       'barthannwin': 'bartlett',
       'freq_min': 40, 'freq_max': 120,
       'psd_1_x': 500,
       'span_smooth': 21,
       'db_thr': 6,
       'filtering_params': {'lowcut': 1.0, 'highcut': 300.0, 'order': 4},
   }
   ch_label = 'MEG 0111'
   ch, psd_data, notches, filt = process_single_channel_notches((ch_label, x, params))
   print('Detected notches:', notches['notch_freqs'])
