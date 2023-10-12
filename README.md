# TeLeS_WLC

Use Kaldi alignment steps in https://www.eleanorchodroff.com/tutorial/kaldi/forced-alignment.html to generate ctm of words in the reference.

Prepare the dataset in this format:

audio_file_name, reference, start_time_of_words (separated by space), end_time_of_words (separated by space)

For example,

kb_all/kb_data_clean_m4a/hindi/test/audio/844424932919645-1202-f.m4a	सुधीर मिश्रा की ये फिल्म लेखक शरत चंद्र चट्टोपाध्याय के उपन्यास देवदास पर आधारित होगी	0.53 0.95 1.23 1.43 1.52 1.77 2.26 2.62 3.28 3.87 3.96 4.39 4.79 4.91 5.41	0.95 1.23 1.43 1.52 1.77 2.26 2.62 3.04 3.84 3.96 4.39 4.76 4.91 5.37 5.55
