import torch
from glob import glob
from tqdm import tqdm
from random import shuffle, sample, randint, choices
from speechbrain.dataio.dataio import read_audio, write_audio
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.lobes.augment import EnvCorrupt
from functools import reduce
import torchaudio
import numpy as np

MN_PATH = ['/data02/corpus/speech_corpus/mongolian_wav/*.wav',
           '/data01/zhaofei/data/Mongolian/蒙古语电话录音/*/*/*.wav']
EN_PATH = ['/data02/corpus/speech_corpus/TIMIT/TIMIT-wav/*/*/*/*.WAV',
           '/data01/zhaofei/data/enhancement_data_wsj0si84/enhancement_data/clean_si84_train/*.wav']
CN_PATH = ['/data02/corpus/speech_corpus/863/*/*/*/*.WAV']
NN_PATH = ['/data01/zhaofei/data/deep_xi_dataset/train_noise/*.wav']

do_time_augment = TimeDomainSpecAugment(speeds=[80, 110, 120],
                                       perturb_prob=1.0,
                                       drop_freq_prob=1.0,
                                       drop_chunk_prob=1.0,
                                       drop_chunk_length_low=1000,
                                       drop_chunk_length_high=3000)


corrupter = EnvCorrupt(openrir_folder='/data01/fanhaipeng/NewStudent/course-v3/zh-nbs/data2/')

corrupt_wav = lambda x: do_time_augment(x, torch.ones(1))
corrupt_noise = lambda x: corrupter(x, torch.ones(1))

mn_data = reduce(lambda x,y: x+y, [glob(x) for x in MN_PATH])
en_data = reduce(lambda x,y: x+y, [glob(x) for x in EN_PATH])
cn_data = reduce(lambda x,y: x+y, [glob(x) for x in CN_PATH])
nn_data = reduce(lambda x,y: x+y, [glob(x) for x in NN_PATH])

# data = sample(mn_data, 10000) + sample(en_data, 10000) + sample(cn_data, 10000) + sample(nn_data, 10000)
# label = 'M'*10000 + 'E'*10000 + 'C'*10000 + 'O'*10000
data = sample(mn_data, 1000) + sample(en_data, 1000) + sample(cn_data, 1000)
label = 'M'*1000 + 'E'*1000 + 'C'*1000

pack = list(zip(data, label))
shuffle(pack)

lab = ''
for i, p in tqdm(enumerate(pack)):
    w, l = p
    lab += l
    clean = read_audio(w)
    clean = clean / clean.max()
    clean = clean.squeeze().unsqueeze(0)
    pipe = [corrupt_noise, corrupt_wav]
    pipe = choices(pipe, k=randint(0, 3))
    for f in pipe:
        clean = f(clean)
    clean = clean.squeeze()
    clean = clean[randint(0, 3000):]
    clean = clean[:randint(30000, 60000)]
    write_audio('/data01/zhaofei/data/freshmen_practice_data/val/%d.wav'%i, clean.squeeze(), 16000)

with open('/data01/zhaofei/data/freshmen_practice_data/val.txt', 'w') as f:
    f.write(lab)