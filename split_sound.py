from scipy.io import wavfile
import os
import sys
import numpy as np
from tqdm import tqdm  # 进度条
import subprocess as sp

WIN_LENGH = 2
RATE = 44100


def energy(samples):
    x = np.sum(np.power(samples, 2.)) / float(len(samples))

    return x


# 转换数据范围 浮点型[-1, 1] 整型[-max, max]
def normalization(y, out_type):
    # 无数据时只改变数据类型
    if y.size == 0:
        return y.astpye(out_type)

    peak = np.abs(y).max()  # 最大幅值
    if peak != 0:  # 非零数据才能归一化
        y = y / peak

    if issubclass(out_type, np.integer):  # 整数不能缩放到[-1, 1]
        y *= np.iinfo(out_type).max
        y = y.astype(out_type)  # 改变数据类型时不可直接用dtype

    return y


# load_audio can not detect the input type
def ffmpeg_load_audio(filename, sr=44100, mono=False, in_type=np.int16, out_type=np.float32):
    in_type = np.dtype(in_type).type  # numpy有多种类型表示方法 进行统一
    out_type = np.dtype(out_type).type
    channels = 1 if mono else 2

    format_strings = {
        np.float64: 'f64le',
        np.float32: 'f32le',
        np.int16: 's16le',
        np.int32: 's32le',
        np.uint32: 'u32le'
    }
    format_string = format_strings[in_type]
    command = [
        'ffmpeg',
        '-i', filename,
        '-f', format_string,
        '-acodec', 'pcm_' + format_string,
        '-ar', str(sr),
        '-ac', str(channels),
        '-']
    p = sp.Popen(command, stdout=sp.PIPE, bufsize=4096, close_fds=True)  # 调用ffmpeg工具解码获取所有音频数据

    raw = b''
    while True:
        data = p.stdout.read()
        if data:
            raw += data
        else:
            break
    audio = np.frombuffer(raw, dtype=in_type)  # 数据转为矩阵格式
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()

    audio = normalization(audio, out_type)  # 归一化到[-1, 1]区间

    return audio, sr


def save_as_wav(file_name, data_list):
    data = np.array(data_list)  # 转换为np格式
    data = normalization(data, np.int16)  # 归一化
    data = np.dstack([data, data])[0]  # 扩展到双通道
    wavfile.write(file_name, RATE, data)


def split_wav(input_filename):
    window_duration = WIN_LENGH  # 窗口值至少不小于step值
    step_duration = 2  # 检测步长
    # silence_threshold = 0.0012  # args.silence_threshold
    silence_threshold = 0.002  # args.silence_threshold

    output_dir = "debug"  # 需要先建立文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename_prefix = os.path.splitext(os.path.basename(input_filename))[0]  # 取出文件名
    dry_run = False  # args.dry_run

    print("Splitting {} where energy is below {}% for longer than {}s.".format(
        input_filename,
        silence_threshold,  # * 100.,
        window_duration
    ))

    # Read and split the file
    # sample_rate, samples = input_data = wavfile.read(filename=input_filename, mmap=True)
    # sample_rate, samples = input_data = wavfile.read(filename=input_filename, mmap=True)

    sr = 44100
    samples, sample_rate = ffmpeg_load_audio(input_filename, sr, mono=True)

    print("sample rate:", sample_rate, " len:", len(samples), " secs:", 1.0 * len(samples) / sample_rate)

    window_size = int(window_duration * sample_rate)
    step_size = int(step_duration * sample_rate)
    print("win size,step size:", window_size, step_size)

    energy_list = []
    pos_list = []
    step_no = 0
    pre_energy = 0

    for i_start in tqdm(range(0, len(samples), step_size)):  # 使用进度条
        i_end = i_start + window_size
        if i_end >= len(samples):
            break
        step_no += 1
        # print "win:",win_no, i_start, " - ", i_end

        window_energy = energy(samples[i_start:i_end])
        change = 100 * pre_energy / window_energy

        if change < 90 or change > 110:
            print("detected change: %.1f" % change)
            pre_energy = window_energy
            data = samples[i_start:i_end]
            output_file_path = "{}_{:03d}.wav".format(os.path.join(output_dir, output_filename_prefix), step_no)
            save_as_wav(output_file_path, data)  # 生成wav文件片段
        else:
            print("%.1f," % change)
        energy_list.append(window_energy)
        pos_list.append([i_start, i_end])


def main(argv):
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "output.wav"
    split_wav(filename)


if __name__ == "__main__":
    main(sys.argv)
