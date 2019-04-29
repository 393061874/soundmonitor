import numpy as np
import matplotlib.pyplot as plt
import wave
import pyaudio


def read_csv_file(file):
    """
    从文件中获取数据集
    x, y = read_csv_file('data_set.csv')
    :param file: 文件名
    :return: (data, label)
    """
    import pandas

    data = pandas.read_csv(file, index_col=0).values  # 读取csv数据

    return data


def save_csv_file(data, file):
    """
    保存数据集 格式为label, data
    :param data: 数据集list-like
    :param file: 目标文件名
    :return: 无
    """
    import pandas

    data_set = pandas.DataFrame(data)
    data_set.to_csv(file)


def read_audio_file(file_path):
    """
    解析音频文件
    :param file_path: 文件路径
    :return: 单通道音频数据, 采样率
    """
    from pydub import AudioSegment  # 使用ffmpeg比librosa要快得多

    sound = AudioSegment.from_file(file_path, format=file_path.split('.')[-1])  # 读取音频文件
    sound = sound.set_channels(1).set_sample_width(2)  # 设置为单通道
    audio = np.frombuffer(sound.raw_data, dtype=np.int16)  # 转换为numpy数据
    audio = audio.astype(np.float) / np.iinfo(np.int16).max

    return audio, sound.frame_rate


def save_audio_file(file_name, audio, sampling_rate):
    """
    音频数据存储为文件
    :param file_name: 文件路径
    :param audio: 单通道音频数据 np.float32
    :param sampling_rate: 采样率
    :return: 无
    """
    from pydub import AudioSegment

    audio = audio * np.iinfo(np.int16).max
    audio = audio.astype(np.int16)

    audio = np.dstack([audio, audio])[0]  # 扩展到双通道
    sound = AudioSegment(data=audio.flatten(), sample_width=2, frame_rate=sampling_rate, channels=2)  # 创建流
    sound.export(file_name, format=file_name.split('.')[-1])  # 保存为文件


def get_mfcc(audio_data, sampling_rage, mfcc_row):
    """
    获取音频mfcc特征矩阵
    :param audio_data: 音频数据 一维np.float32类型
    :param sampling_rage: 采样率
    :param mfcc_row: 特征矩阵行数
    :return: 特征矩阵
    """
    from librosa.feature import mfcc

    mfcc_data = mfcc(audio_data, sampling_rage, n_mfcc=mfcc_row)
    mfcc_data = np.array(mfcc_data)
    detal_1 = np.diff(mfcc_data)
    detal_2 = np.diff(detal_1)
    mfcc_feature = np.hstack([mfcc_data, detal_1, detal_2])

    # from python_speech_features import mfcc
    # mfcc_data = mfcc(audio_data, sampling_rage)
    # detal_1 = np.diff(mfcc_data)
    # detal_2 = np.diff(detal_1)
    # mfcc_feature = np.hstack([mfcc_data, detal_1, detal_2])

    return mfcc_feature


def get_audio_amp(file):
    """
    从音频文件获取特征值
    :param file: 传入音频文件
    :return: 返回[32, 64]特征值矩阵
    """
    import librosa
    import numpy as np
    from skimage.measure import block_reduce

    # 转为频域
    N_FFT = 1024
    data, _ = librosa.load(file)  # 读取音频文件
    fft = librosa.stft(data, n_fft=N_FFT, hop_length=N_FFT // 4, window=np.hanning(N_FFT))
    amp = np.abs(fft)

    # 统一维度
    amp = block_reduce(amp, (10, 1), func=np.mean)  # 每10行取平均降维
    if amp.shape[1] < 64:
        amp = np.pad(amp, ((0, 0), (0, 64 - amp.shape[1])), 'constant')
    amp = amp[:32, :64]  # 截取前[32, 64]矩阵

    # 归一化
    amp = normalization(amp, np.float32)  # 归一化到[0, 1]

    return amp


def to_same_length(data, column):
    """
    不同长度的音频获取的特征值长度也不同 这里统一长度
    :param data: 特征矩阵
    :param column: 目标长度
    :return: 特征矩阵
    """
    from scipy.interpolate import interp1d

    row = data.shape[0]
    length = data.shape[1]
    new_data = []
    x_new = np.linspace(0, length - 1, column).tolist()
    for i in range(row):
        f = interp1d(range(length), data[i], kind='cubic')
        new_data.append(f(x_new))

    return np.array(new_data)


def get_tsne(data, dim):
    """
    TSNE 降维
    :param data: 原始数据 np.array
    :param dim: 目标维度
    :return: 降维后数据
    """
    from sklearn.manifold import TSNE

    data = [feature.flatten() for feature in data]  # 扁平化处理
    tsne = TSNE(
        n_components=dim,  # 目标维度
        perplexity=30.0,  # 困惑度
        early_exaggeration=12.0,
        learning_rate=200.0,  # 学习率
        n_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric="euclidean",
        init="random",
        verbose=0,
        random_state=None,
        method='barnes_hut',
        angle=0.5
    )
    x = tsne.fit_transform(data)  # 进行降维

    return x


def show_1d_plot(data):
    """
    绘制折线图
    :param data: 数据
    :return: 无
    """
    plt.plot(data)
    plt.show()


def show_2d_picture(data):
    """
    绘制图片
    :param data: 数据
    :return: 无
    """
    plt.imshow(data)
    plt.show()


def show_2d_scatter(data, label):
    """
    绘制二维散点图
    :param data: 坐标
    :param label: 标签
    :return: 无
    """
    from sklearn.preprocessing import LabelEncoder

    # # 标签编码
    # label_encoder = LabelEncoder()  # 标签编码
    # label_encoder.fit(label)  # 使用标签管理器对标签编码
    # y = label_encoder.transform(label)

    # 按标签不同分成不同点集
    data_set = {}
    for point, lab in zip(data, label):
        if lab not in data_set:
            data_set[lab] = [point]
        else:
            data_set[lab].append(point)

    # 绘图
    scatter_list = []
    lab_list = []
    for lab, data_list in data_set.items():
        data_list = np.array(data_list)
        a = plt.scatter(data_list[:, 0], data_list[:, 1])  # 画点
        scatter_list.append(a)
        lab_list.append(lab)

    plt.legend(scatter_list, lab_list)
    plt.show()


def show_3d_scatter(data, label):
    """
    绘制三维散点图
    :param data: 坐标
    :param label: 标签
    :return: 无
    """
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.preprocessing import LabelEncoder

    # # 标签编码
    # label_encoder = LabelEncoder()  # 标签编码
    # label_encoder.fit(label)  # 使用标签管理器对标签编码
    # y = label_encoder.transform(label)

    # 按标签不同分成不同点集
    data_set = {}
    for point, lab in zip(data, label):
        if lab not in data_set:
            data_set[lab] = [point]
        else:
            data_set[lab].append(point)

    # 绘图
    scatter_list = []
    lab_list = []
    fig = plt.figure()
    ax = Axes3D(fig)
    for lab, data_list in data_set.items():
        data_list = np.array(data_list)
        a = ax.scatter(data_list[:, 0], data_list[:, 1], data_list[:, 2])
        scatter_list.append(a)
        lab_list.append(lab)

    plt.legend(scatter_list, lab_list)
    plt.show()


def play_audio(audio_data, sampling_rate):
    """
    播放音频数据 播放时有一个前缀噪音 原因未知
    :param audio_data: 音频数据
    :param sampling_rate: 采样率
    :return: 无
    """

    # 处理音频数据
    audio_data = np.array(audio_data)
    audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)

    # 打开音频流
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(2),
               channels=1,
               rate=sampling_rate,
               output=True)

    # 进行播放
    stream.write(audio_data)

    # 关闭资源
    stream.stop_stream()
    stream.close()
    p.terminate()


def play_wav_file(file_path):
    """
    播放音频文件 播放时有一个前缀噪音 原因未知
    :param file_path: 文件路径
    :return: 无
    """

    # 打开文件
    file = wave.open(file_path, "rb")

    # 打开音频流
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(file.getsampwidth()),
                    channels=file.getnchannels(),
                    rate=file.getframerate(),
                    output=True)

    # 读取文件并播放
    data = file.readframes(1024)
    while data != b'':
        stream.write(data)
        data = file.readframes(1024)

    # 关闭资源
    stream.stop_stream()
    stream.close()
    p.terminate()


def get_audio_from_mic(seconds, sampling_rate=16000):
    """
    录音
    :param seconds: 录音秒数
    :param sampling_rate: 采样率
    :return: 录音内容 np数组[-1, 1]
    """

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
               channels=1,
               rate=sampling_rate,
               input=True,
               frames_per_buffer=1024)
    print("* recording len(sec):", seconds)

    data = stream.read(int(16000 * seconds))
    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio = np.frombuffer(data, dtype=np.int16)  # 转换为numpy数据
    audio = audio.astype(np.float) / np.iinfo(np.int16).max

    return audio
