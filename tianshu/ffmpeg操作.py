# """
# ffmpeg 操作流程
#     1、读取输入源
#     2、进行音视频的解封装 Demuxer
#     3、解码每一帧音视频数据 Decoder
#     4、编码每一帧音视频数据 Encoder
#     5、进行音视频的重新封装 Muxer
#     6、输出到目标
# 取两步使用 解封装 解编码 结果为YVU或PCM
# """
#
# # ffprobe -show_streams 00-05-21.mp3  查看文件信息
# """
# wav转pcm:
#     ffmpeg -i input.wav -f s16be -ar 8000 -acodec pcm_s16be output.raw
# pcm转wav:
#     ffmpeg -f s16be -ar 8000 -ac 2 -acodec pcm_s16be -i input.raw output.wav
# """
# def read_audio_file(file_path):
#     """
#     使用 ffmpeg 工具解码音频文件 返回单通道音频数据 默认采样率16000
#     :param file_path: 文件路径
#     :return: 单通道音频数据
#     """
#     import subprocess as sp
#     command = [
#         'ffmpeg',
#         '-i', file_path,
#         '-f', 's16le',
#         '-acodec', 'pcm_s16le',
#         '-ar', str(SAMPLING_RAGE),
#         '-ac', '1',
#         '-']
#     p = sp.Popen(command, stdout=sp.PIPE, bufsize=4096, close_fds=True)  # 调用ffmpeg工具解码获取所有音频数据
#
#     raw = b''
#     while True:
#         data = p.stdout.read()
#         if data:
#             raw += data
#         else:
#             break
#     audio = np.frombuffer(raw, dtype=np.int16)  # 数据转为矩阵格式
#
#     return audio