import os
import argparse
import torch
import time
from faster_whisper import WhisperModel
from moviepy import VideoFileClip
from tqdm import tqdm

def extract_audio(video_path, audio_path):
    """从视频中提取音频"""
    try:
        # 检查文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
        print(f"正在打开视频文件: {video_path}")
        
        # 直接使用ffmpeg提取音频，速度更快
        try:
            import subprocess
            print("使用ffmpeg提取音频...")
            
            # 使用tqdm创建进度条
            with tqdm(total=100, desc="提取音频", unit="%") as pbar:
                # 优化ffmpeg参数，提高提取速度
                # 增加线程数和优先级，使用更快的编码方式
                cmd = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 -threads 8 -priority high "{audio_path}"'
                
                # 使用subprocess.Popen获取实时输出以更新进度条
                process = subprocess.Popen(
                    cmd, 
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # 模拟进度更新
                for i in range(100):
                    time.sleep(0.01)  # 小延迟避免CPU占用过高
                    pbar.update(1)
                    
                    # 检查进程是否已完成
                    if process.poll() is not None:
                        pbar.update(100 - pbar.n)  # 完成剩余进度
                        break
                
                # 等待进程完成
                process.wait()
                
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                print("使用ffmpeg提取音频成功")
                return audio_path
            else:
                raise Exception("ffmpeg提取音频失败")
        except Exception as e:
            print(f"使用ffmpeg提取音频失败: {str(e)}")
            
            # 回退到MoviePy方法
            print("尝试使用MoviePy提取音频...")
            video = VideoFileClip(video_path, verbose=False)
            
            if video.audio is None:
                raise ValueError("视频文件没有音轨")
                
            print(f"正在提取音频到: {audio_path}")
            
            # 创建进度条
            progress_bar = tqdm(total=100, desc="提取音频", unit="%")
            
            # 定义进度回调函数
            def progress_callback(progress):
                # 更新进度条
                progress_bar.update(int(progress * 100) - progress_bar.n)
            
            # 使用进度回调提取音频，优化参数
            video.audio.write_audiofile(
                audio_path, 
                verbose=False, 
                logger=None,
                progress_bar=progress_callback,
                buffersize=2000,  # 增加缓冲区大小
                nbytes=2          # 减少精度以提高速度
            )
            
            # 确保进度条到达100%
            progress_bar.update(100 - progress_bar.n)
            progress_bar.close()
            
            video.close()  # 确保关闭视频文件
            return audio_path
    except Exception as e:
        print(f"提取音频时出错: {str(e)}")
        raise

def transcribe_audio(audio_path, model_size="medium", language=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """使用faster-whisper提取字幕"""
    print(f"使用 {model_size} 模型在 {device} 上进行提取字幕...")
    
    # 加载模型
    model = WhisperModel(model_size, device=device, compute_type="int8")

    print(f"准备提取字幕...")    
    
    # 检查是否支持进度回调
    supports_progress_callback = True
    try:
        # 尝试获取transcribe方法的参数列表
        import inspect
        transcribe_params = inspect.signature(model.transcribe).parameters
        supports_progress_callback = 'progress_callback' in transcribe_params
    except:
        supports_progress_callback = False
    
    if supports_progress_callback:
        # 创建进度条
        progress_bar = tqdm(total=100, desc="提取字幕进度", unit="%")
        last_progress = [0]  # 使用列表存储上一次进度，以便在回调中修改
        
        # 定义进度回调函数
        def progress_callback(progress):
            # 计算增量更新，避免进度条回退
            increment = max(0, int(progress * 100) - last_progress[0])
            if increment > 0:
                progress_bar.update(increment)
                last_progress[0] = int(progress * 100)
        
        try:
            # 转录
            segments, info = model.transcribe(
                audio_path, 
                language=language, 
                beam_size=5, 
                vad_filter=True,
                progress_callback=progress_callback
            )
            
            # 确保进度条到达100%
            progress_bar.update(100 - progress_bar.n)
        finally:
            progress_bar.close()
    else:
        # 如果不支持进度回调，直接使用无进度条模式
        # 创建一个手动更新的进度条来显示经过的时间
        progress_bar = tqdm(total=0, desc="提取字幕中", bar_format='{desc}: {elapsed}')
        
        # 启动一个线程来更新进度条
        import threading
        stop_event = threading.Event()
        
        def update_progress():
            while not stop_event.is_set():
                progress_bar.update(0)  # 强制更新显示
                time.sleep(1)  # 每秒更新一次
        
        # 启动更新线程
        update_thread = threading.Thread(target=update_progress)
        update_thread.daemon = True
        update_thread.start()
        
        try:
            segments, info = model.transcribe(
                audio_path, 
                language=language, 
                beam_size=5, 
                vad_filter=True
            )
        finally:
            # 停止更新线程
            stop_event.set()
            if update_thread.is_alive():
                update_thread.join(timeout=1)
            progress_bar.close()
    
    # 将segments转换为列表以便使用tqdm显示进度
    segments_list = list(segments)
    
    print(f"检测到的语言: {info.language} (概率: {info.language_probability:.2f})")
    
    return segments_list, info

def write_srt(segments, output_path):
    """将转录结果写入SRT文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        # 添加进度条
        for i, segment in enumerate(tqdm(segments, desc="生成字幕", unit="段"), start=1):
            # 格式化时间戳
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            
            # 写入SRT格式
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment.text.strip()}\n\n")
    
    return output_path

def format_timestamp(seconds):
    """将秒数格式化为SRT时间戳格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def main():
    parser = argparse.ArgumentParser(description="从视频中提取语音并生成字幕")
    parser.add_argument("video_path", help="视频文件路径")
    parser.add_argument("--model", default="large-v3-turbo", choices=["tiny", "base", "small", "medium", "large", "large-v3-turbo"], help="Whisper模型大小")
    parser.add_argument("--language", default=None, help="视频语言代码 (如 'zh' 表示中文, 'en' 表示英文, 默认自动检测)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="使用的设备")
    parser.add_argument("--no-keep-audio", action="store_false", dest="keep_audio", help="不保留提取的音频文件")
    parser.add_argument("--no-extract-audio", action="store_true", default=False, help="不提取音频文件，直接从视频中提取字幕")
    
    args = parser.parse_args()
    
    # 获取视频文件信息
    video_path = os.path.abspath(args.video_path)
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 设置临时音频文件和输出字幕文件路径
    temp_audio_path = os.path.join(video_dir, f"{video_name}_temp.wav")
    output_srt_path = os.path.join(video_dir, f"{video_name}.srt")
    
    # 初始化时间统计变量
    audio_extraction_time = 0
    transcription_time = 0
    
    # 添加这段代码来抑制MoviePy的输出
    import sys
    from contextlib import contextmanager
    
    @contextmanager
    def suppress_output():
        """临时抑制标准输出和标准错误"""
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    
    try:
        print(f"处理视频: {video_path}")
        
        # 根据选项决定是否提取音频
        if args.no_extract_audio:
            print("跳过音频提取步骤，直接从视频中提取字幕...")
            audio_source = video_path
        else:
            # 检查是否已存在提取好的音频文件
            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                print(f"发现已存在的音频文件...")
                print("跳过音频提取步骤...")
            else:
                # 提取音频并计时
                print("正在从视频中提取音频...")
                start_time = time.time()
                
                # 使用上下文管理器抑制输出
                with suppress_output():
                    extract_audio(video_path, temp_audio_path)
                
                audio_extraction_time = time.time() - start_time
                print(f"音频提取完成，用时: {audio_extraction_time:.2f} 秒")
            audio_source = temp_audio_path
        
        # 转录音频并计时
        print("开始提取字幕...")
        start_time = time.time()
        
        segments, info = transcribe_audio(
            audio_source, 
            model_size=args.model,
            language=args.language,
            device=args.device
        )
        transcription_time = time.time() - start_time
        
        # 写入SRT文件
        print("正在生成字幕文件...")
        write_srt(segments, output_srt_path)
        
        print(f"字幕生成完成: {output_srt_path}")
        
        # 输出时间统计汇总
        print("\n===== 处理时间统计 =====")
        if not args.no_extract_audio and audio_extraction_time > 0:
            print(f"音频提取用时: {audio_extraction_time:.2f} 秒")
        print(f"字幕提取用时: {transcription_time:.2f} 秒")
        total_time = audio_extraction_time + transcription_time
        print(f"总处理用时: {total_time:.2f} 秒")
        if transcription_time > 0:
            video_duration = 0
            try:
                # 使用suppress_output上下文管理器屏蔽VideoFileClip的输出
                with suppress_output():
                    with VideoFileClip(video_path) as video:
                        video_duration = video.duration
                if video_duration > 0:
                    print(f"视频长度: {video_duration:.2f} 秒")
                    print(f"处理速度: {video_duration/total_time:.2f}x 实时速度")
            except:
                pass
        
    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        # 只有在用户不想保留音频文件时才删除，且不是直接从视频提取字幕的情况
        if not args.no_extract_audio and not args.keep_audio and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"已删除临时音频文件")
        elif not args.no_extract_audio and args.keep_audio and os.path.exists(temp_audio_path):
            print(f"保留临时音频文件: {temp_audio_path}")

if __name__ == "__main__":
    main()