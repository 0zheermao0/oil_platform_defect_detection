import os
import cv2
from typing import List, Optional
from predict import process_images as process_images_by_yolo
from predict import yolo_model
from os.path import dirname
import gradio as gr
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import time


project_name = "zhy_predict"
def get_default_output_path() -> str:
    """
    获取默认输出文件夹路径
    
    Returns:
        str: 默认输出文件夹路径
    """
    current_dir = os.getcwd()
    default_out_path = os.path.join(current_dir, project_name)
    os.makedirs(default_out_path, exist_ok=True)
    return default_out_path


def convert_to_mp4(input_path):
    # 读取 AVI 文件
    clip = VideoFileClip(input_path)

    # 将视频转换为 MP4 格式
    output_path = input_path.replace(".avi", ".mp4")
    print(f"convert input path is {input_path}, output path is {output_path}")
    clip.write_videofile(output_path)
    clip.close()  # 释放资源
    return output_path

def process_video(video_file: str) -> str:
    """
    使用 ultralytics 处理视频文件并保存输出

    Args:
        video_file (str): 上传的视频文件路径

    Returns:
        str: 处理后的视频文件路径
    """
    model = YOLO("./cfg/best.pt")  # 使用适当的模型文件路径

    # 使用模型处理视频
    output_dir = get_default_output_path()
    # result = model(video_file, save=True, project=output_dir, name="output_video", exist_ok=True, show=True)  # 处理并保存
    result = model(video_file, save=True, project=output_dir, name="output_video", exist_ok=True)  # 处理并保存
    print(f"video file name is: {video_file}")
    output_file = os.path.join(output_dir,"output_video",video_file.split('/')[-1].split('.')[0]+'.avi')
    # 将 AVI 文件转换为 MP4
    output_file_new = convert_to_mp4(output_file)

    # print(f"output dir is {output_dir}")
    # print(f"处理后的视频文件保存在: {output_file_new}")
    # cv2.destroyAllWindows()

    return output_file_new  # 返回处理后的视频文件路径

def process_images(input_files: List[str], output_folder: Optional[str] = get_default_output_path()) -> List[str]:
    """
    处理上传的图像文件并保存到输出文件夹

    Args:
        input_files (List[str]): 上传的图像文件路径列表
        output_folder (str, optional): 输出文件夹路径，默认为当前目录下的out文件夹

    Returns:
        List[str]: 处理后的图片路径列表
    """
    # 如果未指定输出文件夹，使用默认路径
    if not output_folder:
        output_folder = get_default_output_path()

    # 创建输出目录（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 存储处理后图片路径的列表
    source_dir = dirname(input_files[0])
    processed_images = process_images_by_yolo(model_path="./cfg/best.pt", source_dir=input_files, output_dir=output_folder)

    return processed_images

def video_stream(rtsp_url: str):
    target_fps = 20
    delay_between_frames = 1.0 / target_fps
    cap = cv2.VideoCapture(rtsp_url)  # 替换为你的 RTSP URL
    if not cap.isOpened():
        raise ValueError("无法打开视频流")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        # 转换 BGR 到 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = yolo_model(frame)[0].plot()
        # cv2.imshow('label', frame)
        # if cv2.waitKey(1) & 0xFF==ord('q'):
        #     break
        yield frame
        elapsed_time = time.time() - start_time
        time.sleep(max(0, delay_between_frames - elapsed_time))
        # time.sleep(0.1)  # 控制帧率

    cap.release()

def create_gradio_interface():
    """
    创建Gradio界面进行图像和视频处理
    """
    with gr.Blocks() as demo:
        # 图像处理选项卡
        with gr.Tab("图像处理"):
            with gr.Row():
                # 使用文件选择器上传图像文件
                input_files = gr.File(
                    label="选择图像文件",
                    file_count="multiple",
                    file_types=["image"]
                )

                # 输出文件夹默认为当前目录下的out，但可以修改
                # output_folder = gr.Textbox(
                #     label="输出文件夹路径",
                #     value=get_default_output_path(),
                #     interactive=True
                # )

            process_btn = gr.Button("处理图像")
            output_gallery = gr.Gallery(label="处理后的图像")

            process_btn.click(
                fn=process_images,
                inputs=[input_files],
                outputs=[output_gallery]
            )

        # 视频流处理选项卡
        with gr.Tab("视频流处理"):
            with gr.Row():
                rtsp_inputs1 = gr.Textbox(
                    label="RTSP流地址1",
                    placeholder="例如: rtsp://username:password@ip:port/stream1, rtsp://username:password@ip:port/stream2"
                )
                stream_btn1 = gr.Button("开始流处理")

            
            video_outputs1 = gr.Image(label="处理后的视频流", streaming=True)

            stream_btn1.click(
                fn=video_stream,
                inputs=[rtsp_inputs1],
                outputs=[video_outputs1],
                queue=True
            )

            with gr.Row():
                rtsp_inputs2 = gr.Textbox(
                    label="RTSP流地址2",
                    placeholder="例如: rtsp://username:password@ip:port/stream1, rtsp://username:password@ip:port/stream2"
                )
                stream_btn2 = gr.Button("开始流处理")

            
            video_outputs2 = gr.Image(label="处理后的视频流", streaming=True)

            stream_btn2.click(
                fn=video_stream,
                inputs=[rtsp_inputs2],
                outputs=[video_outputs2],
                queue=True
            )

        # 本地视频处理选项卡
        with gr.Tab("本地视频处理"):
            video_input = gr.File(label="上传视频文件", type="filepath", file_types=[".mp4"])
            output_video = gr.Video(label="处理后的视频流")  # 使用 Video 组件
            video_input.change(process_video, inputs=video_input, outputs=output_video)
            # video_input.change(process_video, inputs=video_input)

    return demo

def main():
    demo = create_gradio_interface()
    demo.launch()


if __name__ == "__main__":
    main()

