import os
import shutil
import argparse

def parse_args():
    """
    解析命令行参数，获取数据集目录路径。
    :return: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="Process the Tiny ImageNet.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory.')
    return parser.parse_args()

def reorganize_images(base_dir):
    """
    重组数据集目录结构，将每个类别文件夹中的`images`文件夹内的图片移到类别文件夹中，并删除空的`images`文件夹。
    :param base_dir: 数据集的根目录路径
    """
    # 遍历base_dir下的所有类别文件夹
    for class_dir_name in os.listdir(base_dir):
        class_dir_path = os.path.join(base_dir, class_dir_name)
        
        # 确保这是一个目录
        if os.path.isdir(class_dir_path):
            images_dir_path = os.path.join(class_dir_path, "images")
            
            # 检查images文件夹是否存在
            if os.path.exists(images_dir_path) and os.path.isdir(images_dir_path):
                # 移动images文件夹内的所有文件到类别文件夹
                for image_name in os.listdir(images_dir_path):
                    src_path = os.path.join(images_dir_path, image_name)
                    dst_path = os.path.join(class_dir_path, image_name)
                    shutil.move(src_path, dst_path)
                
                # 删除空的images文件夹
                os.rmdir(images_dir_path)
                
                # 删除类别文件夹中的文本文件（如果存在）
                txt_file = os.path.join(class_dir_path, f"{class_dir_name}_boxes.txt")
                if os.path.exists(txt_file):
                    os.remove(txt_file)
            else:
                print(f"No 'images' folder found in {class_dir_path}")
        else:
            print(f"{class_dir_path} is not a directory")

def main():
    """
    主函数，解析命令行参数并调用reorganize_images函数重组数据集目录结构。
    """
    args = parse_args()
    base_directory = args.data_dir
    reorganize_images(base_directory)

if __name__ == "__main__":
    main()
