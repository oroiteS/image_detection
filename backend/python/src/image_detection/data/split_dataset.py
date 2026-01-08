import os
import shutil
import random
import yaml

# ================= 配置区域 =================
# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 溯源到 backend/python 目录
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# 原始数据路径 (您的图片和LabelImg生成的txt都在这里)
SOURCE_DIR = os.path.join(BASE_DIR, "datasets", "raw_data")

# 目标数据集路径 (脚本会自动创建这个文件夹)
TARGET_DIR = os.path.join(BASE_DIR, "datasets", "power_inspection")

# 划分比例 (训练集 : 验证集 : 测试集)
SPLIT_RATIO = [0.7, 0.2, 0.1]

# 您的类别名称 (必须与 LabelImg 中的 classes.txt 顺序一致！)
CLASS_NAMES = [
    "missing_insulator",  # 绝缘子缺失
    "burned_insulator",  # 绝缘子烧蚀
    "bird_nest",  # 鸟巢
    "shifted_grading_ring"  # 均压环移位
]


# ===========================================

def split_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 错误: 找不到源目录 '{SOURCE_DIR}'。")
        print(f"💡 请确保目录存在: {SOURCE_DIR}")
        return

    # 1. 准备目标目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(TARGET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, 'labels', split), exist_ok=True)

    print(f"✅ 已创建目标目录结构: {TARGET_DIR}")

    # 2. 获取所有图片文件
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in os.listdir(SOURCE_DIR) if os.path.splitext(f)[1].lower() in supported_extensions]

    if not images:
        print(f"❌ 未在源目录找到图片: {SOURCE_DIR}")
        return

    # 3. 随机打乱
    random.shuffle(images)
    total_count = len(images)
    print(f"📊 共找到 {total_count} 张图片，准备划分...")

    # 4. 计算划分数量
    train_count = int(total_count * SPLIT_RATIO[0])
    val_count = int(total_count * SPLIT_RATIO[1])

    # 5. 移动文件
    for i, image_file in enumerate(images):
        if i < train_count:
            split = 'train'
        elif i < train_count + val_count:
            split = 'val'
        else:
            split = 'test'

        src_image_path = os.path.join(SOURCE_DIR, image_file)
        src_label_path = os.path.join(SOURCE_DIR, os.path.splitext(image_file)[0] + '.txt')

        dst_image_path = os.path.join(TARGET_DIR, 'images', split, image_file)
        dst_label_path = os.path.join(TARGET_DIR, 'labels', split, os.path.splitext(image_file)[0] + '.txt')

        shutil.copy2(src_image_path, dst_image_path)

        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
        else:
            print(f"⚠️ 警告: 图片 {image_file} 没有对应的 .txt 标签文件。")

    print(f"✅ 划分完成: Train={train_count}, Val={val_count}, Test={total_count - train_count - val_count}")

    # 6. 生成 data.yaml
    yaml_content = {
        'path': TARGET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(CLASS_NAMES)}
    }

    yaml_path = os.path.join(TARGET_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"📄 已生成配置文件: {yaml_path}")


if __name__ == '__main__':
    split_dataset()
