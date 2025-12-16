import imageio.v2 as imageio
import os
import glob

# 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_dir, 'generated_images')
save_path = os.path.join(base_dir, 'gan_training_process.gif')


filenames = glob.glob(os.path.join(img_dir, 'image_at_epoch_*.png'))
filenames.sort()

print(f">> 총 {len(filenames)}장의 이미지를 찾았습니다.")


images = []
for filename in filenames:
    images.append(imageio.imread(filename))



imageio.mimsave(save_path, images, duration=0.2, loop=0)

print(f">> GIF 저장 완료: {save_path}")