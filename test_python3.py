# TODO : can be removed later
import os
import cv2
import numpy as np

from CoarseNet import MinutiaeNet_utils, CoarseNet_utils, CoarseNet_model
from FineNet import FineNet_model

print("skap testovacka")

coarse_net = CoarseNet_model.CoarseNetmodel(
    (None, None, 1),
    '/home/jakub/projects/bp/minutiae_classificator/models/CoarseNet.h5', mode='deploy')
fine_net = FineNet_model.get_fine_net_model(
    num_classes=2,
    pretrained_path='/home/jakub/projects/bp/minutiae_classificator/models/FineNet.h5',
    input_shape=(224, 224, 3))


class Minutiae:
    def __init__(self, file_name, minutiae_data):
        self.file_name = file_name
        self.minutiae_data = minutiae_data

    def print_data(self):
        print("name: ", self.file_name, " data: ", self.minutiae_data.shape[0])


def read_image(image_path):
    original_image = np.array(cv2.imread(image_path, 0))
    image_size = np.array(original_image.shape, dtype=np.int32) // 8 * 8
    image = original_image[:image_size[0], :image_size[1]]

    output = dict()

    output['original_image'] = original_image
    output['image_size'] = image_size
    output['image'] = image

    return output


def get_extracted_minutiae(image_folder, as_image=False):
    minutiae_files = []

    for subdir, dirs, files in os.walk(image_folder):
        for file_name in files:
            file_path = image_folder + file_name

            minutiae_data = get_extracted_minutiae_data(file_path, as_image)

            file_name_without_extension = os.path.splitext(os.path.basename(file_name))[0]
            minutiae = Minutiae(file_name_without_extension, minutiae_data)
            minutiae_files.append(minutiae)

    return minutiae_files


def get_extracted_minutiae_data(image_path, as_image=True):
    image = read_image(image_path)

    extracted_minutiae = extract_minutiae(
        image['image'], image['original_image'])

    return extracted_minutiae


def extract_minutiae(image, original_image):
    # Generate OF
    texture_img = MinutiaeNet_utils.fast_enhance_texture(image, sigma=2.5, show=False)
    dir_map, fre_map = MinutiaeNet_utils.get_maps_stft(
        texture_img, patch_size=64, block_size=16, preprocess=True)

    image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

    enh_img, enh_img_imag, enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = coarse_net.predict(
        image)

    # Use for output mask
    round_seg = np.round(np.squeeze(seg_out))
    seg_out = 1 - round_seg
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    seg_out = cv2.dilate(seg_out, kernel)

    # ========== Adaptive threshold ==================
    final_minutiae_score_threashold = 0.45
    early_minutiae_thres = final_minutiae_score_threashold + 0.05

    # In cases of small amount of minutiae given, try adaptive threshold
    while final_minutiae_score_threashold >= 0:
        mnt = CoarseNet_utils.label2mnt(
            np.squeeze(mnt_s_out) *
            np.round(
                np.squeeze(seg_out)),
            mnt_w_out,
            mnt_h_out,
            mnt_o_out,
            thresh=early_minutiae_thres)

        mnt_nms_1 = MinutiaeNet_utils.py_cpu_nms(mnt, 0.5)
        mnt_nms_2 = MinutiaeNet_utils.nms(mnt)
        # Make sure good result is given
        if mnt_nms_1.shape[0] > 4 and mnt_nms_2.shape[0] > 4:
            break
        else:
            final_minutiae_score_threashold = final_minutiae_score_threashold - 0.05
            early_minutiae_thres = early_minutiae_thres - 0.05

    mnt_nms = MinutiaeNet_utils.fuse_nms(mnt_nms_1, mnt_nms_2)

    mnt_nms = mnt_nms[mnt_nms[:, 3] > early_minutiae_thres, :]

    mnt_refined = []
    print("ideme na fine net")
    # ======= Verify using FineNet ============
    for idx_minu in range(mnt_nms.shape[0]):
        try:
            # Extract patch from image
            x_begin = int(mnt_nms[idx_minu, 1]) - 22
            y_begin = int(mnt_nms[idx_minu, 0]) - 22
            patch_minu = original_image[x_begin:x_begin + 2 *
                                        22, y_begin:y_begin + 2 * 22]

            try:
                patch_minu = cv2.resize(patch_minu, dsize=(
                    224, 224), interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                # TODO : add some reasonable code here - programme will fail on next step
                print(str(e))

            ret = np.empty(
                (patch_minu.shape[0], patch_minu.shape[1], 3), dtype=np.uint8)
            ret[:, :, 0] = patch_minu
            ret[:, :, 1] = patch_minu
            ret[:, :, 2] = patch_minu
            patch_minu = ret
            patch_minu = np.expand_dims(patch_minu, axis=0)

            # Use soft decision: merge FineNet score with CoarseNet score
            [is_minutiae_prob] = fine_net.predict(patch_minu)
            is_minutiae_prob = is_minutiae_prob[0]

            tmp_mnt = mnt_nms[idx_minu, :].copy()
            tmp_mnt[3] = (4*tmp_mnt[3] + is_minutiae_prob) / 5
            mnt_refined.append(tmp_mnt)

        except BaseException:
            mnt_refined.append(mnt_nms[idx_minu, :])

    mnt_nms_backup = mnt_nms.copy()
    mnt_nms = np.array(mnt_refined)

    if mnt_nms.shape[0] > 0:
        mnt_nms = mnt_nms[mnt_nms[:, 3] >
                          final_minutiae_score_threashold, :]

    CoarseNet_model.fuse_minu_orientation(dir_map, mnt_nms, mode=3)

    return mnt_nms
