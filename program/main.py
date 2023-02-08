import utils
import config
import cv2
import os

if __name__ == "__main__":
    fg_path = config.fg_path
    bg_path = config.bg_path
    leave_path = config.leave_path
    signatures_path = config.signatures_path
    save_path = config.SAVE_PATH
    
    # Program_1
    Program_1 = utils.fusion_depth(fg_path, bg_path)
    fusion_image, mask, fg_highpass, bg_highpass = Program_1.hipass()
    cv2.imwrite(os.path.join(save_path, "Fusion_depth.jpg"), fusion_image)
    cv2.imwrite(os.path.join(save_path, "mask.jpg"), 255 * mask)
    cv2.imwrite(os.path.join(save_path, "fg_highpass.jpg"), 3 * fg_highpass)
    cv2.imwrite(os.path.join(save_path, "bg_highpass.jpg"), 3 * bg_highpass)

    # Program_2
    Program_2 = utils.simulate_abnormal_vision(fg_path)
    blue_yellow_blind = Program_2.blue_yellow_blind()
    red_green_blind = Program_2.red_green_blind()
    glaucoma = Program_2.glaucoma(sigma=300)

    cv2.imwrite(os.path.join(save_path, "blue_yellow_blind.jpg"), blue_yellow_blind)
    cv2.imwrite(os.path.join(save_path, "red_green_blind.jpg"), red_green_blind)
    cv2.imwrite(os.path.join(save_path, "glaucoma.jpg"), glaucoma)


    # Program_3
    leave_list = []

    for root, dirs, files in os.walk(leave_path):
        for file in files:
            leave_list.append(os.path.join(root, file))
    Program_3 = utils.different_leave(leave_list, signatures_path)
    key, static_image = Program_3.plot_statistics()
    
    for idx in range(len(key)):
        cv2.imwrite(os.path.join(save_path, key[idx] + ".jpg"), static_image[idx])