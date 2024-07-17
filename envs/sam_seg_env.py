import numpy as np
import glob
import re
import cv2
import gymnasium as gym
from gymnasium import spaces
from envs.utils.repvit_sam_wrapper import RepVITSamWrapper


class SamSegEnv(gym.Env):

    def __init__(self, img_shape, embedding_shape, mask_shape, 
                 tgt_class_idx, max_steps, img_dir, gt_mask_dir, 
                 sam_ckpt_fp, img_patch_size=64):
        self.img_shape = img_shape  # The size of the image 
        self.embedding_shape = embedding_shape  # The size of the SAM image encoder output
        self.mask_shape = mask_shape  # The size of the mask

        self.tgt_class_idx = tgt_class_idx  # The target class index to use from ground truth mask
        self.max_steps = max_steps  # The maximum number of steps the agent can take

        self.img_dir = img_dir
        self.gt_mask_dir = gt_mask_dir

        self.img_patch_size = img_patch_size

        self._image = None
        self._sam_image_embeddings = None
        self._sam_pred_mask_prob = None
        self._sam_pred_mask = None
        self._gt_mask = None
        self._num_steps = 0
        self._last_actions = {'input_points':[], 'input_labels':[]}
        self._last_score = 0

        self.sam_predictor = RepVITSamWrapper(sam_ckpt_fp)

        self.pattern = re.compile(r'uid([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})')

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(*img_shape,), dtype=int),
                "sam_image_embeddings": spaces.Box(-np.inf, np.inf, shape=(*embedding_shape,), dtype=float),
                "sam_pred_mask_prob": spaces.Box(0, 1, shape=(*mask_shape,), dtype=float),
            }
        )

        # Number of actions is equal to number of patches in the image*2 + 1
        # Each patch can be marked as positive or negative input
        img_h, img_w = img_shape[:2]
        # number of patches along a single dimension is the ceiling of the division of the image size by the patch size
        num_width_patches = img_w//img_patch_size + int(img_w%img_patch_size != 0)
        num_height_patches = img_h//img_patch_size + int(img_h%img_patch_size != 0)
        num_patches = num_width_patches * num_height_patches
        # print(num_width_patches, num_height_patches)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the input_points and input_labels submitted to SAM in if that action is taken.
        I.e. 0 corresponds to "(0, 0), positive", 1 to "(0, 0), negative",
        2 to "(0, 1), positive", 3 to "(0, 1), negative", etc
        """
        self._action_to_input = {
        }
        for i in range(0, num_patches*2, 2):
            row_idx = i // num_width_patches
            col_idx = i % num_width_patches
            # Mark patch center as input point (x, y)
            input_point = (col_idx * img_patch_size + img_patch_size//2, 
                           row_idx * img_patch_size + img_patch_size//2)
            
            # print(i, col_idx, row_idx, input_point)
            
            self._action_to_input[i] = (input_point, 1)
            self._action_to_input[i + 1] = (input_point, 0)

        # The 2nd last action is to remove previous input
        self._action_to_input[num_patches*2] = ('remove', -1)
        # The last action is to mark the task as done
        self._action_to_input[num_patches*2 + 1] = ('done', -1)
        self.action_space = spaces.Discrete(len(self._action_to_input))

        self._load_image_and_mask_fps()


    def extract_uuid(self, filename):
        match = self.pattern.search(filename)
        if match is None:
            raise Exception(f'No uuid match found. {filename}')

        uuid_key = match.group()[3:]  # Strip the 'uid' in the beginning.
        return uuid_key


    def _load_image_and_mask_fps(self):
        image_extns = [".jpg", ".jpeg", ".png", ".PNG"]
        mask_extn = ".png"
        self.img_fps = []
        for extn in image_extns:
            self.img_fps += glob.glob(self.img_dir + "/*" + extn)

        self.img_fps.sort()

        mask_uid_to_fp = {}
        for mask_fp in glob.glob(self.gt_mask_dir + "/*" + mask_extn):
            mask_uid_to_fp[self.extract_uuid(mask_fp)] = mask_fp

        self.mask_fps = [mask_uid_to_fp[self.extract_uuid(img_fp)] for img_fp in self.img_fps]

        chosen_idx = self.np_random.integers(0, len(self.img_fps))
        self._load_image_and_mask(self.img_fps[chosen_idx], self.mask_fps[chosen_idx])


    def _get_obs(self):
        return {
            "image": self._image, 
            "sam_image_embeddings": self._sam_image_embeddings,
            "sam_pred_mask_prob": self._sam_pred_mask_prob,
        }
    

    def _load_image_and_mask(self, img_fp, gt_mask_fp):
        self._image = cv2.resize(cv2.cvtColor(cv2.imread(img_fp, -1), cv2.COLOR_BGR2RGB), self.img_shape[:2][::-1])
        self.sam_predictor.set_image(self._image)
        self._sam_image_embeddings = self.sam_predictor.get_image_embeddings()
        self._sam_pred_mask_prob = np.ones(self.mask_shape, dtype=np.float32)
        self._sam_pred_mask = np.zeros(self.img_shape[:2], dtype=np.float32)

        mask = cv2.resize(cv2.imread(gt_mask_fp, cv2.IMREAD_GRAYSCALE), 
                          self.img_shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        mask = ( mask/ 50).astype(np.uint8)
        self._gt_mask = (mask == self.tgt_class_idx).astype(np.float32)


    def _get_info(self):
        return {
            "last_input_points": self._last_actions["input_points"].copy(),
            "last_input_labels": self._last_actions["input_labels"].copy(),
            "sam_pred_mask": self._sam_pred_mask.copy(),
        }
    

    def run_sam(self):
        masks, ious, low_res_mask_logits = self.sam_predictor.predict(
            np.array(self._last_actions["input_points"]), 
            np.array(self._last_actions["input_labels"])
        )

        best_mask_idx = np.argmax(ious)
        best_pred_mask_prob = 1 / (1 + np.exp(-low_res_mask_logits[best_mask_idx]))
        self._sam_pred_mask_prob = best_pred_mask_prob
        self._sam_pred_mask = (1 / (1 + np.exp(-masks[best_mask_idx]))).astype(np.float32)


    def compute_reward(self, pred_mask):
        # Compute dice score
        resized_gt_mask = cv2.resize(self._gt_mask,
                                     pred_mask.shape[::-1],
                                     interpolation=cv2.INTER_NEAREST)
        assert resized_gt_mask.shape == pred_mask.shape

        intersection = np.sum(resized_gt_mask * pred_mask)
        union = np.sum(resized_gt_mask) + np.sum(pred_mask) - intersection

        dice_score = 2 * intersection / union
        reward = dice_score - self._last_score
        self._last_score = dice_score
        return reward


    def step(self, action):
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action: {action}")
        
        if self.max_steps is not None and self._num_steps >= self.max_steps:
            raise ValueError("Maximum number of steps reached")
        
        
        if self._action_to_input[action][0] == 'done':
            # Task is done
            terminated = True
            reward = 0
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, False, info
        
        if self._action_to_input[action][0] == 'remove':
            # Remove the last input
            if len(self._last_actions["input_points"]) > 0:
                self._last_actions["input_points"].pop()
                self._last_actions["input_labels"].pop()
        else:
            input_point, input_label = self._action_to_input[action]
            
            self._last_actions["input_points"].append(input_point)
            self._last_actions["input_labels"].append(input_label)

        if len(self._last_actions["input_points"]) > 0:
            self.run_sam()
        else:
            # Set the mask and mask_prob to initial state
            self._sam_pred_mask_prob = np.ones(self.mask_shape, dtype=np.float32)
            self._sam_pred_mask = np.zeros(self.img_shape[:2], dtype=np.float32)
        
        self._num_steps += 1

        reward = self.compute_reward(self._sam_pred_mask)
        terminated = self._num_steps >= self.max_steps

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        chosen_idx = self.np_random.integers(0, len(self.img_fps))
        img_fp = self.img_fps[chosen_idx]
        gt_mask_fp = self.mask_fps[chosen_idx]

        self._load_image_and_mask(img_fp, gt_mask_fp)
        self._last_score = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


if __name__ == "__main__":
    img_shape = (640, 852, 3) # HxWxC
    embedding_shape = (256, 64, 64) # CxHxW
    mask_shape = (256, 256) # HxW
    tgt_class_idx = 3  
    max_steps = 10

    img_dir = '/media/Data/sam_align_data/images/train'
    gt_mask_dir = '/media/Data/sam_align_data/annotations/train'
    sam_ckpt_fp = '/media/Projects/RepViT/sam/weights/repvit_sam.pt'

    view_width = 426
    view_height = 320

    env = SamSegEnv(img_shape=img_shape, 
                    embedding_shape=embedding_shape,
                    mask_shape=mask_shape,
                    tgt_class_idx=tgt_class_idx,
                    max_steps=max_steps,
                    img_dir=img_dir,
                    gt_mask_dir=gt_mask_dir,
                    sam_ckpt_fp=sam_ckpt_fp)

    obs, info = env.reset()
    
    global sample_action
    sample_action = env.action_space.n - 2 #np.random.randint(0, env.action_space.n)

    def get_action(event,x,y,flags,param):
        global sample_action
        if event == cv2.EVENT_LBUTTONDBLCLK:
            tgt_label = 1
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            tgt_label = 0
        else:
            return
        x = x % view_width
        y = y % view_height
        scaled_x = int(x * img_shape[1] / view_width)
        scaled_y = int(y * img_shape[0] / view_height)
        input_dist = np.array([np.linalg.norm([scaled_x-point[0], scaled_y-point[1]]) + (1e6*(label != tgt_label))\
                                for (point, label) in env._action_to_input.values() if type(point) == tuple])
        sample_action = np.argmin(input_dist)

        # col_num = scaled_x // env.img_patch_size
        # row_num = scaled_y // env.img_patch_size
        # num_patches_width = (img_shape[1] // env.img_patch_size) + int(img_shape[1] % env.img_patch_size != 0)
        # sample_action = (row_num * num_patches_width + col_num) + offset
        # print(scaled_x, scaled_y, tgt_label, sample_action, env._action_to_input[sample_action])

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_action)
    while True:
        obs, reward, done, _, info = env.step(sample_action)
        print(reward)

        # print(obs['image'].shape, info['sam_pred_mask'].shape)
        print(info['last_input_labels'], info['last_input_points'])

        img = cv2.resize(cv2.cvtColor(obs['image'], cv2.COLOR_BGR2RGB), (view_width, view_height))
        mask = cv2.cvtColor((info['sam_pred_mask'] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for point,label in zip(info['last_input_points'], info['last_input_labels']):
            cv2.circle(mask, point, 10, (0, 255, 0) if label == 1 else (0, 0, 255), -1)
        mask = cv2.resize(mask, (view_width, view_height))
        
        concat_img = np.concatenate([img, mask], axis=1)
        cv2.imshow('image', concat_img)

        sample_action = env.action_space.n - 1  # Mark task as done by default

        if cv2.waitKey(0) == 27 or done:  # Esc key to stop the loop or task is done
            break



    cv2.destroyAllWindows()



    