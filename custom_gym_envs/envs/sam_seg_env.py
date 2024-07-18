import numpy as np
import glob
import re
import cv2
import gymnasium as gym
from gymnasium import spaces
from custom_gym_envs.envs.utils.repvit_sam_wrapper import RepVITSamWrapper


class SamSegEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 1}

    def __init__(self, img_shape, embedding_shape, mask_shape, render_frame_shape,
                 tgt_class_idx, max_steps, img_dir, gt_mask_dir, 
                 sam_ckpt_fp, img_patch_size=64, render_mode='rgb_array'):
        self.img_shape = img_shape  # The size of the image 
        self.embedding_shape = embedding_shape  # The size of the SAM image encoder output
        self.mask_shape = mask_shape  # The size of the mask
        self.render_frame_shape = render_frame_shape  # The size of the frame to render

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

        assert render_mode in self.metadata["render_modes"], "Invalid render mode"
        self.render_mode = render_mode

        self.sam_predictor = RepVITSamWrapper(sam_ckpt_fp)

        self.pattern = re.compile(r'uid([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})')

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(*img_shape,), dtype=np.uint8),
                "sam_image_embeddings": spaces.Box(-np.inf, np.inf, shape=(*embedding_shape,), dtype=np.float32),
                "sam_pred_mask_prob": spaces.Box(0, 1, shape=(*mask_shape,), dtype=np.float32),
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
        for i in range(0, num_patches, 2):
            row_idx = i // num_width_patches
            col_idx = i % num_width_patches
            # Mark patch center as input point (x, y)
            input_point = (col_idx * img_patch_size + img_patch_size//2, 
                           row_idx * img_patch_size + img_patch_size//2)
            
            # print(i, col_idx, row_idx, input_point)
            
            self._action_to_input[i] = (input_point, 1)
            self._action_to_input[i + 1] = (input_point, 0)

        # The 2nd last action is to remove previous input
        self._action_to_input[num_patches] = ('remove', -1)
        # The last action is to mark the task as done
        self._action_to_input[num_patches + 1] = ('done', -1)
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

        if len(self.img_fps) == 0:
            raise ValueError(f"No images found in {self.img_dir}")

        self.img_fps.sort()

        mask_uid_to_fp = {}
        for mask_fp in glob.glob(self.gt_mask_dir + "/*" + mask_extn):
            mask_uid_to_fp[self.extract_uuid(mask_fp)] = mask_fp

        if len(mask_uid_to_fp) == 0:
            raise ValueError(f"No masks found in {self.gt_mask_dir}")

        self.mask_fps = [mask_uid_to_fp[self.extract_uuid(img_fp)] for img_fp in self.img_fps]

        chosen_idx = self.np_random.integers(0, len(self.img_fps))
        self._load_image_and_mask(self.img_fps[chosen_idx], self.mask_fps[chosen_idx])


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


    def _get_obs(self):
        return {
            "image": self._image, 
            "sam_image_embeddings": self._sam_image_embeddings,
            "sam_pred_mask_prob": self._sam_pred_mask_prob,
        }
    

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


    def convert_raw_input_to_action(self, input_point, input_label):
        input_dist = np.array([np.linalg.norm([input_point[0]-point[0], input_point[1]-point[1]]) + \
                               (1e6*(label != input_label)) for (point, label) in env._action_to_input.values() \
                                if type(point) == tuple])
        sample_action = np.argmin(input_dist)
        return sample_action


    def compute_reward(self, pred_mask, act):
        # Compute dice score
        resized_gt_mask = cv2.resize(self._gt_mask,
                                     pred_mask.shape[::-1],
                                     interpolation=cv2.INTER_NEAREST)
        assert resized_gt_mask.shape == pred_mask.shape

        intersection = np.sum(resized_gt_mask * pred_mask)
        union = np.sum(resized_gt_mask) + np.sum(pred_mask) #- intersection

        eps = 1e-6
        dice_score = (2 * intersection + eps)/ (union + eps)
        reward = dice_score - self._last_score
        self._last_score = dice_score

        if act == 'add':
            input_point, input_label = self._last_actions["input_points"][-1], \
                self._last_actions["input_labels"][-1]
            point_image_indices = tuple(map(int, (input_point[1], input_point[0])))
            # print(point_image_indices, input_label)
            gt_label = self._gt_mask[point_image_indices]
            reward += int(input_label == gt_label)

        return reward


    def step(self, action):
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action: {action}")
        
        terminated = self._action_to_input[action][0] == 'done'
        trunc = (self.max_steps is not None and self._num_steps >= self.max_steps)
        if terminated or trunc:
            return self._get_obs(), 0, terminated, trunc, self._get_info()
        
    
        act = None
        if self._action_to_input[action][0] == 'remove':
            # Remove the last input
            if len(self._last_actions["input_points"]) > 0:
                self._last_actions["input_points"].pop()
                self._last_actions["input_labels"].pop()
                act = 'remove'
        else:
            input_point, input_label = self._action_to_input[action]
            
            self._last_actions["input_points"].append(input_point)
            self._last_actions["input_labels"].append(input_label)
            act = 'add'

        if len(self._last_actions["input_points"]) > 0:
            self.run_sam()
        else:
            # Set the mask and mask_prob to initial state
            self._sam_pred_mask_prob = np.ones(self.mask_shape, dtype=np.float32)
            self._sam_pred_mask = np.zeros(self.img_shape[:2], dtype=np.float32)
        
        self._num_steps += 1

        reward = self.compute_reward(self._sam_pred_mask, act)
        trunc = (self.max_steps is not None and self._num_steps >= self.max_steps)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, False, trunc, info
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        chosen_idx = self.np_random.integers(0, len(self.img_fps))
        img_fp = self.img_fps[chosen_idx]
        gt_mask_fp = self.mask_fps[chosen_idx]
        self._load_image_and_mask(img_fp, gt_mask_fp)

        self._last_actions = {'input_points':[], 'input_labels':[]}
        self._num_steps = 0
        self._last_score = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def render(self):
        img = cv2.resize(cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR), self.render_frame_shape[::-1])

        gt_mask = cv2.resize((self._gt_mask * 255).astype(np.uint8), self.render_frame_shape[::-1])
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)


        mask = cv2.cvtColor((self._sam_pred_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for point,label in zip(self._last_actions["input_points"], self._last_actions["input_labels"]):
            cv2.circle(mask, point, 10, (0, 255, 0) if label == 1 else (0, 0, 255), -1)
        mask = cv2.resize(mask, self.render_frame_shape[::-1])
        
        concat_img = np.concatenate([img, gt_mask, mask], axis=1)

        if self.render_mode == 'rgb_array':
            return cv2.cvtColor(concat_img, cv2.COLOR_BGR2RGB)

        return concat_img


if __name__ == "__main__":
    img_shape = (640, 852, 3) # HxWxC
    embedding_shape = (256, 64, 64) # CxHxW
    mask_shape = (256, 256) # HxW
    render_frame_shape = (320, 426) # HxW
    tgt_class_idx = 3  
    max_steps = 10
    img_patch_size = 64
    render_mode = 'human'

    img_dir = '/media/shantanu/Data/sam_align_data/images/train'
    gt_mask_dir = '/media/shantanu/Data/sam_align_data/annotations/train'
    sam_ckpt_fp = '/home/shantanu/Projects/RepViT/sam/weights/repvit_sam.pt'


    env = SamSegEnv(img_shape=img_shape, 
                    embedding_shape=embedding_shape,
                    mask_shape=mask_shape,
                    render_frame_shape=render_frame_shape,
                    tgt_class_idx=tgt_class_idx,
                    max_steps=max_steps,
                    img_dir=img_dir,
                    gt_mask_dir=gt_mask_dir,
                    sam_ckpt_fp=sam_ckpt_fp,
                    img_patch_size=img_patch_size,
                    render_mode=render_mode)

    print(env.action_space.n)
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
        x = x % render_frame_shape[1]
        y = y % render_frame_shape[0]
        scaled_x = int(x * img_shape[1] / render_frame_shape[1])
        scaled_y = int(y * img_shape[0] / render_frame_shape[0])
        sample_action = env.convert_raw_input_to_action((scaled_x, scaled_y), tgt_label)

        # col_num = scaled_x // env.img_patch_size
        # row_num = scaled_y // env.img_patch_size
        # num_patches_width = (img_shape[1] // env.img_patch_size) + int(img_shape[1] % env.img_patch_size != 0)
        # sample_action = (row_num * num_patches_width + col_num) + offset
        # print(scaled_x, scaled_y, tgt_label, sample_action, env._action_to_input[sample_action])

    cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('image', get_action)
    while True:
        obs, reward, done, _, info = env.step(sample_action)
        print(reward)

        # print(obs['image'].shape, info['sam_pred_mask'].shape)
        print(info['last_input_labels'], info['last_input_points'])

        frame = env.render()
        cv2.imshow('image', frame)

        sample_action = env.action_space.n - 1  # Mark task as done by default

        if cv2.waitKey(0) == 27 or done:  # Esc key to stop the loop or task is done
            break



    cv2.destroyAllWindows()



    