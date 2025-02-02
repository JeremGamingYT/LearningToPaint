import os
import cv2
import torch
import numpy as np
import argparse
import random
import time
from torch import nn
from torch.optim import Adam, SGD
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import tensorboardX as tb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128

# Neural Renderer
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv5 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv6 = nn.Conv2d(8, 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 128, 128)

# Actor Network
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_inputs=9, num_outputs=65):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_outputs)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)

# Environment
class PaintEnvironment:
    def __init__(self, batch_size, max_step, data_path, renderer):
        self.batch_size = batch_size
        self.max_step = max_step
        self.renderer = renderer
        self.action_space = 13
        self.width = width
        self.img_train = []
        self.img_test = []
        self.load_data(data_path)
        self.canvas = torch.zeros((batch_size, 3, width, width), device=device)
        self.gt = None
        self.tot_reward = None
        self.stepnum = 0

    def load_data(self, data_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        all_images = []
        
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    all_images.append(os.path.join(root, file))

        random.shuffle(all_images)
        test_split = int(0.1 * len(all_images))
        
        self.img_test = all_images[:test_split]
        self.img_train = all_images[test_split:]

    def pre_data(self, img_path, test=False):
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (width, width))
            
            if not test:
                img = transforms.ToPILImage()(img)
                img = transforms.RandomHorizontalFlip()(img)
                img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(img)
                img = np.array(img)
            
            return torch.tensor(img).permute(2, 0, 1).to(device)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return None

    def reset(self, test=False, begin_num=0):
        self.gt = torch.zeros((self.batch_size, 3, width, width), device=device)
        valid_count = 0
        
        while valid_count < self.batch_size:
            if test:
                img_path = self.img_test[(begin_num + valid_count) % len(self.img_test)]
            else:
                img_path = random.choice(self.img_train)
            
            img_tensor = self.pre_data(img_path, test)
            if img_tensor is not None:
                self.gt[valid_count] = img_tensor
                valid_count += 1
        
        self.tot_reward = (self.gt.float() / 255).pow(2).mean(dim=(1,2,3))
        self.stepnum = 0
        self.canvas = torch.zeros_like(self.gt)
        return self.observation()

    def observation(self):
        T = torch.ones((self.batch_size, 1, width, width), device=device) * self.stepnum
        coord = torch.zeros((self.batch_size, 2, width, width), device=device)
        for i in range(width):
            for j in range(width):
                coord[:, 0, i, j] = i / (width - 1)
                coord[:, 1, i, j] = j / (width - 1)
        return torch.cat([self.canvas.float()/255, self.gt.float()/255, T.float()/self.max_step, coord], dim=1)

    def step(self, action):
        canvas = decode(action, self.canvas.float()/255, self.renderer) * 255
        self.stepnum += 1
        reward = self.cal_reward(canvas)
        self.canvas = canvas.byte()
        done = self.stepnum >= self.max_step
        return self.observation(), reward, done, None

    def cal_reward(self, canvas):
        current_dist = ((canvas - self.gt.float()) / 255).pow(2).mean(dim=(1,2,3))
        return (self.tot_reward - current_dist) / (self.tot_reward + 1e-8)

# Agent DDPG
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states), torch.stack(actions), torch.stack(rewards), 
                torch.stack(next_states), torch.stack(dones))

    def __len__(self):
        return len(self.buffer)

class DDPG:
    def __init__(self, batch_size=64, max_step=40, tau=0.001, gamma=0.95, buffer_size=1000000, renderer=None):
        self.renderer = renderer
        self.actor = ResNet(BasicBlock, [2,2,2,2]).to(device)
        self.actor_target = ResNet(BasicBlock, [2,2,2,2]).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-4)
        
        self.critic = ResNet(BasicBlock, [2,2,2,2], num_inputs=12).to(device)
        self.critic_target = ResNet(BasicBlock, [2,2,2,2], num_inputs=12).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-3)
        
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.max_step = max_step
        
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, noise_factor=0.1):
        with torch.no_grad():
            action = self.actor(state)
            if noise_factor > 0:
                noise = torch.randn_like(action) * noise_factor
                action = torch.clamp(action + noise, 0, 1)
            return action

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(torch.cat([next_states, next_actions], dim=1))
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q
        
        current_q = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, actor_actions], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        
        return actor_loss.item(), critic_loss.item()

def decode(x, canvas, renderer):
    x = x.view(-1, 13)
    stroke = 1 - renderer(x[:, :10])
    stroke = stroke.view(-1, 128, 128, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    for _ in range(5):
        canvas = canvas * (1 - stroke) + color_stroke
    return canvas

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def train(data_path, max_step=40, batch_size=64, episodes=10000, renderer_path='renderer.pkl'):
    renderer = FCN().to(device)
    renderer.load_state_dict(torch.load(renderer_path))
    
    env = PaintEnvironment(batch_size, max_step, data_path, renderer)
    agent = DDPG(batch_size, max_step, renderer=renderer)
    
    writer = tb.SummaryWriter()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_step):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward.mean()
            
            actor_loss, critic_loss = agent.update()
            
            writer.add_scalar('Loss/actor', actor_loss, episode*max_step + step)
            writer.add_scalar('Loss/critic', critic_loss, episode*max_step + step)
            writer.add_scalar('Reward/step', reward.mean(), episode*max_step + step)
        
        writer.add_scalar('Reward/episode', total_reward, episode)
        
        if episode % 100 == 0:
            torch.save(agent.actor.state_dict(), f'actor_{episode}.pth')
            torch.save(agent.critic.state_dict(), f'critic_{episode}.pth')

# Configuration et entraînement
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_step', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--renderer', type=str, default='renderer.pkl')
    parser.add_argument('--data_path', type=str, required=True, 
                      help="Path to anime images dataset folder")
    args = parser.parse_args()
    
    train(args.data_path, args.max_step, args.batch_size, args.episodes, args.renderer)