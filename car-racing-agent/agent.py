from PIL import Image
import gym
import numpy as np

env = gym.make('CarRacing-v0')

CAR_COLORS = [
    [204, 0, 0],
]

CAR_WHEEL_COLORS = [
    [76, 76, 76]
]

GRASS_COLORS = [
    [102, 255, 102],
    [102, 299, 102],
]

ROAD_COLORS = [
    [102, 102, 102],
    [105, 105, 105],
    [107, 107, 107],
]

GUARD_COLORS = [
    [255, 255, 255],
    [255, 0, 0]
]

ROAD_WHEEL_TRACE_COLORS = [
    [0, 0, 0]
]

GRASS_WHEEL_TRACE_COLORS = [
    [102, 102, 0]
]

OUT_OF_MAP_COLORS = [
    [0, 0, 0]
]

NORMALIZED_COLORS = {
    'BLACK': [0, 0, 0],
    'WHITE': [255, 255, 255],
    'RED': [255, 0, 0],
    'GREEN': [0, 255, 0]
}


def getAction(steering, acc, brk):
    return(np.array([steering, acc, brk], dtype="f"))


def checkColorIsInColorsArray(color, colors_array):
    flag = False
    for item in colors_array:
        if color[0] == item[0] and color[1] == item[1] and color[2] == item[2]: flag = True
    return flag


def normalize_observation(observation):
    new_observation = []
    for y in range(0, 84):
        temp_observations = []
        for x in range(0, 96):
            if checkColorIsInColorsArray(observation[y][x], CAR_COLORS):
                temp_observations.append(NORMALIZED_COLORS['RED'])
            elif checkColorIsInColorsArray(observation[y][x], OUT_OF_MAP_COLORS):
                temp_observations.append(NORMALIZED_COLORS['GREEN'])
            elif checkColorIsInColorsArray(observation[y][x], CAR_COLORS) or \
                    checkColorIsInColorsArray(observation[y][x], GUARD_COLORS) or \
                    checkColorIsInColorsArray(observation[y][x], GRASS_WHEEL_TRACE_COLORS):
                temp_observations.append(NORMALIZED_COLORS['BLACK'])
            elif checkColorIsInColorsArray(observation[y][x], ROAD_COLORS) or \
                    checkColorIsInColorsArray(observation[y][x], CAR_WHEEL_COLORS):
                temp_observations.append(NORMALIZED_COLORS['WHITE'])
            else:
                temp_observations.append(NORMALIZED_COLORS['BLACK'])
        new_observation.append(temp_observations)
    return new_observation


def saveImage(observation, name):
    img = Image.new('RGB', (len(observation), len(observation[0])), color=(0, 0, 0))
    for y in range(0, len(observation)):
        for x in range(0, len(observation[0])):
            r, g, b = observation[y][x]
            img.putpixel((y, x), (r, g, b, 255))

    img.save('images/{}.png'.format(name))


for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        # if t % 5 == 0: saveImage(normalize_observation(observation), t)
        env.render()
        action = env.action_space.sample()
        # action = getAction(0, 0.5, 0)
        observation, reward, done, info = env.step(action)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
