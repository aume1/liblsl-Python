import time
import pygame
from pygame.locals import *
import random

fps = 60


class Block(pygame.sprite.Sprite):
    VEL = 300
    scores = []

    def __init__(self, screen_size):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([640, 720])
        self.image.fill([0, 0, 0])
        self.rect = self.image.get_rect()
        self.rect.y = screen_size[1] - self.rect.height
        self.dt = 1/fps
        self.max_x = screen_size[0] - self.rect.width
        self.max_y = screen_size[1] - self.rect.height
        self.scores = [-1]
        self.pos = 'l'
        self.show_time = time.time()
        self.green_time = 0
        self.red_time = 0
        self.last_actions = []
        self.last_poses = []
        self.reset()

    def update(self, *args, **kwargs):
        if time.time() < self.green_time:
            self.image.fill([0, 255, 0])
        elif time.time() < self.red_time:
            self.image.fill([255, 0, 0])
        elif time.time() < self.show_time:
            self.rect.y = -720
            self.last_actions = []
        elif self.rect.y == -720:
            self.last_actions = []
            rand = random.randint(0, 1)
            self.pos = 'l' if rand == 0 else 'r'
            if len(self.last_poses) == 5:
                if all(self.pos == x for x in self.last_poses):
                    self.pos = 'r' if self.pos == 'l' else 'r'
                self.last_poses = self.last_poses[1:]
            self.last_poses += [self.pos]
            self.rect.x = rand*640
            self.rect.y = 0
            self.image.fill([0, 0, 0])

        return self.draw(args[0])

    def draw(self, surface):
        surface.blit(self.image, (self.rect.x, self.rect.y))

    def handle_keys(self, left_right, eeg=True):
        left, right = left_right
        if eeg:
            if right:
                self.last_actions += [1]
            elif left:
                self.last_actions += [-1]
            else:
                self.last_actions += [0]
            if len(self.last_actions) >= 125:
                self.last_actions = self.last_actions[1:]
                avg = sum(self.last_actions) / len(self.last_actions)
                left = avg < -0.33
                right = avg > 0.33
            else:
                left, right = 0, 0

        if time.time() > self.show_time and (left or right):
            Block.scores += [time.time() - self.show_time]
            print('time = {:0.3f}. '.format(Block.scores[-1]), end='')
            if (right and self.pos == 'l') or (left and self.pos == 'r'):  # if the wrong command is sent, invert time
                print('incorrect.  ', end='')
                if right:
                    print('left, got right')
                else:
                    print('right, got left')
                self.red_time = time.time() + 0.5
                Block.scores[-1] = -Block.scores[-1]
            else:
                self.green_time = time.time() + 0.5
                print('correct. was ', end='')
                if left:
                    print('left')
                else:
                    print('right')
            self.reset()

    def reset(self):
        self.show_time = time.time() + 0.75 + 3*random.random()


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(size=[1280, 720])
        pygame.display.set_caption("MTRN4093 - EEG BCI")

        self.clock = pygame.time.Clock()
        self.block = Block(self.screen.get_size())
        self.running = True

    def start(self):
        import threading
        self.running = True
        threading.Thread(target=self.run_keyboard).start()

    def iter(self):
        pygame.event.get()

    def run_keyboard(self, run_time=30):
        end_time = run_time + time.time()
        self.running = True
        while end_time > time.time():
            self.clock.tick(fps)
            pygame.event.get()
            keys = "{0:b}".format(pygame.key.get_mods())
            keys = int(keys[6]), int(keys[5])
            self.block.handle_keys(keys, eeg=False)

            if self.running:
                self.screen.fill((255, 255, 255))
                self.block.update(self.screen)
                pygame.display.flip()
        self.running = False

    def run_eeg(self, run_time=10):
        end_time = run_time + time.time()
        self.running = True
        while end_time > time.time():
            self.clock.tick(fps)
            pygame.event.get()
            if self.running:
                self.screen.fill((255, 255, 255))
                self.block.update(self.screen)
                pygame.display.flip()
        self.running = False

    def quit(self):
        pygame.quit()


def main():
    game = Game()
    game.start()
    while game.running:
        # print('iter')
        game.iter()
    pygame.quit()


if __name__ == "__main__":
    main()
