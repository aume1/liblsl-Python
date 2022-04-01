import copy
import time

import pygame
from pygame.locals import *
import random
# from ctypes import windll
# SetWindowPos = windll.user32.SetWindowPos

fps = 60


class Player(pygame.sprite.Sprite):
    ACCEL = 3000
    DECCEL = 3000
    MAX_VEL = 1000

    left_keys = (pygame.K_a, pygame.K_LCTRL, pygame.K_LEFT)
    right_keys = (pygame.K_d, pygame.K_RCTRL, pygame.K_RIGHT)

    def __init__(self, screen_size):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([200, 30])
        self.image.fill([255, 0, 0])
        self.rect = self.image.get_rect()
        self.rect.y = screen_size[1] - self.rect.height - 10
        self.vel = {'x': 0, 'y': 0}
        self.accel = {'x': 0, 'y': 0}
        self.dt = 1/fps
        self.max_x = screen_size[0] - self.rect.width
        self.max_y = screen_size[1] - self.rect.height
        self.keys = {'l': False, 'r': False}

    def add_to_group(self, group=None):
        if group is None:
            pygame.sprite.Group().add(self)
        else:
            group.add(self)

    def remove_from_group(self):
        pygame.sprite.Group().remove(self)

    def handle_keys(self, left_right):
        self.keys['l'], self.keys['r'], *_ = left_right

    def update(self, *args, **kwargs):
        if self.keys['l'] and not self.keys['r']:
            self.accel['x'] = -self.ACCEL
        elif self.keys['r'] and not self.keys['l']:
            self.accel['x'] = self.ACCEL
        else:
            self.accel['x'] = 0

        for key in self.accel:
            a = self.accel[key]
            if self.accel[key] == 0:
                a = -Player.DECCEL if self.vel[key] > 0 else Player.DECCEL
            self.vel[key] = self.vel[key] + int(self.dt * a)
            if self.vel[key] > Player.MAX_VEL:
                self.vel[key] = Player.MAX_VEL
            elif self.vel[key] < -Player.MAX_VEL:
                self.vel[key] = -Player.MAX_VEL
        self.rect.x = self.rect.x + int(self.dt*self.vel['x'])
        self.rect.y = self.rect.y + int(self.dt*self.vel['y'])

        if self.rect.x > self.max_x:
            self.rect.x = self.max_x
            self.vel['x'] = -int(0.8*self.vel['x'])
        elif self.rect.x < 0:
            self.rect.x = 0
            self.vel['x'] = -int(0.8*self.vel['x'])

        if self.rect.y > self.max_y:
            self.rect.y = self.max_y
            self.vel['y'] = 0
        elif self.rect.y < 0:
            self.rect.y = 0
            self.vel['y'] = 0

        if isinstance(args[0], pygame.Surface):
            return self.draw(args[0])

    def draw(self, surface):
        surface.blit(self.image, (self.rect.x, self.rect.y))

    def collide(self, sprite_group):
        return


class Enemy(pygame.sprite.Sprite):
    VEL = 300
    # scores = [-1]

    def __init__(self, screen_size):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([40, 40])
        self.image.fill([0, 0, 255])
        self.rect = self.image.get_rect()
        self.rect.y = screen_size[1] - self.rect.height - 10
        self.dt = 1/fps
        self.max_x = screen_size[0] - self.rect.width
        self.max_y = screen_size[1] - self.rect.height
        self.scores = [-1]

    def add_to_group(self, group=None):
        if group is None:
            pygame.sprite.Group().add(self)
        else:
            group.add(self)

    def remove_from_group(self):
        pygame.sprite.Group().remove(self)

    def update(self, *args, **kwargs):
        self.rect.y = self.rect.y + int(self.dt*self.VEL)
        if self.rect.y > self.max_y:
            print('resetting score!\n0')
            self.scores.append(0)
            self.reset()

        if isinstance(args[0], pygame.Surface):
            return self.draw(args[0])

    def draw(self, surface):
        surface.blit(self.image, (self.rect.x, self.rect.y))

    def reset(self):
        self.rect.y = -self.rect.height
        newpos = self.rect.x
        while self.rect.x - 100 < newpos < self.rect.x + 100:
            newpos = random.randrange(self.rect.width, self.max_x-self.rect.width)
        self.rect.x = newpos

    def collide(self, sprite_group):
        if pygame.sprite.spritecollide(self, sprite_group, False):
            self.scores[-1] += 1
            print(self.scores[-1])
            self.reset()


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(size=[1280, 720])
        pygame.display.set_caption("MTRN4093 - EEG BCI")
        time.sleep(0.5)
        # hwnd = pygame.display.get_wm_info()['window']
        # SetWindowPos(hwnd, -1, 0, 0, 0, 0, 2|1)

        self.clock = pygame.time.Clock()

        self.sprites = pygame.sprite.Group()
        self.p1 = Player(self.screen.get_size())
        self.p1.add_to_group(self.sprites)
        self.e = Enemy(self.screen.get_size())
        self.e.add_to_group(self.sprites)
        self.running = True

    def start(self):
        import threading
        self.running = True
        threading.Thread(target=self.run_keyboard).start()

    def iter(self):
        print(pygame.event.get())

    def run_keyboard(self, run_time=10):
        end_time = run_time + time.time()
        while end_time > time.time():
            self.clock.tick(fps)
            pygame.event.get()
            keys = "{0:b}".format(pygame.key.get_mods())

            self.p1.handle_keys((int(keys[6]), int(keys[5])))  # sends the left and right keys to
            if self.running:
                self.screen.fill((255, 255, 255))
                for block in self.sprites:
                    self.sprites.remove(block)
                    block.collide(self.sprites)
                    self.sprites.add(block)
                self.p1.update(self.screen)
                self.e.update(self.screen)
                pygame.display.flip()

    def run_eeg(self, run_time=10):
        end_time = run_time + time.time()
        while end_time > time.time():
            self.clock.tick(fps)
            pygame.event.get()
            if self.running:
                self.screen.fill((255, 255, 255))
                for block in self.sprites:
                    self.sprites.remove(block)
                    block.collide(self.sprites)
                    self.sprites.add(block)
                self.p1.update(self.screen)
                self.e.update(self.screen)
                pygame.display.flip()

    def quit(self):
        pygame.quit()


def main():
    game = Game()
    # while True:
    #     pygame.event.get()
    #     keys = "{0:b}".format(pygame.key.get_mods())
    #     # print(keys)
    #     if int(keys[5]):
    #         print('RCTRL')
    #     elif int(keys[6]):
    #         print('LCTRL')
    #     # for i in range(1, len(keys)):
    #     #     if keys[i] == '1':
    #     #         print(i)
    #     # print(pygame.key.get_pressed())
    game.start()
    while True:
        game.iter()
    # pygame.init()
    # screen = pygame.display.set_mode(size=[1280, 720])
    # pygame.display.set_caption("MTRN4093 - EEG BCI")
    # time.sleep(0.5)
    # hwnd = pygame.display.get_wm_info()['window']
    # SetWindowPos(hwnd, -1, 0, 0, 0, 0, 2|1)
    #
    # clock = pygame.time.Clock()
    #
    # sprites = pygame.sprite.Group()
    # p1 = Player(screen.get_size())
    # p1.add_to_group(sprites)
    # e = Enemy(screen.get_size())
    # e.add_to_group(sprites)
    #
    # running = True
    # # end_time = time.time() + 20
    # while running:# and end_time > time.time():
    #     clock.tick(fps)
    #     events = pygame.event.get()
    #     for event in events:
    #         if event.type == pygame.QUIT:
    #             running = False
    #             pygame.quit()
    #         if event.type in (pygame.KEYUP, pygame.KEYDOWN):
    #             p1.handle_keys(event)
    #     if running:
    #         screen.fill((255, 255, 255))
    #         for block in sprites:
    #             sprites.remove(block)
    #             block.collide(sprites)
    #             sprites.add(block)
    #         p1.update(screen)
    #         e.update(screen)
    #         pygame.display.flip()
    print('scores:', game.e.scores)


if __name__ == "__main__":
    main()
