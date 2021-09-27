import pygame
from pygame.locals import *
import random

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
        self.score = -1

    def add_to_group(self, group=None):
        if group is None:
            pygame.sprite.Group().add(self)
        else:
            group.add(self)

    def remove_from_group(self):
        pygame.sprite.Group().remove(self)

    def handle_keys(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key in self.right_keys:
                self.keys['r'] = True
            elif event.key in self.left_keys:
                self.keys['l'] = True
        elif event.type == pygame.KEYUP:
            if event.key in self.left_keys:
                self.keys['l'] = False
            elif event.key in self.right_keys:
                self.keys['r'] = False

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
        if pygame.sprite.spritecollide(self, sprite_group, False):
            self.score += 1
            print(self.score)


class Enemy(pygame.sprite.Sprite):
    VEL = 300

    def __init__(self, screen_size):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([40, 40])
        self.image.fill([0, 0, 255])
        self.rect = self.image.get_rect()
        self.rect.y = screen_size[1] - self.rect.height - 10
        self.dt = 1/fps
        self.max_x = screen_size[0] - self.rect.width
        self.max_y = screen_size[1] - self.rect.height

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
            print('Game over!')
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
            self.reset()


def main():
    pygame.init()
    screen = pygame.display.set_mode(size=[1280, 720])
    pygame.display.set_caption("MTRN4093 - EEG BCI")

    clock = pygame.time.Clock()

    sprites = pygame.sprite.Group()
    p1 = Player(screen.get_size())
    p1.add_to_group(sprites)
    e = Enemy(screen.get_size())
    e.add_to_group(sprites)

    running = True
    while running:
        clock.tick(fps)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            if event.type in (pygame.KEYUP, pygame.KEYDOWN):
                p1.handle_keys(event)
        if running:
            screen.fill((255, 255, 255))
            for block in sprites:
                sprites.remove(block)
                block.collide(sprites)
                sprites.add(block)
            p1.update(screen)
            e.update(screen)
            pygame.display.flip()


if __name__ == "__main__":
    main()
