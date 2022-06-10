import pygame
import time

class game():
    def __init__(self,broad_size=11,batch_size=32):
        self.broad_size = broad_size
        self.batch_size = batch_size
    def delay(second):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()           
        time.sleep(second)

    def display(self,all_action):
        pygame.init()

        DISPLAY_WIDTH = 800
        DISPLAY_HEIGHT = 600

        gameDisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        pygame.display.set_caption("Draw Shapes")

        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)

        clock = pygame.time.Clock()

        playing = True
        while playing:
            #pygame.event.get()
            for i in range(11):
                for j in range(11):
                    pygame.draw.rect(gameDisplay, WHITE, pygame.Rect((10+18*j)+36*i, 100+36*j, 30, 30)) #位於100,100 長度為40
            player=2
            for i in range(len(all_action)):
                pygame.event.get()
                if(player==2):
                    player=1
                    pygame.draw.circle(gameDisplay, RED, (((10+18*all_action[i][1])+36*all_action[i][0])+15, (100+36*all_action[i][1])+15), 11)
                    pygame.time.delay(200) 
                    pygame.display.update()
                else:
                    player=2
                    pygame.draw.circle(gameDisplay, BLUE, (((10+18*all_action[i][1])+36*all_action[i][0])+15, (100+36*all_action[i][1])+15), 11)
                    pygame.time.delay(200) 
                    pygame.display.update()
            #pygame.draw.rect(gameDisplay, RED, pygame.Rect(28, 136, 30, 30))
            #pygame.time.delay(1000) 
            #pygame.display.update()
            #pygame.draw.circle(gameDisplay, RED, (28+15, 136+15), 11)
            #pygame.time.delay(1000) 
            pygame.display.update()

            clock.tick(30)

        pygame.quit()
        quit()
