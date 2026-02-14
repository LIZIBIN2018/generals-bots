"""
Play online against real players on generals.io!

This lets you (human) play on the real generals.io servers.
You need to:
1. Go to https://generals.io and create an account
2. Get your user_id from the browser (check localStorage or profile)
3. Run this script with your user_id
"""

import time
import pygame
import numpy as np
from socketio import SimpleClient

from generals.core.config import Direction
from generals.core.observation import Observation
from generals.remote.generalsio_state import GeneralsIOstate

DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
BOT_ENDPOINT = "https://botws.generals.io/"


class HumanOnlinePlayer:
    """Human-controlled online player for generals.io"""
    
    def __init__(self):
        self.cursor = (0, 0)
        self.action_queue = []  # List of (source, dest, split) tuples
        self.split_mode = False
        self.grid_dims = (1, 1)
        self.cell_size = 40
        
    def set_grid_dims(self, width, height):
        self.grid_dims = (height, width)
        self.cursor = (height // 2, width // 2)
    
    def handle_event(self, event, observation):
        """Handle pygame input events."""
        if event.type == pygame.KEYDOWN:
            return self._handle_key(event, observation)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            return self._handle_click(event, observation)
        return None
    
    def _handle_click(self, event, observation):
        x, y = event.pos
        col = x // self.cell_size
        row = y // self.cell_size
        h, w = self.grid_dims
        if 0 <= row < h and 0 <= col < w:
            self.cursor = (row, col)
        return None
    
    def _handle_key(self, event, observation):
        key_to_dir = {
            pygame.K_UP: 0, pygame.K_w: 0,
            pygame.K_DOWN: 1, pygame.K_s: 1,
            pygame.K_LEFT: 2, pygame.K_a: 2,
            pygame.K_RIGHT: 3, pygame.K_d: 3,
        }
        
        if event.key == pygame.K_SPACE:
            self.split_mode = not self.split_mode
            print(f"Split: {'ON' if self.split_mode else 'OFF'}")
            return None
        
        if event.key == pygame.K_q:
            self.action_queue.clear()
            print("Queue cleared")
            return None
        
        if event.key == pygame.K_e:
            if self.action_queue:
                self.action_queue.pop()
                print(f"Undo [q={len(self.action_queue)}]")
            return None
        
        direction = key_to_dir.get(event.key)
        if direction is not None:
            return self._queue_move(direction, observation)
        
        return None
    
    def _queue_move(self, direction, observation):
        row, col = self.cursor
        h, w = self.grid_dims
        
        dr = [-1, 1, 0, 0][direction]
        dc = [0, 0, -1, 1][direction]
        new_row, new_col = row + dr, col + dc
        
        if not (0 <= new_row < h and 0 <= new_col < w):
            return None
        
        # Calculate indices for generals.io format
        source_idx = row * w + col
        dest_idx = new_row * w + new_col
        split = 1 if self.split_mode else 0
        
        action = (source_idx, dest_idx, split)
        self.action_queue.append(action)
        
        self.cursor = (new_row, new_col)
        
        dir_names = ['↑', '↓', '←', '→']
        print(f"({row},{col}){dir_names[direction]} [q={len(self.action_queue)}]")
        
        return action
    
    def get_action(self):
        """Get next action from queue, or None."""
        if self.action_queue:
            return self.action_queue.pop(0)
        return None


class OnlineGameClient(SimpleClient):
    """Client for playing on generals.io with human control."""
    
    def __init__(self, user_id: str):
        super().__init__()
        self.user_id = user_id
        self.bot_key = "sd09fjd203i0ejwi_changeme"
        self.game_state = None
        self.replay_id = ""
        self.player = HumanOnlinePlayer()
        
        # Pygame setup
        pygame.init()
        self.screen = None
        self.font = None
        
        print("Connecting to generals.io...")
        self.connect(BOT_ENDPOINT)
        print("Connected!")
    
    def join_lobby(self, lobby_id: str):
        """Join a private lobby."""
        payload = (lobby_id, self.user_id, self.bot_key)
        self.emit("join_private", payload)
        self.receive()
        print(f"Joined lobby: {lobby_id}")
        print(f"Share this link: https://bot.generals.io/games/{lobby_id}")
        print("Waiting for game to start... (press Force Start in browser or wait for opponent)")
    
    def join_1v1_queue(self):
        """Join the public 1v1 matchmaking queue."""
        self.emit("set_username", (self.user_id, self.bot_key))
        self.receive()
        self.emit("join_1v1", self.user_id)
        self.receive()
        self.queue_id = "1v1"
        print("Joined 1v1 queue! Waiting for opponent...")
    
    def wait_for_game(self):
        """Wait for game to start."""
        last_force = time.time()
        while True:
            # Handle pygame events while waiting
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False
            
            # Force start periodically
            if time.time() - last_force > 3:
                self.emit("set_force_start", (self.queue_id, True))
                last_force = time.time()
            
            try:
                event, *data = self.receive(timeout=0.1)
                if event == "game_start":
                    self._init_game(data[0])
                    return True
            except:
                pass
            
            pygame.time.wait(50)
    
    def _init_game(self, data):
        """Initialize game state."""
        self.game_state = GeneralsIOstate(data)
        self.replay_id = data.get("replay_id", "")
        self.queue_id = ""
        
        width = self.game_state.map[0]
        height = self.game_state.map[1]
        self.player.set_grid_dims(width, height)
        
        # Setup pygame window
        cell_size = self.player.cell_size
        screen_w = width * cell_size + 200
        screen_h = height * cell_size + 50
        self.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("Generals.io - Online Game")
        self.font = pygame.font.Font(None, 24)
        
        print(f"Game started! Map: {width}x{height}")
    
    def play_game(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        
        while True:
            # Handle input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return "quit"
                
                obs = self.game_state.get_observation()
                self.player.handle_event(event, obs)
            
            # Check for server updates
            try:
                event, data, _ = self.receive(timeout=0.01)
                
                if event == "game_update":
                    self.game_state.update(data)
                    
                    # Send queued action
                    action = self.player.get_action()
                    if action:
                        self.emit("attack", action)
                
                elif event == "game_won":
                    print(f"\n*** YOU WON! ***")
                    print(f"Replay: https://bot.generals.io/replays/{self.replay_id}")
                    return "won"
                
                elif event == "game_lost":
                    print(f"\nYou lost...")
                    print(f"Replay: https://bot.generals.io/replays/{self.replay_id}")
                    return "lost"
                    
            except:
                pass
            
            # Render
            self._render()
            clock.tick(30)
    
    def _render(self):
        """Render the game."""
        if not self.screen or not self.game_state:
            return
        
        self.screen.fill((40, 44, 52))
        
        obs = self.game_state.get_observation()
        cell_size = self.player.cell_size
        h, w = self.player.grid_dims
        
        # Draw grid
        for row in range(h):
            for col in range(w):
                x = col * cell_size
                y = row * cell_size
                
                # Determine cell color
                if obs.owned_cells[row, col]:
                    color = (70, 130, 180)  # Blue - yours
                elif obs.opponent_cells[row, col]:
                    color = (180, 70, 70)  # Red - enemy
                elif obs.mountains[row, col]:
                    color = (100, 100, 100)  # Gray - mountain
                elif obs.cities[row, col]:
                    color = (128, 128, 128)  # City
                elif obs.fog_cells[row, col]:
                    color = (50, 50, 50)  # Fog
                else:
                    color = (200, 200, 200)  # Neutral
                
                pygame.draw.rect(self.screen, color, (x, y, cell_size-1, cell_size-1))
                
                # Draw army count
                army = int(obs.armies[row, col])
                if army > 0 and not obs.fog_cells[row, col]:
                    text = self.font.render(str(army), True, (255, 255, 255))
                    text_rect = text.get_rect(center=(x + cell_size//2, y + cell_size//2))
                    self.screen.blit(text, text_rect)
                
                # Draw general indicator
                if obs.generals[row, col]:
                    pygame.draw.circle(self.screen, (255, 215, 0), 
                                     (x + cell_size//2, y + cell_size//2), 5)
        
        # Draw cursor
        crow, ccol = self.player.cursor
        rect = pygame.Rect(ccol * cell_size, crow * cell_size, cell_size-1, cell_size-1)
        color = (255, 200, 0) if self.player.split_mode else (255, 255, 255)
        pygame.draw.rect(self.screen, color, rect, 3)
        
        # Draw stats
        stats_x = w * cell_size + 10
        self.screen.blit(self.font.render(f"Army: {int(obs.owned_army_count)}", True, (255,255,255)), (stats_x, 10))
        self.screen.blit(self.font.render(f"Land: {int(obs.owned_land_count)}", True, (255,255,255)), (stats_x, 35))
        self.screen.blit(self.font.render(f"Queue: {len(self.player.action_queue)}", True, (255,255,0)), (stats_x, 60))
        self.screen.blit(self.font.render(f"Split: {'ON' if self.player.split_mode else 'OFF'}", True, (255,100,100) if self.player.split_mode else (100,255,100)), (stats_x, 85))
        
        # Controls hint
        self.screen.blit(self.font.render("WASD: move", True, (150,150,150)), (stats_x, 120))
        self.screen.blit(self.font.render("Space: split", True, (150,150,150)), (stats_x, 140))
        self.screen.blit(self.font.render("Q: clear", True, (150,150,150)), (stats_x, 160))
        self.screen.blit(self.font.render("E: undo", True, (150,150,150)), (stats_x, 180))
        
        pygame.display.flip()
    
    def leave_game(self):
        self.emit("leave_game")


def main():
    # 默认使用的用户ID
    DEFAULT_USER_ID = "Phantom_Bot_2025"
    
    print("=" * 50)
    print("  Generals.io - Online Play")
    print("=" * 50)
    print()
    
    user_id = DEFAULT_USER_ID
    print(f"User ID: {user_id}")
    print()
    print("Mode: 1v1 Matchmaking (auto-match with real players)")
    print()
    
    try:
        client = OnlineGameClient(user_id)
        client.join_1v1_queue()  # 自动匹配 1v1
        
        pygame.init()
        # Small waiting window
        screen = pygame.display.set_mode((400, 200))
        pygame.display.set_caption("Waiting for opponent...")
        font = pygame.font.Font(None, 32)
        
        if client.wait_for_game():
            result = client.play_game()
            print(f"\nGame result: {result}")
            time.sleep(2)
            client.leave_game()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
